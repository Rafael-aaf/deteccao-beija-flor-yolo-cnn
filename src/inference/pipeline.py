from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from src.classification.predict import classify_boxes, load_classifier
from src.common.paths import WEIGHTS_DIR


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    track_id: Optional[int]
    species: str
    confidence: float
    top2: tuple[tuple[str, float], ...]
    det_confidence: float
    uncertain: bool


class HummingbirdPipeline:
    def __init__(self, cfg: dict, device: torch.device):
        self.device = device
        self.det_conf = cfg["det_conf"]
        self.min_species_conf = cfg["min_species_conf"]
        self.detector = YOLO(str(WEIGHTS_DIR / cfg["detector"]))
        self.classifier, self.ckpt = load_classifier(WEIGHTS_DIR / cfg["classifier"], device)
        n_classes = len(self.ckpt["classes"])
        self.votes = defaultdict(lambda: [np.zeros(n_classes), 0])

    @property
    def n_tracks(self) -> int:
        return len(self.votes)

    def process(self, frame: np.ndarray) -> list[Detection]:
        result = self.detector.track(frame, persist=True, conf=self.det_conf, verbose=False)[0]
        boxes = result.boxes.xyxy.tolist()
        if not boxes:
            return []

        track_ids = result.boxes.id.int().tolist() if result.boxes.id is not None else [None] * len(boxes)
        det_confs = result.boxes.conf.tolist()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        probabilities = classify_boxes(self.classifier, self.ckpt, image, boxes, self.device)

        detections = []
        for track_id, box, det_conf, probs in zip(track_ids, boxes, det_confs, probabilities):
            if track_id is not None:
                probs = self._vote(track_id, probs)
            top2 = self._top2(probs)
            species, conf = top2[0]
            detections.append(
                Detection(
                    box=tuple(int(v) for v in box),
                    track_id=track_id,
                    species=species,
                    confidence=conf,
                    top2=top2,
                    det_confidence=det_conf,
                    uncertain=conf < self.min_species_conf,
                )
            )
        return detections

    def _vote(self, track_id: int, probs: np.ndarray) -> np.ndarray:
        entry = self.votes[track_id]
        entry[0] += probs
        entry[1] += 1
        return entry[0] / entry[1]

    def _top2(self, probs: np.ndarray) -> tuple[tuple[str, float], ...]:
        best = np.argsort(probs)[::-1][:2]
        return tuple((self.ckpt["classes"][i], float(probs[i])) for i in best)

    def summary(self) -> dict[int, tuple[tuple[str, float], ...]]:
        return {track_id: self._top2(total / n) for track_id, (total, n) in self.votes.items()}