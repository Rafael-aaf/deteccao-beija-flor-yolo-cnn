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
    det_confidence: float
    uncertain: bool


class HummingbirdPipeline:
    def __init__(self, cfg: dict, device: torch.device):
        self.device = device
        self.det_conf = cfg["det_conf"]
        self.min_species_conf = cfg["min_species_conf"]
        self.detector = YOLO(str(WEIGHTS_DIR / cfg["detector"]))
        self.classifier, self.ckpt = load_classifier(WEIGHTS_DIR / cfg["classifier"], device)
        self.votes = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))

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
        predictions = classify_boxes(self.classifier, self.ckpt, image, boxes, self.device)

        detections = []
        for track_id, box, det_conf, (species, conf) in zip(track_ids, boxes, det_confs, predictions):
            if track_id is not None:
                species, conf = self._vote(track_id, species, conf)
            detections.append(
                Detection(
                    box=tuple(int(v) for v in box),
                    track_id=track_id,
                    species=species,
                    confidence=conf,
                    det_confidence=det_conf,
                    uncertain=conf < self.min_species_conf,
                )
            )
        return detections

    def _vote(self, track_id: int, species: str, conf: float) -> tuple[str, float]:
        bird = self.votes[track_id]
        bird[species][0] += conf
        bird[species][1] += 1
        winner = max(bird, key=lambda s: bird[s][0])
        return winner, bird[winner][0] / bird[winner][1]