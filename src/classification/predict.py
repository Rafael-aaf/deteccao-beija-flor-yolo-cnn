from pathlib import Path

import torch
from PIL import Image
import numpy as np

from src.classification.dataset import eval_transform
from src.classification.model import build_model
from src.classification.preprocessing import square_crop


def load_classifier(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def classify_boxes(model, ckpt, image: Image.Image, boxes: list, device: torch.device) -> list[np.ndarray]:
    if not boxes:
        return []
    transform = eval_transform(ckpt["img_size"])
    crops = [square_crop(image, box, ckpt["crop_margin"]) for box in boxes]
    batch = torch.stack([transform(crop) for crop in crops]).to(device)
    return list(model(batch).softmax(dim=1).cpu().numpy())