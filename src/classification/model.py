import torch.nn as nn
from torchvision import models

HEADS = {
    "efficientnet_b0": "classifier.1",
    "resnet50": "fc",
    "convnext_tiny": "classifier.2",
}


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model = models.get_model(arch, weights="DEFAULT" if pretrained else None)
    parent_name, _, attr = HEADS[arch].rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    old = getattr(parent, attr)
    setattr(parent, attr, nn.Linear(old.in_features, num_classes))
    return model


def split_parameters(model: nn.Module, arch: str) -> tuple[list, list]:
    prefix = f"{HEADS[arch]}."
    backbone, head = [], []
    for name, param in model.named_parameters():
        (head if name.startswith(prefix) else backbone).append(param)
    return backbone, head


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> None:
    backbone, _ = split_parameters(model, arch)
    for param in backbone:
        param.requires_grad = trainable