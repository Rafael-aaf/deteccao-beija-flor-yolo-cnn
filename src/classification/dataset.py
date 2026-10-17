from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.common.paths import CROPS_DIR

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform(img_size: int, aug: dict) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=tuple(aug["scale"]), ratio=tuple(aug["ratio"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(aug["rotation"]),
        transforms.ColorJitter(
            brightness=aug["brightness"], contrast=aug["contrast"], saturation=aug["saturation"]
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_dataset(split: str, cfg: dict) -> ImageFolder:
    if split == "train":
        transform = train_transform(cfg["img_size"], cfg["augmentation"])
    else:
        transform = eval_transform(cfg["img_size"])
    return ImageFolder(CROPS_DIR / split, transform=transform)