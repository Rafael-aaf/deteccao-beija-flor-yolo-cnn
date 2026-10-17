from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
YOLO_DIR = DATA_DIR / "yolo"
CROPS_DIR = DATA_DIR / "crops"
SPLITS_CSV = DATA_DIR / "splits.csv"
WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR = ROOT / "runs"