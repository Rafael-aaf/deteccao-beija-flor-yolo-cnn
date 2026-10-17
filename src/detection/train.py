from pathlib import Path

from ultralytics import YOLO

from src.common.config import load_yaml
from src.common.paths import ROOT, RUNS_DIR, WEIGHTS_DIR, YOLO_DIR


def main() -> None:
    cfg = load_yaml("detector")

    data_yaml = YOLO_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml.relative_to(ROOT).as_posix()} não encontrado. "
            "Rode python -m src.data.build_yolo_dataset primeiro."
        )

    WEIGHTS_DIR.mkdir(exist_ok=True)
    model = YOLO(str(WEIGHTS_DIR / cfg["model"]))
    model.train(
        data=str(data_yaml),
        project=str(RUNS_DIR / "detect"),
        name=f"{cfg['name']}_{Path(cfg['model']).stem}",
        seed=cfg["seed"],
        **cfg["train"],
    )

    print(f"\nMelhor modelo: {Path(model.trainer.best).relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()