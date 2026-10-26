from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.classification.dataset import build_dataset
from src.classification.predict import load_classifier
from src.common.config import load_yaml
from src.common.paths import ROOT, RUNS_DIR, SPLITS_CSV, WEIGHTS_DIR
from src.explainability.gradcam import GradCAM, overlay


def save_panels(panels: list, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.4))
    for ax, (image, subtitle) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(subtitle, fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    cfg = load_yaml("classifier")
    gcfg = load_yaml("gradcam")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{cfg['name']}_{cfg['arch']}_{cfg['version']}"
    model, ckpt = load_classifier(WEIGHTS_DIR / f"{run_name}.pt", device)
    classes = ckpt["classes"]

    dataset = build_dataset(gcfg["split"], cfg)
    if dataset.classes != classes:
        raise ValueError("As classes do conjunto diferem das do modelo")

    gradcam = GradCAM(model, model.get_submodule(gcfg["target_layer"]))
    size = cfg["img_size"]
    out_dir = RUNS_DIR / "gradcam" / run_name
    show = lambda name: name.replace("_", " ")

    rows = []
    for index, (path, target) in enumerate(tqdm(dataset.samples)):
        tensor = dataset[index][0].unsqueeze(0).to(device)
        image = np.asarray(Image.open(path).convert("RGB").resize((size, size))) / 255.0

        cam, probs = gradcam(tensor)
        pred = int(probs.argmax())
        confidence = float(probs[pred])
        correct = pred == target

        panels = [
            (image, "original"),
            (overlay(image, cam, gcfg["alpha"], gcfg["colormap"]), "Grad-CAM: previsto"),
        ]
        if not correct and gcfg["true_class_on_errors"]:
            cam_true, _ = gradcam(tensor, target)
            panels.append((overlay(image, cam_true, gcfg["alpha"], gcfg["colormap"]), "Grad-CAM: real"))

        stem = Path(path).stem
        species_dir = out_dir / classes[target]
        species_dir.mkdir(parents=True, exist_ok=True)
        title = f"real: {show(classes[target])} | previsto: {show(classes[pred])} ({confidence:.0%})"
        save_panels(panels, title, species_dir / f"{'ok' if correct else 'erro'}_{stem}.png")

        rows.append({
            "stem": stem,
            "true": classes[target],
            "pred": classes[pred],
            "confidence": confidence,
            "correct": correct,
        })

    results = pd.DataFrame(rows)
    meta = pd.read_csv(SPLITS_CSV)[["stem", "stage"]]
    meta["stage"] = meta["stage"].fillna("sem_estagio")
    results = results.merge(meta, on="stem")
    results.to_csv(out_dir / "gradcam_predictions.csv", index=False)

    print(f"\n{len(results)} imagens | acurácia {results['correct'].mean():.3f}")
    print(f"Salvos em {out_dir.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()