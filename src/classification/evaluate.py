from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.classification.dataset import build_dataset
from src.classification.engine import collect_predictions
from src.classification.metrics import confusion_matrix, per_class_metrics
from src.classification.predict import load_classifier
from src.common.config import load_yaml
from src.common.paths import ROOT, RUNS_DIR, SPLITS_CSV, WEIGHTS_DIR


def save_confusion_matrix(cm, classes: list[str], path: Path) -> None:
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black")
    ax.set(xlabel="previsto", ylabel="real", title="Matriz de confusão (teste)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_yaml("classifier")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{cfg['name']}_{cfg['arch']}_{cfg['version']}"
    model, ckpt = load_classifier(WEIGHTS_DIR / f"{run_name}.pt", device)
    classes = ckpt["classes"]

    dataset = build_dataset("test", cfg)
    if dataset.classes != classes:
        raise ValueError("As classes do conjunto de teste diferem das do modelo")
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False)

    _, probs, targets = collect_predictions(model, loader, device)
    preds, confidence = probs.argmax(axis=1), probs.max(axis=1)

    cm = confusion_matrix(targets, preds, len(classes))
    precision, recall, f1 = per_class_metrics(cm)
    print(
        f"Teste: {len(targets)} imagens | acurácia {(preds == targets).mean():.3f} | "
        f"F1 macro {f1.mean():.3f} | modelo da época {ckpt['epoch']}\n"
    )

    print(f"{'Espécie':<28} {'prec':>6} {'rec':>6} {'f1':>6} {'n':>4}")
    for i, name in enumerate(classes):
        print(f"{name:<28} {precision[i]:>6.3f} {recall[i]:>6.3f} {f1[i]:>6.3f} {cm[i].sum():>4}")

    errors = [
        (cm[i, j], classes[i], classes[j])
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and cm[i, j] > 0
    ]
    print("\nConfusões mais frequentes (real -> previsto):")
    for n, true, pred in sorted(errors, reverse=True)[:8]:
        print(f"  {true} -> {pred}: {n}")

    results = pd.DataFrame({
        "stem": [Path(path).stem for path, _ in dataset.samples],
        "true": [classes[i] for i in targets],
        "pred": [classes[i] for i in preds],
        "confidence": confidence,
    })
    results["correct"] = results["true"] == results["pred"]
    meta = pd.read_csv(SPLITS_CSV)[["stem", "stage"]]
    meta["stage"] = meta["stage"].fillna("sem_estagio")
    results = results.merge(meta, on="stem")

    print("\nAcurácia por estágio (M: macho adulto, F: fêmea ou imaturo):")
    print(results.groupby("stage")["correct"].agg(acuracia="mean", n="count").round(3))

    print("\nConfiança média da previsão:")
    print(results.groupby("correct")["confidence"].mean().round(3).rename({False: "erros", True: "acertos"}))

    print("\nSe só aceitarmos previsões acima de uma confiança mínima:")
    for threshold in (0.5, 0.7, 0.9):
        kept = results[results["confidence"] >= threshold]
        print(
            f"  >= {threshold:.1f}: {len(kept) / len(results):.1%} das imagens, "
            f"acurácia {kept['correct'].mean():.3f}"
        )

    out_dir = RUNS_DIR / "classify" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "test_predictions.csv", index=False)
    save_confusion_matrix(cm, classes, out_dir / "confusion_matrix_test.png")
    print(f"\nSalvos em {out_dir.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()