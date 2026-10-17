import csv
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.classification.dataset import build_dataset
from src.classification.engine import collect_predictions, train_epoch
from src.classification.metrics import confusion_matrix, per_class_metrics
from src.classification.model import build_model, set_backbone_trainable, split_parameters
from src.common.config import load_yaml
from src.common.paths import CROPS_DIR, ROOT, RUNS_DIR, WEIGHTS_DIR
from src.common.seed import set_seed


def class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(targets, minlength=num_classes)
    return torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32)


def make_loader(dataset, tcfg: dict, shuffle: bool) -> DataLoader:
    workers = tcfg["num_workers"]
    return DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
    )


def build_optimizer(model, arch: str, tcfg: dict, backbone_trainable: bool):
    backbone, head = split_parameters(model, arch)
    groups = [{"params": head, "lr": tcfg["lr_head"]}]
    if backbone_trainable:
        groups.append({"params": backbone, "lr": tcfg["lr_backbone"]})
    return torch.optim.AdamW(groups, weight_decay=tcfg["weight_decay"])


def main() -> None:
    cfg = load_yaml("classifier")
    tcfg = cfg["train"]
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not (CROPS_DIR / "train").exists():
        raise FileNotFoundError(
            f"{CROPS_DIR.relative_to(ROOT).as_posix()} não encontrado. "
            "Rode python -m src.data.make_crops primeiro."
        )

    train_set, val_set = build_dataset("train", cfg), build_dataset("val", cfg)
    classes = train_set.classes
    train_loader = make_loader(train_set, tcfg, shuffle=True)
    val_loader = make_loader(val_set, tcfg, shuffle=False)

    model = build_model(cfg["arch"], len(classes)).to(device)
    weights = class_weights(train_set.targets, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    run_name = f"{cfg['name']}_{cfg['arch']}_{cfg['version']}"
    checkpoint_path = WEIGHTS_DIR / f"{run_name}.pt"
    history_path = RUNS_DIR / "classify" / run_name / "history.csv"
    WEIGHTS_DIR.mkdir(exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    phases = [
        ("cabeça", tcfg["epochs_head"], False),
        ("completo", tcfg["epochs_finetune"], True),
    ]
    total_epochs = tcfg["epochs_head"] + tcfg["epochs_finetune"]
    fields = ["epoch", "phase", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr", "seconds"]

    epoch, best_epoch, best_f1, stale = 0, 0, -1.0, 0
    print(f"Dispositivo: {device} | classes: {len(classes)} | treino: {len(train_set)} | val: {len(val_set)}\n")

    with open(history_path, "w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fields)
        writer.writeheader()

        for phase, phase_epochs, backbone_trainable in phases:
            set_backbone_trainable(model, cfg["arch"], backbone_trainable)
            optimizer = build_optimizer(model, cfg["arch"], tcfg, backbone_trainable)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase_epochs)

            for _ in range(phase_epochs):
                epoch += 1
                start = time.time()
                lr = optimizer.param_groups[0]["lr"]

                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device, freeze_bn=not backbone_trainable
                )
                val_loss, probs, targets = collect_predictions(model, val_loader, device, criterion)
                preds = probs.argmax(axis=1)
                val_acc = float((preds == targets).mean())
                _, _, f1 = per_class_metrics(confusion_matrix(targets, preds, len(classes)))
                val_f1 = float(f1.mean())
                scheduler.step()
                seconds = time.time() - start

                improved = val_f1 > best_f1
                if improved:
                    best_f1, best_epoch, stale = val_f1, epoch, 0
                    torch.save(
                        {
                            "arch": cfg["arch"],
                            "classes": classes,
                            "img_size": cfg["img_size"],
                            "crop_margin": cfg["crop"]["margin"],
                            "state_dict": model.state_dict(),
                            "epoch": epoch,
                            "val_f1": val_f1,
                        },
                        checkpoint_path,
                    )
                elif backbone_trainable:
                    stale += 1

                writer.writerow({
                    "epoch": epoch, "phase": phase, "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1, "lr": lr, "seconds": seconds,
                })
                history_file.flush()

                print(
                    f"Época {epoch:02d}/{total_epochs} [{phase}] "
                    f"treino loss {train_loss:.3f} acc {train_acc:.3f} | "
                    f"val loss {val_loss:.3f} acc {val_acc:.3f} F1 {val_f1:.3f} | "
                    f"lr {lr:.1e} | {seconds:.0f}s{' *' if improved else ''}"
                )

                if stale >= tcfg["patience"]:
                    print(f"\nParada antecipada: sem melhora no F1 macro há {stale} épocas")
                    break

    print(f"\nMelhor época: {best_epoch} (F1 macro no val: {best_f1:.3f})")
    print(f"Modelo salvo em {checkpoint_path.relative_to(ROOT).as_posix()}")
    print(f"Histórico em {history_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()