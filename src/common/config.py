"""Leitura e validação dos arquivos YAML em configs/."""
from dataclasses import dataclass

import yaml

from src.common.paths import CONFIGS_DIR

SPLIT_NAMES = {"train", "val", "test"}


def load_yaml(name: str) -> dict:
    """Lê configs/<name>.yaml e devolve um dict. Reutilizável por detector.yaml e classifier.yaml."""
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config não encontrada: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class DataConfig:
    seed: int
    datasets: list[str]
    img_exts: frozenset[str]
    split_ratios: dict[str, float]
    species_aliases: dict[str, str]


def load_data_config() -> DataConfig:
    raw = load_yaml("data")

    ratios = {k: float(v) for k, v in raw["split_ratios"].items()}
    if set(ratios) != SPLIT_NAMES:
        raise ValueError(f"split_ratios deve ter exatamente as chaves {sorted(SPLIT_NAMES)}")
    if any(r <= 0 for r in ratios.values()):
        raise ValueError("Todas as proporções de split_ratios devem ser maiores que 0")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios deve somar 1.0, soma atual: {sum(ratios.values())}")

    return DataConfig(
        seed=int(raw["seed"]),
        datasets=list(raw["datasets"]),
        img_exts=frozenset(ext.lower() for ext in raw["img_exts"]),
        split_ratios=ratios,
        species_aliases=dict(raw.get("species_aliases") or {}),
    )