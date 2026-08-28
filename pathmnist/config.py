from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class VariantConfig:
    name: str
    augmentation: bool
    optimization: bool
    multiscale: bool


@dataclass(frozen=True)
class ExperimentConfig:
    primary_metric: str
    seeds: tuple[int, ...]
    epochs: int
    batch_size: int
    num_workers: int
    early_stop_patience: int
    baseline_learning_rate: float
    learning_rates: tuple[float, ...]
    weight_decays: tuple[float, ...]
    variants: tuple[VariantConfig, ...]
    ablations: tuple[VariantConfig, ...]


@dataclass(frozen=True)
class DatasetConfig:
    path: Path
    sha256: str
    expected_splits: dict[str, int]
    classes: int


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig
    experiment: ExperimentConfig
    raw: dict[str, Any]


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ConfigError(f"Missing {context}.{key}") from exc


def _variant(raw: dict[str, Any], context: str) -> VariantConfig:
    return VariantConfig(
        name=str(_require(raw, "name", context)),
        augmentation=bool(_require(raw, "augmentation", context)),
        optimization=bool(_require(raw, "optimization", context)),
        multiscale=bool(_require(raw, "multiscale", context)),
    )


def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping")
    dataset_raw = _require(raw, "dataset", "root")
    experiment_raw = _require(raw, "experiment", "root")
    variants_raw = _require(raw, "variants", "root")
    ablations_raw = _require(raw, "ablations", "root")
    tuning_raw = _require(experiment_raw, "tuning_grid", "experiment")
    expected_splits_raw = _require(dataset_raw, "expected_splits", "dataset")
    dataset = DatasetConfig(
        path=Path(_require(dataset_raw, "path", "dataset")),
        sha256=str(_require(dataset_raw, "sha256", "dataset")).lower(),
        expected_splits={name: int(count) for name, count in expected_splits_raw.items()},
        classes=int(_require(dataset_raw, "classes", "dataset")),
    )
    experiment = ExperimentConfig(
        primary_metric=str(_require(experiment_raw, "primary_metric", "experiment")),
        seeds=tuple(int(seed) for seed in _require(experiment_raw, "seeds", "experiment")),
        epochs=int(_require(experiment_raw, "epochs", "experiment")),
        batch_size=int(_require(experiment_raw, "batch_size", "experiment")),
        num_workers=int(_require(experiment_raw, "num_workers", "experiment")),
        early_stop_patience=int(
            _require(experiment_raw, "early_stop_patience", "experiment")
        ),
        baseline_learning_rate=float(
            _require(experiment_raw, "baseline_learning_rate", "experiment")
        ),
        learning_rates=tuple(
            float(value)
            for value in _require(tuning_raw, "learning_rate", "experiment.tuning_grid")
        ),
        weight_decays=tuple(
            float(value)
            for value in _require(tuning_raw, "weight_decay", "experiment.tuning_grid")
        ),
        variants=tuple(_variant(item, "variants") for item in variants_raw),
        ablations=tuple(_variant(item, "ablations") for item in ablations_raw),
    )
    _validate(experiment)
    return AppConfig(dataset=dataset, experiment=experiment, raw=raw)


def _validate(config: ExperimentConfig) -> None:
    if config.primary_metric != "macro_f1":
        raise ConfigError("M4 primary metric must be macro_f1")
    if len(config.seeds) != 3 or len(set(config.seeds)) != 3:
        raise ConfigError("Exactly three distinct seeds are required")
    if config.epochs < 1 or config.batch_size < 1 or config.num_workers < 0:
        raise ConfigError(
            "Epochs and batch size must be positive; workers must be non-negative"
        )
    names = [variant.name for variant in config.variants + config.ablations]
    if len(names) != len(set(names)):
        raise ConfigError("Variant and ablation names must be unique")
    required = {"baseline", "augmentation", "optimization", "multiscale", "combined"}
    if not required.issubset(set(names)):
        raise ConfigError(f"Missing required variants: {sorted(required - set(names))}")
