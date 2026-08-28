from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    accuracy: float
    macro_f1: float


@dataclass
class TrainingRun:
    variant: str
    seed: int
    epochs: list[EpochMetrics]
    best_epoch: int
    stop_reason: str
    training_seconds: float

    def best(self) -> EpochMetrics:
        return max(self.epochs, key=lambda item: (item.macro_f1, -item.validation_loss))


def confusion_metrics(confusion: list[list[int]]) -> tuple[float, float]:
    scores: list[float] = []
    for index in range(len(confusion)):
        true_positive = confusion[index][index]
        false_positive = sum(row[index] for row in confusion) - true_positive
        false_negative = sum(confusion[index]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(len(confusion)))
    return (correct / total if total else 0.0, sum(scores) / len(scores))


def early_stop(epochs: list[EpochMetrics], patience: int, max_epochs: int) -> tuple[bool, str]:
    if not epochs:
        return False, "no_epoch"
    best_index = max(
        range(len(epochs)),
        key=lambda index: (epochs[index].macro_f1, -epochs[index].validation_loss),
    )
    if len(epochs) >= max_epochs:
        return True, "max_epochs"
    if len(epochs) - best_index > patience:
        return True, "early_stopping"
    return False, "continue"


def write_run_artifacts(
    run: TrainingRun, output_dir: Path, extra: dict[str, Any] | None = None
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "variant": run.variant,
        "seed": run.seed,
        "best_epoch": run.best().epoch,
        "stop_reason": run.stop_reason,
        "training_seconds": round(run.training_seconds, 6),
        "epochs": [asdict(item) for item in run.epochs],
    }
    if extra:
        payload.update(extra)
    (output_dir / "run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
