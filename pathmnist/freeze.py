from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FreezeError(ValueError):
    pass


REQUIRED_APPROVAL = "I APPROVE ONE-TIME TEST EVALUATION"


@dataclass(frozen=True)
class FrozenCandidate:
    variant: str
    learning_rate: float
    weight_decay: float
    label_smoothing: float
    augmentation: bool
    multiscale: bool
    one_cycle: bool
    seeds: tuple[int, ...]
    dataset_sha256: str


def load_frozen_candidate(path: Path) -> FrozenCandidate:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise FreezeError("Unsupported frozen candidate schema")
    if raw.get("status") != "approved_frozen":
        raise FreezeError("Candidate is not approved and frozen")
    if raw.get("primary_metric") != "macro_f1":
        raise FreezeError("Frozen primary metric must be macro_f1")
    test_policy = raw.get("test_policy", {})
    if test_policy.get("evaluations_allowed") != 1:
        raise FreezeError("Exactly one test evaluation must be allowed")
    if test_policy.get("tuning_feedback_allowed") is not False:
        raise FreezeError("Test feedback must be disabled")
    expected = {
        "augmentation": False,
        "multiscale": False,
        "one_cycle": True,
        "label_smoothing": 0.1,
        "learning_rate": 0.001,
        "weight_decay": 1e-05,
    }
    if raw.get("hyperparameters") != expected:
        raise FreezeError("Frozen hyperparameters differ from the approved candidate")
    seeds = tuple(int(seed) for seed in raw.get("seeds", []))
    if seeds != (7, 17, 27):
        raise FreezeError("Frozen seeds must be 7, 17, and 27")
    if raw.get("model") != "SmallResNet" or raw.get("variant") != "optimization":
        raise FreezeError("Frozen model or variant differs from the approved candidate")
    return FrozenCandidate(
        variant="optimization",
        learning_rate=0.001,
        weight_decay=1e-05,
        label_smoothing=0.1,
        augmentation=False,
        multiscale=False,
        one_cycle=True,
        seeds=seeds,
        dataset_sha256=str(raw.get("dataset_sha256", "")),
    )


def checkpoint_path(final_root: Path, seed: int) -> Path:
    return final_root / "optimization" / f"seed_{seed}" / "checkpoint.pt"


def require_ready_checkpoints(
    candidate: FrozenCandidate, dataset_sha256: str, final_root: Path
) -> list[Path]:
    if candidate.dataset_sha256 != dataset_sha256:
        raise FreezeError("Frozen dataset hash does not match experiment config")
    checkpoints = [checkpoint_path(final_root, seed) for seed in candidate.seeds]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FreezeError(f"Final checkpoints incomplete; missing: {missing}")
    return checkpoints


def write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FreezeError(f"One-time output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
