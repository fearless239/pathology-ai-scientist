from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev


def aggregate_variant(variant_dir: Path) -> dict[str, object]:
    best_values = []
    for run_path in sorted(variant_dir.glob("seed_*/run.json")):
        payload = json.loads(run_path.read_text())
        best_values.append(payload["epochs"][payload["best_epoch"] - 1]["macro_f1"])
    return {
        "runs": len(best_values),
        "macro_f1_mean": mean(best_values),
        "macro_f1_std": stdev(best_values) if len(best_values) > 1 else 0.0,
    }
