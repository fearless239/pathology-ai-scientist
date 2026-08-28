"""Deterministic, no-network demonstration artifacts for interviews and CI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .research_stages import RESEARCH_STAGES
from .framework import ResearchTaskConfig


DEMO_DIRECTION = (
    "Evaluate whether label smoothing improves PathMNIST macro-F1 over a fixed baseline."
)


def _write(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def build_demo(output_root: Path) -> dict[str, Any]:
    """Create a small static evidence bundle without data, network, GPU, or API keys."""
    root = output_root.resolve()
    config = ResearchTaskConfig(
        task_id="pathmnist-offline-demo",
        direction=DEMO_DIRECTION,
        dataset_adapter="pathmnist.dataset_adapter.DatasetAdapter",
        dataset_path=Path("<synthetic-fixture>"),
        # Keep the evidence content location-independent so two clean runs hash identically.
        output_root=Path("<demo-output>"),
    )
    hashes: dict[str, str] = {}
    hashes["config"] = _write(root / "task_config.json", config.as_dict())
    timeline = {
        "schema_version": 1,
        "mode": "offline_fixture",
        "network_used": False,
        "paid_calls": 0,
        "stages": [
            {"stage": stage, "status": "accepted", "attempts": 1}
            for stage in RESEARCH_STAGES
        ],
    }
    hashes["timeline"] = _write(root / "timeline.json", timeline)
    contract = {
        "schema_version": 1,
        "approved": True,
        "baseline": "cross_entropy",
        "intervention": "label_smoothing_0.1",
        "primary_metric": "macro_f1",
        "repeat_seeds": [7, 17, 27],
        "sealed_test_policy": "single evaluation after candidate freeze",
    }
    hashes["contract"] = _write(root / "research_contract.json", contract)
    evidence = {
        "schema_version": 1,
        "fixture_only": True,
        "baseline_macro_f1": {"mean": 0.842, "std": 0.006, "seeds": 3},
        "intervention_macro_f1": {"mean": 0.851, "std": 0.005, "seeds": 3},
        "delta": 0.009,
        "claim": "The fixture exercises evidence plumbing; it is not a scientific result.",
        "source": "deterministic synthetic fixture",
    }
    hashes["evidence"] = _write(root / "trusted_metrics.json", evidence)
    acceptance = {
        "schema_version": 1,
        "passed": True,
        "mode": "offline_fixture",
        "checks": {
            "research_contract": "passed",
            "split_isolation": "passed",
            "candidate_frozen_before_test": "passed",
            "metric_provenance": "passed",
            "publication_disclosure": "passed",
        },
        "limitations": ["synthetic fixture", "no clinical claim", "no peer review"],
    }
    hashes["acceptance"] = _write(root / "acceptance_report.json", acceptance)
    manifest = {"schema_version": 1, "artifacts": hashes}
    _write(root / "manifest.json", manifest)
    return {"output_root": str(root), "artifacts": hashes, "passed": True}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the deterministic offline interview demo")
    parser.add_argument("--output", type=Path, default=Path(".demo/pathmnist-offline"))
    args = parser.parse_args()
    print(json.dumps(build_demo(args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
