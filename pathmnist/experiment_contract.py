from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ExperimentContractError(RuntimeError):
    pass


TEST_METRIC_ALIASES = {
    "validation_accuracy": "test_accuracy",
    "validation_inference_seconds": "dynamic_inference_seconds",
}


def canonicalize_metrics(metrics: dict[str, Any], split: str) -> dict[str, Any]:
    """Return canonical metric names without mutating historical evidence."""
    normalized = dict(metrics)
    if split == "test":
        for legacy, canonical in TEST_METRIC_ALIASES.items():
            if legacy in normalized:
                if canonical in normalized and normalized[canonical] != normalized[legacy]:
                    raise ExperimentContractError(
                        f"Conflicting legacy/canonical test metric: {legacy}/{canonical}"
                    )
                normalized[canonical] = normalized.pop(legacy)
        # Frozen research programs receive the sealed test arrays through the
        # read-only `validation_*` interface they were trained against. Their
        # arbitrary output keys therefore often retain validation/val tokens.
        # At the independent evaluator boundary those tokens describe the
        # sealed test split and must be canonicalized before paper generation.
        for legacy in list(normalized):
            canonical = re.sub(r"(^|_)validation(?=_|$)", r"\1test", legacy)
            canonical = re.sub(r"(^|_)val(?=_|$)", r"\1test", canonical)
            if canonical == legacy:
                continue
            if canonical in normalized and normalized[canonical] != normalized[legacy]:
                raise ExperimentContractError(
                    f"Conflicting legacy/canonical test metric: {legacy}/{canonical}"
                )
            normalized[canonical] = normalized.pop(legacy)
    return normalized


@dataclass(frozen=True)
class ExperimentResult:
    method_name: str
    code_sha256: str
    seed: int
    metrics: dict[str, Any]
    resource_usage: dict[str, Any]
    artifacts: dict[str, str]
    parent_experiment_id: str | None = None
    schema_version: int = 2
    status: str = "completed"
    split: str = "validation"
    test_data_accessed: bool = False
    training_history: list[dict[str, Any]] | None = None
    predictions: list[int] | None = None
    probabilities: list[list[float]] | None = None
    targets: list[int] | None = None
    sample_ids: list[str] | None = None
    class_names: list[str] | None = None

    def validate(self, allow_test: bool = False) -> None:
        if self.schema_version not in {1, 2} or self.status != "completed":
            raise ExperimentContractError("Only completed schema-v1/v2 experiment results are accepted")
        if self.split == "test" and not allow_test:
            raise ExperimentContractError("Research experiments may not report test metrics")
        if self.test_data_accessed and not allow_test:
            raise ExperimentContractError("Research experiment accessed sealed test data")
        if not self.method_name.strip() or not self.metrics:
            raise ExperimentContractError("Method name and at least one metric are required")
        if self.schema_version >= 2 and self.split == "test":
            legacy = sorted(TEST_METRIC_ALIASES.keys() & self.metrics.keys())
            if legacy:
                raise ExperimentContractError(f"Schema-v2 test result uses legacy metrics: {legacy}")
        if len(self.code_sha256) != 64:
            raise ExperimentContractError("Invalid code SHA-256")
        lengths = [
            len(value)
            for value in (self.predictions, self.probabilities, self.targets, self.sample_ids)
            if value is not None
        ]
        if lengths and any(value != lengths[0] for value in lengths):
            raise ExperimentContractError("Predictions, probabilities, and targets must align")
        if self.probabilities is not None and self.class_names is not None:
            if any(len(row) != len(self.class_names) for row in self.probabilities):
                raise ExperimentContractError("Probability width differs from class_names")
        if self.sample_ids is not None and len(set(self.sample_ids)) != len(self.sample_ids):
            raise ExperimentContractError("Sample IDs must be unique")

    def write(self, path: Path, allow_test: bool = False) -> Path:
        self.validate(allow_test=allow_test)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path, allow_test: bool = False) -> "ExperimentResult":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("schema_version") == 1 and raw.get("split") == "test":
                raw["metrics"] = canonicalize_metrics(raw.get("metrics", {}), "test")
            result = cls(**raw)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExperimentContractError(f"Invalid experiment result: {path}") from exc
        result.validate(allow_test=allow_test)
        return result


def code_sha256(code: str) -> str:
    canonical = code.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
