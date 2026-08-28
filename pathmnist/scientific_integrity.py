from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

class IntegrityError(RuntimeError):
    """Raised when evidence cannot support a formal scientific output."""


EVALUATOR_VERSION = "p0-classification-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetExecutionReceipt:
    schema_version: int
    dataset_profile_sha256: str
    mounted_source: str
    split: str
    sample_ids_consumed: list[str]
    samples_consumed: int
    class_count_observed: int
    class_ids_observed: list[int]
    source_files: list[str]
    array_keys_read: list[str]
    code_sha256: str
    recorded_by: str = "trusted-runner"

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


def _profile_samples(profile: dict[str, Any], split: str) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    labels = profile.get("label_mapping") or {
        str(name): index for index, name in enumerate(profile.get("classes", []))
    }
    for sample in profile.get("samples", []):
        if sample.get("split") != split:
            continue
        sample_id = str(sample.get("id", ""))
        label = str(sample.get("label", ""))
        if not sample_id or label not in labels:
            raise IntegrityError("Dataset profile contains an invalid sample record")
        mapping[sample_id] = {**sample, "class_id": int(labels[label])}
    return mapping


def validate_sample_evidence(
    profile: dict[str, Any], split: str, sample_ids: list[str], targets: list[int],
    predictions: list[int], probabilities: list[list[float]] | None = None,
    *, require_complete_split: bool = True,
) -> list[dict[str, Any]]:
    if not sample_ids or not predictions or not targets:
        raise IntegrityError("Predictions, targets, and sample_ids are required")
    if not (len(sample_ids) == len(targets) == len(predictions)):
        raise IntegrityError("Per-sample evidence lengths differ")
    if len(set(sample_ids)) != len(sample_ids):
        raise IntegrityError("Duplicate sample IDs are forbidden")
    expected = _profile_samples(profile, split)
    if not expected:
        raise IntegrityError(f"Dataset profile has no {split!r} samples")
    unknown = sorted(set(sample_ids) - set(expected))
    missing = sorted(set(expected) - set(sample_ids))
    if unknown:
        raise IntegrityError(f"Unknown sample IDs: {unknown[:5]}")
    if require_complete_split and missing:
        raise IntegrityError(f"Missing {split} samples: {missing[:5]}")
    class_count = len(profile.get("classes") or profile.get("label_mapping") or [])
    if class_count < 2:
        raise IntegrityError("Classification requires at least two profile classes")
    rows = []
    for sample_id, target, prediction in zip(sample_ids, targets, predictions):
        if isinstance(target, bool) or int(target) != expected[sample_id]["class_id"]:
            raise IntegrityError(f"Target does not match trusted profile for sample {sample_id}")
        if not 0 <= int(prediction) < class_count:
            raise IntegrityError(f"Prediction is outside the class range for sample {sample_id}")
        rows.append(expected[sample_id])
    if probabilities is not None:
        if len(probabilities) != len(rows):
            raise IntegrityError("Probability rows do not align with samples")
        for index, values in enumerate(probabilities):
            if len(values) != class_count or not all(math.isfinite(float(v)) for v in values):
                raise IntegrityError(f"Invalid probability row {index}")
            if any(float(v) < 0 for v in values) or not math.isclose(sum(map(float, values)), 1.0, abs_tol=1e-4):
                raise IntegrityError(f"Probability row {index} is not normalized")
    return rows


class TrustedMetricEvaluator:
    def evaluate(self, predictions: list[int], targets: list[int], class_count: int) -> dict[str, Any]:
        if class_count < 2 or not predictions or len(predictions) != len(targets):
            raise IntegrityError("Trusted metrics require aligned non-empty classification outputs")
        confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
        for target, prediction in zip(targets, predictions):
            if not 0 <= int(target) < class_count or not 0 <= int(prediction) < class_count:
                raise IntegrityError("Metric input contains an out-of-range class")
            confusion[int(target)][int(prediction)] += 1
        per_class = []
        for class_id in range(class_count):
            tp = confusion[class_id][class_id]
            support = sum(confusion[class_id])
            predicted = sum(row[class_id] for row in confusion)
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class.append({"class_id": class_id, "precision": precision, "recall": recall, "f1": f1, "support": support})
        total = len(targets)
        return {
            "accuracy": sum(confusion[index][index] for index in range(class_count)) / total,
            "macro_f1": sum(row["f1"] for row in per_class) / class_count,
            "weighted_f1": float(sum(row["f1"] * row["support"] for row in per_class) / total),
            "per_class": per_class,
            "confusion_matrix": confusion,
        }


def record_trusted_evaluation(
    *, profile_path: Path, split: str, sample_ids: list[str], targets: list[int],
    predictions: list[int], probabilities: list[list[float]] | None, code_sha256: str,
    output_dir: Path, mounted_source: str = "/dataset", reported_metrics: dict[str, Any] | None = None,
    require_complete_split: bool = True,
    receipt_split: str | None = None,
) -> tuple[DatasetExecutionReceipt, dict[str, Any], dict[str, Any]]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    rows = validate_sample_evidence(profile, split, sample_ids, targets, predictions, probabilities, require_complete_split=require_complete_split)
    class_count = len(profile.get("classes") or profile.get("label_mapping") or [])
    metrics = TrustedMetricEvaluator().evaluate(predictions, targets, class_count)
    receipt = DatasetExecutionReceipt(
        1, _sha256(profile_path), mounted_source, receipt_split or split, list(sample_ids), len(sample_ids),
        len(set(map(int, targets))), sorted(set(map(int, targets))),
        sorted({str(row.get("path")) for row in rows if row.get("path")}),
        sorted({str(row.get("array_key")) for row in rows if row.get("array_key")}), code_sha256,
    )
    compared = reported_metrics or {}
    mismatches = {}
    for name in ("accuracy", "macro_f1", "weighted_f1", "confusion_matrix"):
        if name in compared and compared[name] != metrics[name]:
            mismatches[name] = {"reported": compared[name], "trusted": metrics[name]}
    risks = []
    if predictions == targets:
        risks.append("predictions_identical_to_targets")
    if set(targets) != set(range(class_count)):
        risks.append("not_all_profile_classes_observed")
    if mismatches:
        risks.append("reported_metrics_disagree_with_trusted_metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt.write(output_dir / "dataset_execution_receipt.json")
    (output_dir / "trusted_metrics.json").write_text(json.dumps({"schema_version": 1, "evaluator_version": EVALUATOR_VERSION, "metrics": metrics}, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": 1, "evaluator_version": EVALUATOR_VERSION,
        "dataset_profile_sha256": receipt.dataset_profile_sha256, "code_sha256": code_sha256,
        "split": receipt_split or split, "sample_count": len(sample_ids), "class_count": class_count,
        "duplicate_sample_ids": [], "missing_sample_ids": [], "metric_mismatches": mismatches,
        "risks": risks, "calculated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metric_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return receipt, metrics, provenance


def validate_no_synthetic_dataset(code: str) -> None:
    """Reject obvious whole-dataset substitution without banning normal randomness."""
    tree = ast.parse(code)
    dataset_names = {"data", "dataset", "train_data", "train_dataset", "validation_data", "val_data", "test_data", "images", "inputs", "x_train", "x_val", "x_test"}
    random_calls = {"rand", "randn", "randint", "random", "normal", "uniform", "default_rng", "random_sample"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "FakeData":
            raise IntegrityError("FakeData cannot replace the mounted research dataset")
        if isinstance(node, ast.Attribute) and node.attr == "download" and isinstance(node.ctx, ast.Load):
            raise IntegrityError("Generated experiments may not download datasets")
        if isinstance(node, ast.Call) and any(
            keyword.arg == "download" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
            for keyword in node.keywords
        ):
            raise IntegrityError("Generated experiments may not download datasets")
        if isinstance(node, ast.ClassDef) and any(
            (isinstance(base, ast.Name) and base.id == "Dataset")
            or (isinstance(base, ast.Attribute) and base.attr == "Dataset")
            for base in node.bases
        ):
            getitem = next((item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__getitem__"), None)
            if getitem and any(
                isinstance(item, ast.Call)
                and ((isinstance(item.func, ast.Attribute) and item.func.attr in random_calls) or (isinstance(item.func, ast.Name) and item.func.id in random_calls))
                for item in ast.walk(getitem)
            ):
                raise IntegrityError("Custom Dataset returns randomly generated samples")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            value = node.value
            if names & dataset_names and isinstance(value, ast.Call):
                func = value.func
                name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
                if name in random_calls:
                    raise IntegrityError(f"Randomly generated data assigned to dataset variable: {sorted(names & dataset_names)}")
