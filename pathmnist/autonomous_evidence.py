"""Host-only verification of saved validation evidence; never executes generated code."""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .experiment_manifest import load_manifest
from .scientific_integrity import IntegrityError, TrustedMetricEvaluator, validate_sample_evidence


def repair_missing_primary_metric(
    raw_directory: Path, task_root: Path, profile_path: Path
) -> Path:
    """Create verified evidence from an immutable raw run with one missing metric label."""
    receipt_path = raw_directory / "raw_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != 1 or receipt.get("status") != "unvalidated":
        raise IntegrityError("Raw execution receipt is not an unvalidated schema-v1 record")
    hashes = receipt.get("artifacts")
    if not isinstance(hashes, dict) or not hashes:
        raise IntegrityError("Raw execution receipt has no artifact hashes")
    for name, expected in hashes.items():
        if Path(name).name != name or not (raw_directory / name).is_file():
            raise IntegrityError("Raw execution receipt contains an unsafe or missing artifact")
        actual = hashlib.sha256((raw_directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise IntegrityError(f"Raw execution artifact hash mismatch: {name}")

    from .experiment_manifest import metric_policy
    from .research_contract import load_contract
    from .scientific_integrity import record_trusted_evaluation

    contract = load_contract(task_root, require_approved=True)
    primary = str(contract["metrics"]["primary"]["name"])
    original_manifest = json.loads(
        (raw_directory / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    if "primary_metric" in original_manifest:
        raise IntegrityError("Repair applies only when primary_metric is absent")
    if original_manifest.get("selection_metric") != primary:
        raise IntegrityError("Cannot infer primary_metric from a different selection_metric")
    checkpoint = original_manifest.get("checkpoint_selection")
    if checkpoint not in (
        {"metric": primary, "mode": "max"},
        {"metric": "validation_loss", "mode": "min"},
    ):
        raise IntegrityError("Cannot repair an ambiguous checkpoint selection policy")

    code_hash = hashlib.sha256((raw_directory / "run.py").read_bytes()).hexdigest()
    if hashes.get("run.py") != code_hash:
        raise IntegrityError("Raw code hash and receipt disagree")
    pending = task_root / "experiment_logs" / (
        f"metadata-repair-{code_hash}.pending-{uuid.uuid4().hex}"
    )
    pending.mkdir(parents=True)
    try:
        for name in hashes:
            shutil.copy2(raw_directory / name, pending / name)
        manifest = {**original_manifest, "primary_metric": primary}
        (pending / "experiment_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        manifest = load_manifest(pending / "experiment_manifest.json", require_training_policy=True)
        metric_policy(manifest, primary)

        result_path = pending / "experiment_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        execution = json.loads((pending / "contract_execution.json").read_text(encoding="utf-8"))
        if execution.get("contract_role") != "baseline":
            raise IntegrityError("Only a baseline raw run can be repaired here")
        seed = int(manifest["seed"])
        if int(result.get("seed", -1)) != seed or int(execution.get("training_seed", -1)) != seed:
            raise IntegrityError("Raw result, manifest and execution seed disagree")
        predictions, targets, sample_ids = (
            result.get("predictions"), result.get("targets"), result.get("sample_ids")
        )
        if not all(isinstance(value, list) for value in (predictions, targets, sample_ids)):
            raise IntegrityError("Raw predictions, targets and sample IDs must be arrays")
        _, trusted, _ = record_trusted_evaluation(
            profile_path=profile_path,
            split="validation",
            sample_ids=[str(value) for value in sample_ids],
            targets=targets,
            predictions=predictions,
            probabilities=result.get("probabilities"),
            code_sha256=code_hash,
            output_dir=pending,
            reported_metrics=result.get("metrics"),
        )
        from .metrics import add_contract_metric
        trusted = add_contract_metric(task_root, trusted, baseline_hash=code_hash)
        canonical = {
            "schema_version": 1,
            "status": "completed",
            "method_name": str(result.get("method_name") or "agent_generated_method"),
            "parent_experiment_id": result.get("parent_experiment_id"),
            "code_sha256": code_hash,
            "seed": seed,
            "split": "validation",
            "metrics": {**result.get("metrics", {}), **trusted},
            "resource_usage": result.get("resource_usage", {}),
            "artifacts": result.get("artifacts", {}),
            "test_data_accessed": False,
            "predictions": predictions,
            "targets": targets,
            "sample_ids": [str(value) for value in sample_ids],
            "probabilities": result.get("probabilities"),
            "class_names": result.get("class_names"),
        }
        result_path.write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
        repair = {
            "schema_version": 1,
            "source_raw_execution": raw_directory.name,
            "source_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "contract_sha256": contract["contract_sha256"],
            "change": {"field": "primary_metric", "before": None, "after": primary},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (pending / "metadata_repair.json").write_text(
            json.dumps(repair, indent=2) + "\n", encoding="utf-8"
        )
        verified_metrics(pending, profile_path, code_hash)
        final = task_root / "experiment_logs" / "evidence" / code_hash
        snapshot_evidence(pending, final)
        verified_metrics(final, profile_path, code_hash)
        return final
    finally:
        shutil.rmtree(pending, ignore_errors=True)


def verified_metrics(directory: Path, profile_path: Path, code_hash: str) -> dict:
    def read(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))

    result = read("experiment_result.json")
    provenance = read("metric_provenance.json")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if result.get("code_sha256") != code_hash or provenance.get("code_sha256") != code_hash:
        raise IntegrityError("Saved evidence belongs to another generated program")
    if provenance.get("dataset_profile_sha256") != hashlib.sha256(profile_path.read_bytes()).hexdigest():
        raise IntegrityError("Saved evidence belongs to another dataset profile")
    if result.get("split") != "validation" or result.get("test_data_accessed") is not False:
        raise IntegrityError("Only isolated validation evidence may be reused")
    manifest = load_manifest(directory / "experiment_manifest.json")
    if manifest["seed"] != result.get("seed"):
        raise IntegrityError("Saved manifest and result seeds disagree")
    validate_sample_evidence(
        profile, "validation", result["sample_ids"], result["targets"],
        result["predictions"], result.get("probabilities"),
    )
    metrics = TrustedMetricEvaluator().evaluate(
        result["predictions"], result["targets"], len(profile["classes"])
    )
    if metrics != read("trusted_metrics.json").get("metrics"):
        raise IntegrityError("Saved trusted metrics disagree with recomputed predictions")
    if not (directory / "model_checkpoint.pt").is_file():
        raise IntegrityError("Saved model checkpoint is missing")
    receipt = directory / "artifact_hashes.json"
    if receipt.is_file():
        for name, expected in read("artifact_hashes.json").items():
            if Path(name).name != name or hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise IntegrityError("Saved artifact hash mismatch")
    from .metrics import add_contract_metric
    return add_contract_metric(profile_path.parent.parent.parent, metrics)


def metric_rows(metrics: dict) -> list[dict]:
    return [
        {"metric_name": name, "lower_is_better": False,
         "description": "Host-recomputed validation metric",
         "data": [{"dataset_name": "validation", "final_value": metrics[name], "best_value": metrics[name]}]}
        for name, value in metrics.items() if type(value) in (int, float)
    ]


def preserve_unvalidated_execution(source: Path, root: Path, code: str) -> Path:
    """Retain raw training outputs before validation; never treat this as accepted evidence."""
    digest = hashlib.sha256(code.encode('utf-8')).hexdigest()
    final = root / (digest + '-' + uuid.uuid4().hex)
    pending = final.with_name(final.name + '.pending')
    pending.mkdir(parents=True)
    (pending / 'run.py').write_text(code, encoding='utf-8')
    hashes = {'run.py': digest}
    for name in ('experiment_result.json', 'experiment_manifest.json', 'model_checkpoint.pt',
                 'tuning_evidence.json', 'tuning_progress.json', 'contract_execution.json'):
        if (source / name).is_file():
            shutil.copy2(source / name, pending / name)
            hashes[name] = hashlib.sha256((pending / name).read_bytes()).hexdigest()
    (pending / 'raw_receipt.json').write_text(json.dumps({
        'schema_version': 1, 'status': 'unvalidated', 'artifacts': hashes,
        'warning': 'Not accepted evidence. Files require consistency and provenance checks before reuse.',
    }, indent=2), encoding='utf-8')
    pending.replace(final)
    return final


def snapshot_evidence(source: Path, destination: Path) -> None:
    """Preserve completed artifacts before another worker overwrites its workspace."""
    if destination.exists():
        receipt = destination / 'artifact_hashes.json'
        if not receipt.is_file():
            raise IntegrityError('Incomplete historical evidence requires explicit recovery')
        for name, digest in json.loads(receipt.read_text(encoding='utf-8')).items():
            if Path(name).name != name or hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest:
                raise IntegrityError('Existing evidence hash mismatch')
        return
    final = destination
    destination = destination.with_name(destination.name + '.pending-' + uuid.uuid4().hex)
    destination.mkdir(parents=True)
    hashes = {}
    for name in (
        "experiment_result.json", "experiment_manifest.json", "trusted_metrics.json",
        "metric_provenance.json", "dataset_execution_receipt.json", "contract_execution.json",
        "model_checkpoint.pt",
    ):
        shutil.copy2(source / name, destination / name)
        hashes[name] = hashlib.sha256((destination / name).read_bytes()).hexdigest()
    for name in ('tuning_evidence.json', 'tuning_progress.json', 'metadata_repair.json'):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
            hashes[name] = hashlib.sha256((destination / name).read_bytes()).hexdigest()
    (destination / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    destination.replace(final)
