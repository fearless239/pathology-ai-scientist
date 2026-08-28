"""Host-only verification of saved validation evidence; never executes generated code."""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

from .experiment_manifest import load_manifest
from .scientific_integrity import IntegrityError, TrustedMetricEvaluator, validate_sample_evidence


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
    for name in ('tuning_evidence.json', 'tuning_progress.json'):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
            hashes[name] = hashlib.sha256((destination / name).read_bytes()).hexdigest()
    (destination / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    destination.replace(final)
