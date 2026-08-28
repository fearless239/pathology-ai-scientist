from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .experiment_contract import ExperimentResult, code_sha256


class CandidateError(RuntimeError):
    pass


def require_inference_candidate(task_root: Path) -> None:
    code = (task_root / "candidate_frozen/run.py").read_text(encoding="utf-8")
    checkpoint = task_root / "candidate_frozen/model_checkpoint.pt"
    required = ("HAS_TRAIN_SPLIT", "model_checkpoint.pt", "torch.load")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or not all(token in code for token in required):
        raise CandidateError("Candidate lacks a checkpoint-backed inference-only branch; test approval is blocked")
    bundle_path = task_root / "candidate_frozen/comparison_bundle.json"
    if bundle_path.is_file():
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        arms = bundle.get("experiments")
        if not isinstance(arms, list) or not arms:
            raise CandidateError("Frozen comparison bundle is empty")
        if bundle.get("selected_candidate_experiment_id") not in {arm.get("experiment_id") for arm in arms}:
            raise CandidateError("Selected candidate is absent from the frozen comparison bundle")
        identities = [(arm.get("role_id"), arm.get("seed")) for arm in arms]
        if len(set(identities)) != len(identities):
            raise CandidateError("Duplicate frozen comparison arm")
        for arm in arms:
            arm_code = task_root / "candidate_frozen" / str(arm.get("code", ""))
            arm_checkpoint = task_root / "candidate_frozen" / str(arm.get("checkpoint", ""))
            if not arm_code.is_file() or not arm_checkpoint.is_file() or arm_checkpoint.stat().st_size == 0:
                raise CandidateError(f"Frozen comparison arm is incomplete: {arm.get('experiment_id')}")
            source = arm_code.read_text(encoding="utf-8")
            if code_sha256(source) != arm.get("code_sha256") or not all(token in source for token in required):
                raise CandidateError(f"Frozen comparison arm is not immutable inference code: {arm.get('experiment_id')}")
            if _sha256(arm_checkpoint) != arm.get("checkpoint_sha256"):
                raise CandidateError("Frozen checkpoint hash missing or changed; explicit migration required")


@dataclass(frozen=True)
class FrozenCandidate:
    schema_version: int
    experiment_id: str
    method_name: str
    primary_metric: str
    validation_value: float
    maximize: bool
    source_result: str
    code_sha256: str
    snapshot_sha256: str
    frozen_at: str


def select_validation_candidate(
    results: Iterable[tuple[str, Path]], primary_metric: str, maximize: bool = True
) -> tuple[str, Path, ExperimentResult]:
    eligible = []
    for experiment_id, path in results:
        result = ExperimentResult.read(path)
        if result.split != "validation" or result.test_data_accessed:
            raise CandidateError(f"Candidate {experiment_id} is not validation-only")
        value = result.metrics.get(primary_metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CandidateError(f"Candidate {experiment_id} lacks numeric metric {primary_metric!r}")
        eligible.append((float(value), experiment_id, path, result))
    if not eligible:
        raise CandidateError("No completed validation candidates")
    value, experiment_id, path, result = sorted(
        eligible, key=lambda item: ((-item[0] if maximize else item[0]), item[1])
    )[0]
    return experiment_id, path, result


def freeze_candidate(
    experiment_id: str,
    result_path: Path,
    code_path: Path,
    destination: Path,
    primary_metric: str,
    maximize: bool = True,
    validation_value: float | None = None,
) -> FrozenCandidate:
    result = ExperimentResult.read(result_path)
    # The experiment contract hashes canonical source text so CRLF/LF checkout
    # differences do not invalidate an otherwise identical generated program.
    actual_code_hash = code_sha256(code_path.read_text(encoding="utf-8"))
    if actual_code_hash != result.code_sha256:
        raise CandidateError("Generated code differs from the experiment result hash")
    value = validation_value if validation_value is not None else result.metrics.get(primary_metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateError(f"Frozen result lacks numeric metric {primary_metric!r}")
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = destination / "run.py"
    shutil.copy2(code_path, snapshot)
    shutil.copy2(result_path, destination / "validation_result.json")
    snapshot.chmod(0o444)
    record = FrozenCandidate(
        2,
        experiment_id,
        result.method_name,
        primary_metric,
        float(value),
        maximize,
        "validation_result.json",
        result.code_sha256,
        code_sha256(snapshot.read_text(encoding="utf-8")),
        datetime.now(timezone.utc).isoformat(),
    )
    (destination / "candidate.json").write_text(
        json.dumps(asdict(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


class OneTimeTestEvaluator:
    """Execute a frozen snapshot once; never returns a retry/tuning signal."""

    def __init__(self, final_evaluation_dir: Path):
        self.root = final_evaluation_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        candidate_dir: Path,
        execute: Callable[[Path, Path], ExperimentResult],
        sealed_test_view: Path,
    ) -> ExperimentResult:
        approval = self.root / "approval.json"
        attempted = self.root / "attempted.json"
        output = self.root / "experiment_result.json"
        if not approval.is_file():
            raise CandidateError("Test evaluation requires explicit durable approval")
        if attempted.exists() or output.exists():
            raise CandidateError("The frozen candidate test evaluation was already attempted")
        candidate = json.loads((candidate_dir / "candidate.json").read_text(encoding="utf-8"))
        code = candidate_dir / "run.py"
        if code_sha256(code.read_text(encoding="utf-8")) != candidate["snapshot_sha256"]:
            raise CandidateError("Frozen candidate snapshot was modified")
        approval_record = json.loads(approval.read_text(encoding="utf-8"))
        bundle = candidate_dir / "comparison_bundle.json"
        bundle_hash = _sha256(bundle) if bundle.is_file() else None
        if approval_record.get("comparison_bundle_sha256") != bundle_hash:
            raise CandidateError("Frozen comparison bundle differs from the approved snapshot")
        if approval_record.get('candidate_snapshot_sha256') != candidate['snapshot_sha256']:
            raise CandidateError('Approval belongs to another candidate')
        try:
            handle = attempted.open("x", encoding="utf-8")
        except FileExistsError as error:
            raise CandidateError("The frozen candidate test evaluation was already attempted") from error
        with handle:
            handle.write(json.dumps(
                {
                    "schema_version": 2,
                    "attempted_at": datetime.now(timezone.utc).isoformat(),
                    "candidate_snapshot_sha256": candidate["snapshot_sha256"],
                    "comparison_bundle_sha256": bundle_hash,
                },
                indent=2,
            )
            + "\n")
        result = execute(code, sealed_test_view)
        if result.split != "test" or not result.test_data_accessed:
            raise CandidateError("Independent evaluator must explicitly report test access")
        result.write(output, allow_test=True)
        files = [output, *self.root.glob('comparison_results*.json'),
                 *self.root.glob('comparison_results/*.json'), *self.root.glob('integrity/**/*.json')]
        record = {'candidate_snapshot_sha256': candidate['snapshot_sha256'],
                  'comparison_bundle_sha256': bundle_hash,
                  'artifacts': {str(p.relative_to(self.root)): _sha256(p) for p in files}}
        temporary = self.root / 'completed.tmp'
        temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
        temporary.replace(self.root / 'completed.json')
        return result

    def recover(self, candidate_dir: Path) -> ExperimentResult | None:
        receipt = self.root / 'completed.json'
        if not receipt.is_file():
            return None
        record = json.loads(receipt.read_text(encoding='utf-8'))
        approval = json.loads((self.root / 'approval.json').read_text(encoding='utf-8'))
        bundle = candidate_dir / 'comparison_bundle.json'
        if (record.get('candidate_snapshot_sha256') != code_sha256((candidate_dir / 'run.py').read_text(encoding='utf-8'))
                or record.get('comparison_bundle_sha256') != (_sha256(bundle) if bundle.exists() else None)
                or any(record.get(k) != approval.get(k) for k in ('candidate_snapshot_sha256', 'comparison_bundle_sha256'))):
            raise CandidateError('Completed test belongs to another approved snapshot')
        if 'experiment_result.json' not in record.get('artifacts', {}):
            raise CandidateError('Completed test receipt is incomplete')
        for name, digest in record['artifacts'].items():
            artifact = (self.root / name).resolve()
            if not artifact.is_relative_to(self.root.resolve()) or _sha256(artifact) != digest:
                raise CandidateError('Completed test artifact hash mismatch')
        return ExperimentResult.read(self.root / 'experiment_result.json', allow_test=True)


def approve_test_evaluation(final_evaluation_dir: Path, candidate: FrozenCandidate) -> Path:
    final_evaluation_dir.mkdir(parents=True, exist_ok=True)
    path = final_evaluation_dir / "approval.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        bundle = final_evaluation_dir.parent / 'candidate_frozen/comparison_bundle.json'
        if existing.get('candidate_snapshot_sha256') != candidate.snapshot_sha256 or existing.get('comparison_bundle_sha256') != (_sha256(bundle) if bundle.is_file() else None):
            raise CandidateError('Test evaluation is already approved for another snapshot')
        return path
    bundle = final_evaluation_dir.parent / "candidate_frozen/comparison_bundle.json"
    bundle_hash = _sha256(bundle) if bundle.is_file() else None
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "approved": True,
                "candidate_snapshot_sha256": candidate.snapshot_sha256,
                "comparison_bundle_sha256": bundle_hash,
                "approved_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def evidence_value(result_path: Path, metric: str, allow_test: bool = False) -> dict[str, Any]:
    result = ExperimentResult.read(result_path, allow_test=allow_test)
    if metric not in result.metrics:
        raise CandidateError(f"Metric {metric!r} is absent from {result_path}")
    return {
        "value": result.metrics[metric],
        "artifact": str(result_path.resolve()),
        "artifact_sha256": _sha256(result_path),
        "json_pointer": f"/metrics/{metric.replace('~', '~0').replace('/', '~1')}",
        "code_sha256": result.code_sha256,
        "split": result.split,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
