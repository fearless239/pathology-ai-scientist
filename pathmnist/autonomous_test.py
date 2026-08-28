from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

from .candidates import CandidateError, FrozenCandidate, OneTimeTestEvaluator, approve_test_evaluation, require_inference_candidate
from .experiment_contract import ExperimentResult, code_sha256
from .dataset_adapter import DatasetSpec, SampleRecord, materialize_split_view
from .scientific_integrity import IntegrityError, record_trusted_evaluation
from .trusted_statistics import comparison_summary
from .execution_control import task_operation


def _candidate(task_root: Path) -> FrozenCandidate:
    raw = json.loads((task_root / "candidate_frozen/candidate.json").read_text(encoding="utf-8"))
    return FrozenCandidate(**raw)


def validate_frozen_inference(project_root: Path, task_root: Path):
    """Exercise every frozen arm on validation only, before consuming test approval."""
    import hashlib
    from gate_a.config import load_config
    from gate_a.runner import DockerRunner
    from .autonomous_preflight import _load_spec
    from .research_contract import load_contract
    bundle_path = task_root / 'candidate_frozen/comparison_bundle.json'
    bundle = json.loads(bundle_path.read_text(encoding='utf-8'))
    contract = load_contract(task_root, require_approved=True)
    expected = {(role, seed) for role in ('baseline', 'proposed_method') for seed in contract['repeat_plan']['seeds']}
    if {(arm['role_id'], arm['seed']) for arm in bundle['experiments']} != expected:
        raise CandidateError('Frozen bundle does not cover the approved repeat plan')
    fingerprint = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    directory = task_root / 'research/inference_preflight'
    receipt = directory / 'completed.json'
    if receipt.is_file() and json.loads(receipt.read_text(encoding='utf-8')).get('bundle_sha256') == fingerprint:
        return
    view = directory / 'validation_view'
    if not (view / 'dataset_profile.json').is_file():
        materialize_split_view(_load_spec(task_root / 'dataset/dataset_profile.json'), view, {'validation'})
    config = load_config(project_root / 'configs/gate_a_llm.yaml')
    runner = DockerRunner(replace(config.runner, image='path-scientist-pathmnist-runner:0.1',
                                   timeout_seconds=1800), gpus='all', shm_size='2g')
    for arm in bundle['experiments']:
        work = directory / f"{arm['role_id']}-{arm['seed']}"
        work.mkdir(parents=True, exist_ok=True)
        frozen = task_root / 'candidate_frozen'
        shutil.copy2(frozen / arm['checkpoint'], work / 'model_checkpoint.pt')
        source = (frozen / arm['code']).read_text(encoding='utf-8')
        result = runner.run_python(source, work, 'run.py', dataset_mount=view)
        if not result.succeeded:
            raise CandidateError(f"Frozen inference preflight failed: {arm['experiment_id']}: {result.stderr[-2000:]}")
        raw = json.loads((work / 'working/experiment_result.json').read_text(encoding='utf-8'))
        record_trusted_evaluation(profile_path=view / 'dataset_profile.json', split='validation',
            sample_ids=raw['sample_ids'], targets=raw['targets'], predictions=raw['predictions'],
            probabilities=raw.get('probabilities'), code_sha256=code_sha256(source), output_dir=work / 'integrity')
        expected_result = json.loads((frozen / arm['validation_result']).read_text(encoding='utf-8'))
        if dict(zip(raw['sample_ids'], raw['predictions'])) != dict(zip(expected_result['sample_ids'], expected_result['predictions'])):
            raise CandidateError('Restored checkpoint does not reproduce frozen validation predictions')
    receipt.write_text(json.dumps({'bundle_sha256': fingerprint, 'passed': True}), encoding='utf-8')


@task_operation
def approve(project_root: Path, state_root: Path, task_id: str) -> dict[str, object]:
    task_root = state_root.resolve() / task_id
    candidate = _candidate(task_root)
    require_inference_candidate(task_root)
    validate_frozen_inference(project_root, task_root)
    path = approve_test_evaluation(task_root / "final_evaluation", candidate)
    task_path = task_root / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["stages"]["test_evaluation_approved"] = "completed"
    task["completed_stage"] = "test_evaluation_approved"
    task["control"] = "paused"
    temporary = task_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(task_path)
    return {"task_id": task_id, "approved": True, "approval": str(path)}


def _make_sealed_view(task_root: Path, destination: Path) -> Path:
    profile = json.loads((task_root / "dataset/dataset_profile.json").read_text(encoding="utf-8"))
    if destination.exists() and any(destination.iterdir()):
        if (destination / "dataset_profile.json").is_file():
            return destination
        raise CandidateError("Existing sealed view is incomplete; refusing to mix or overwrite it")
    samples = [SampleRecord(**item) for item in profile.get("samples", []) if item.get("split") == "test"]
    if not samples:
        raise CandidateError("Dataset profile has no sealed test samples")
    remapped = [replace(sample, split="validation") for sample in samples]
    fields = DatasetSpec.__dataclass_fields__
    values = {key: profile[key] for key in fields if key in profile and key != "samples"}
    values.update(
        samples=remapped,
        split_counts={"validation": len(remapped)},
        class_counts={"validation": profile.get("class_counts", {}).get("test", {})},
    )
    return materialize_split_view(DatasetSpec(**values), destination, {"validation"})


@task_operation
def evaluate(project_root: Path, state_root: Path, task_id: str) -> dict[str, object]:
    from gate_a.config import load_config
    from gate_a.runner import DockerRunner
    from .autonomous import AutonomousTaskWorkspace

    project_root, state_root = project_root.resolve(), state_root.resolve()
    workspace = AutonomousTaskWorkspace.create(state_root, task_id)
    task_root = workspace.root
    _candidate(task_root)
    final_root = task_root / "final_evaluation"
    evaluator = OneTimeTestEvaluator(final_root)
    recovered = evaluator.recover(task_root / 'candidate_frozen')
    if recovered is not None:
        require_inference_candidate(task_root)
        return _commit_test(task_root, task_id, recovered)
    require_inference_candidate(task_root)
    if (final_root / 'attempted.json').exists():
        raise CandidateError('Test was attempted without a complete receipt; manual recovery required')
    validate_frozen_inference(project_root, task_root)
    sealed = _make_sealed_view(task_root, final_root / "sealed_test")
    evaluator = OneTimeTestEvaluator(final_root)
    config = load_config(project_root / "configs/gate_a_llm.yaml")
    runner_config = replace(config.runner, image="path-scientist-pathmnist-runner:0.1", timeout_seconds=1800, cpus=4.0, memory="8g")
    runner = DockerRunner(runner_config, gpus="all", shm_size="2g")
    code_path = task_root / "candidate_frozen/run.py"

    def contract_metric(metrics: dict, name: str, pair: list[int] | None, class_id=None) -> float:
        from .metrics import metric_value
        value = metric_value(metrics, name, pair, class_id)
        if value is None:
            raise CandidateError(f'Trusted test result lacks metric {name}')
        return value

    def execute(snapshot: Path, test_view: Path) -> ExperimentResult:
        bundle_path = task_root / "candidate_frozen/comparison_bundle.json"
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        arm_records = []
        selected_result = None
        selected_integrity = None
        for arm in bundle["experiments"]:
            arm_name = f"{arm['role_id']}_{arm['seed']}"
            run_dir = final_root / "execution/arms" / arm_name
            run_dir.mkdir(parents=True, exist_ok=True)
            arm_code_path = task_root / "candidate_frozen" / arm["code"]
            arm_checkpoint = task_root / "candidate_frozen" / arm["checkpoint"]
            shutil.copy2(arm_checkpoint, run_dir / "model_checkpoint.pt")
            source = arm_code_path.read_text(encoding="utf-8")
            result = runner.run_python(source, run_dir, "run.py", dataset_mount=test_view)
            if not result.succeeded:
                raise CandidateError(f"Frozen comparison arm {arm_name} failed: {result.stderr[-4000:]}")
            raw_path = run_dir / "working/experiment_result.json"
            if not raw_path.is_file():
                raise CandidateError(f"Frozen comparison arm {arm_name} did not write experiment_result.json")
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            metrics = raw.get("metrics")
            predictions, targets, sample_ids = raw.get("predictions"), raw.get("targets"), raw.get("sample_ids")
            probabilities = raw.get("probabilities")
            if not isinstance(metrics, dict) or not metrics or not all(isinstance(value, list) for value in (predictions, targets, sample_ids)):
                raise CandidateError(f"Frozen comparison arm {arm_name} lacks complete per-sample evidence")
            integrity_dir = final_root / "integrity/arms" / arm_name
            try:
                _, trusted_metrics, _ = record_trusted_evaluation(
                    profile_path=sealed / "dataset_profile.json", split="validation", receipt_split="test",
                    sample_ids=[str(value) for value in sample_ids], targets=targets,
                    predictions=predictions, probabilities=probabilities,
                    code_sha256=code_sha256(source), output_dir=integrity_dir, reported_metrics=metrics,
                )
            except IntegrityError as error:
                raise CandidateError(f"Sealed test integrity failed for {arm_name}: {error}") from error
            arm_result = ExperimentResult(
                method_name=str(arm["role_id"]), code_sha256=code_sha256(source), seed=int(arm["seed"]),
                split="test", metrics={f"test_{key}" if key in {"accuracy", "macro_f1", "weighted_f1"} else key: value for key, value in trusted_metrics.items()},
                resource_usage={"elapsed_seconds": result.elapsed_seconds},
                artifacts={"code": arm["code"], "sealed_view": "sealed_test", "integrity": str(integrity_dir.relative_to(task_root))},
                test_data_accessed=True, predictions=predictions, probabilities=probabilities,
                targets=targets, sample_ids=[str(value) for value in sample_ids], class_names=raw.get("class_names"),
            )
            arm_result.write(final_root / "comparison_results" / f"{arm_name}.json", allow_test=True)
            arm_records.append({**arm, "trusted_metrics": trusted_metrics, "elapsed_seconds": result.elapsed_seconds})
            if arm["experiment_id"] == bundle["selected_candidate_experiment_id"]:
                selected_result = arm_result
                selected_integrity = integrity_dir
        if selected_result is None or selected_integrity is None:
            raise CandidateError("Selected candidate is absent from the frozen comparison bundle")
        for name in ("dataset_execution_receipt.json", "trusted_metrics.json", "metric_provenance.json"):
            shutil.copy2(selected_integrity / name, final_root / "integrity" / name)
        primary = str(bundle["primary_metric"])
        pair = bundle.get("locked_confusion_pair")
        grouped = {"baseline": {}, "proposed_method": {}}
        for row in arm_records:
            grouped[row["role_id"]][int(row["seed"])] = contract_metric(row["trusted_metrics"], primary, pair, bundle.get('primary_class_id'))
        paired_seeds = sorted(set(grouped["baseline"]) & set(grouped["proposed_method"]))
        statistics = comparison_summary(
            [grouped["baseline"][seed] for seed in paired_seeds],
            [grouped["proposed_method"][seed] for seed in paired_seeds],
            bootstrap_seed=int(json.loads((task_root / "task.json").read_text(encoding="utf-8")).get("seed", 7)),
        )
        comparison_record = {
            "schema_version": 1, "attempt_count": 1, "split": "test", "primary_metric": primary,
            "locked_confusion_pair": pair, "arms": arm_records, "statistics": statistics,
        }
        (final_root / "comparison_results.json").write_text(json.dumps(comparison_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return selected_result

    try:
        result = evaluator.evaluate(code_path.parent, execute, sealed)
    except CandidateError as error:
        diagnosis_root = task_root / "paper/failure_diagnosis"
        diagnosis_root.mkdir(parents=True, exist_ok=True)
        diagnosis = {
            "schema_version": 1,
            "publication_mode": "failure_diagnosis",
            "error": str(error),
            "sealed_test_retriable": False,
            "attempt": "final_evaluation/attempted.json",
        }
        (diagnosis_root / "diagnosis.json").write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (diagnosis_root / "diagnosis.md").write_text(
            "# Failure Diagnosis\n\nThe one-time sealed-test attempt failed scientific-integrity validation.\n\n"
            f"- {error}\n- The sealed test must not be repeated.\n",
            encoding="utf-8",
        )
        task_path = task_root / "task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["publication_mode"] = "failure_diagnosis"
        task["control"] = "paused"
        task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    return _commit_test(task_root, task_id, result)


def _commit_test(task_root, task_id, result):
    final_root = task_root / 'final_evaluation'
    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    task["stages"]["test_evaluation_approved"] = "completed"
    task["stages"]["test_evaluated"] = "completed"
    task["completed_stage"] = "test_evaluated"
    task["control"] = "paused"
    (task_root / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"task_id": task_id, "split": result.split, "metrics": result.metrics, "result": str(final_root / "experiment_result.json")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("approve", "evaluate"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    value = approve(args.project_root, args.state_root, args.task_id) if args.mode == "approve" else evaluate(args.project_root, args.state_root, args.task_id)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
