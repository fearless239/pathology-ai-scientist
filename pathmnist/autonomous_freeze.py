from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .autonomous import _has_indexed_high_subset
from .candidates import CandidateError, freeze_candidate
from .experiment_manifest import load_manifest
from .research_contract import load_contract
from .execution_control import task_operation


def _general_metric_names(task_root: Path) -> list[str]:
    contract = load_contract(task_root, require_approved=True)
    name = str(contract["metrics"]["primary"]["name"])
    return [name, f"validation_{name}", f"validation_{name}_score"]


def _paired_experiments(experiments, fulfillment, contract):
    # Fulfillment is the sole authority for paired experiment identity.
    references = fulfillment.get("paired_sources", [])
    selected_ids = {
        row[key] for row in references
        for key in ("baseline_experiment_id", "proposed_experiment_id")
    }
    expected_seeds = set(map(int, contract["repeat_plan"]["seeds"]))
    if {row.get("seed") for row in references} != expected_seeds or len(references) != len(expected_seeds):
        raise CandidateError("Verified paired_sources are missing or incomplete; regenerate fulfillment")
    experiments = [item for item in experiments if item.get("experiment_id") in selected_ids]
    if len(experiments) != 2 * len(expected_seeds) or len(selected_ids) != len(experiments):
        raise CandidateError("Verified paired experiment identities are missing or duplicated")
    by_id = {item['experiment_id']: item for item in experiments}
    for row in references:
        for key, role in (('baseline_experiment_id', 'baseline'), ('proposed_experiment_id', 'proposed_method')):
            item = by_id[row[key]]
            if item.get('seed') != row['seed'] or item.get('contract_role') != role:
                raise CandidateError('Verified paired experiment role or seed does not match fulfillment')
    return experiments


def _trusted_value(metrics: dict, metric_name: str, pair: list[int] | None) -> float | None:
    from .metrics import metric_value
    return metric_value(metrics, metric_name, pair)


@task_operation
def freeze_best(project_root: Path, state_root: Path, task_id: str) -> dict[str, object]:
    task_root = state_root.resolve() / task_id
    results_root = task_root / "experiment_logs/results"
    manifest_path = results_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiments = manifest.get("experiments", [])
    fulfillment = json.loads((task_root / "research/contract_fulfillment.json").read_text(encoding="utf-8"))
    if not fulfillment.get("passed"):
        raise CandidateError("Research contract fulfillment has not passed")
    contract = load_contract(task_root, require_approved=True)
    contract_primary = str(contract["metrics"]["primary"]["name"])
    experiments = _paired_experiments(experiments, fulfillment, contract)
    locked_pair = fulfillment.get("locked_confusion_pair")
    dynamic_audit = any(
        item.get("stage") == "3_creative_research_2_hard_routing_repair"
        for item in experiments
    )
    eligible = []
    seen: set[str] = set()
    for item in experiments:
        if item.get("contract_role") != "proposed_method":
            continue
        if item.get("seed") not in set(map(int, contract["repeat_plan"]["seeds"])):
            continue
        if dynamic_audit and item.get("stage") != "3_creative_research_2_hard_routing_repair":
            continue
        experiment_id = str(item["experiment_id"])
        if experiment_id in seen:
            continue
        seen.add(experiment_id)
        result_dir = results_root / experiment_id
        integrity_paths = {
            name: result_dir / name
            for name in (
                "dataset_execution_receipt.json",
                "trusted_metrics.json",
                "metric_provenance.json",
            )
        }
        if not all(path.is_file() for path in integrity_paths.values()):
            continue
        trusted_record = json.loads(
            integrity_paths["trusted_metrics.json"].read_text(encoding="utf-8")
        )
        trusted_metrics = trusted_record.get("metrics", {})
        if not isinstance(trusted_metrics, dict):
            continue
        from .metrics import metric_value
        derived_value = metric_value(trusted_metrics, contract_primary, locked_pair, contract['metrics']['primary'].get('class_id'))
        selected = (contract_primary, derived_value) if derived_value is not None else None
        if selected is None:
            continue
        primary_metric, value = selected
        code_path = results_root / experiment_id / "run.py"
        result_path = results_root / experiment_id / "experiment_result.json"
        checkpoint_path = result_path.parent / "model_checkpoint.pt"
        if not code_path.is_file() or not result_path.is_file() or not checkpoint_path.is_file():
            continue
        if dynamic_audit and not _has_indexed_high_subset(code_path.read_text(encoding="utf-8")):
            continue
        if not isinstance(value, (int, float)):
            continue
        eligible.append((float(value), experiment_id, code_path, result_path, primary_metric))
    if not eligible:
        qualifier = "audited hard-routing " if dynamic_audit else ""
        raise CandidateError(f"No {qualifier}validation candidate is available")
    score, experiment_id, code_path, result_path, primary_metric = max(
        eligible, key=lambda value: (value[0], value[1])
    )
    selection = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "primary_metric": primary_metric,
        "validation_value": score,
        "selection_split": "validation",
        "eligible_candidates": len(eligible),
        "research_contract_sha256": contract["contract_sha256"],
        "hypothesis_supported": fulfillment.get("hypothesis_supported"),
    }
    (task_root / "candidates").mkdir(parents=True, exist_ok=True)
    (task_root / "candidates/selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source_manifest = result_path.parent / "experiment_manifest.json"
    if not source_manifest.is_file():
        raise CandidateError("Autonomous candidate has no experiment_manifest.json")
    load_manifest(source_manifest)
    integrity_artifacts = [
        result_path.parent / name
        for name in ("dataset_execution_receipt.json", "trusted_metrics.json", "metric_provenance.json")
    ]
    missing_integrity = [path.name for path in integrity_artifacts if not path.is_file()]
    if missing_integrity:
        raise CandidateError(f"Autonomous candidate lacks integrity artifacts: {missing_integrity}")
    final_destination = task_root / "candidate_frozen"
    if final_destination.exists():
        required_existing = ("candidate.json", "run.py", "model_checkpoint.pt", "comparison_bundle.json", "contract_fulfillment.json")
        if not all((final_destination / name).is_file() for name in required_existing):
            raise CandidateError(f"Candidate freeze is incomplete at {final_destination}")
        existing = json.loads((final_destination / "candidate.json").read_text(encoding="utf-8"))
        task_path = task_root / "task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["stages"]["candidate_selected"] = "completed"
        task["stages"]["candidate_frozen"] = "completed"
        task["completed_stage"] = "candidate_frozen"
        task["control"] = "paused"
        task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"task_id": task_id, "candidate": existing, "path": str(final_destination), "recovered": True}
    destination = task_root / "candidate_frozen.staging"
    if destination.exists():
        shutil.rmtree(destination)
    frozen = freeze_candidate(
        experiment_id,
        result_path,
        code_path,
        destination,
        primary_metric=primary_metric,
        validation_value=score,
    )
    (destination / "experiment_manifest.json").write_bytes(source_manifest.read_bytes())
    shutil.copy2(result_path.parent / "model_checkpoint.pt", destination / "model_checkpoint.pt")
    for source in integrity_artifacts:
        name = source.name
        shutil.copy2(source, destination / name)
    shutil.copy2(task_root / "research/contract_fulfillment.json", destination / "contract_fulfillment.json")
    shutil.copy2(task_root / "research/semantic_review.json", destination / "semantic_review.json")
    expected_seeds = set(map(int, contract["repeat_plan"]["seeds"]))
    selected_arms: dict[tuple[str, int], dict] = {}
    for item in experiments:
        role = item.get("contract_role")
        seed = item.get("seed")
        if role in {"baseline", "proposed_method"} and isinstance(seed, int) and seed in expected_seeds:
            if (str(role), seed) in selected_arms:
                raise CandidateError("Duplicate verified comparison arm")
            selected_arms[(str(role), seed)] = item
    comparison_entries = []
    arms_root = destination / "comparison_arms"
    arms_root.mkdir(parents=True, exist_ok=True)
    for (role, seed), item in sorted(selected_arms.items()):
        source_dir = task_root / str(item["result"])
        source_dir = source_dir.parent
        arm_name = f"{role}_{seed}"
        arm_root = arms_root / arm_name
        arm_root.mkdir(parents=True, exist_ok=False)
        required_sources = {
            "run.py": source_dir / "run.py",
            "model_checkpoint.pt": source_dir / "model_checkpoint.pt",
            "validation_result.json": source_dir / "experiment_result.json",
            "contract_execution.json": source_dir / "contract_execution.json",
        }
        missing = [name for name, path in required_sources.items() if not path.is_file()]
        if missing:
            raise CandidateError(f"Comparison arm {arm_name} lacks frozen artifacts: {missing}")
        for name, source in required_sources.items():
            shutil.copy2(source, arm_root / name)
        comparison_entries.append({
            "experiment_id": item["experiment_id"],
            "role_id": role,
            "seed": seed,
            "code": f"comparison_arms/{arm_name}/run.py",
            "checkpoint": f"comparison_arms/{arm_name}/model_checkpoint.pt",
            "validation_result": f"comparison_arms/{arm_name}/validation_result.json",
            "code_sha256": item.get("code_sha256"),
            "checkpoint_sha256": __import__('hashlib').sha256((arm_root / "model_checkpoint.pt").read_bytes()).hexdigest(),
        })
    for role in ("baseline", "proposed_method"):
        present = {item["seed"] for item in comparison_entries if item["role_id"] == role}
        if present != expected_seeds:
            raise CandidateError(f"Frozen comparison bundle has incomplete {role} seeds: {sorted(present)}")
    (destination / "comparison_bundle.json").write_text(
        json.dumps({
            "schema_version": 1,
            "contract_sha256": contract["contract_sha256"],
            "sealed_test_attempts_allowed": 1,
            "selected_candidate_experiment_id": experiment_id,
            "locked_confusion_pair": locked_pair,
            "primary_metric": contract_primary,
            "primary_class_id": contract['metrics']['primary'].get('class_id'),
            "experiments": comparison_entries,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    destination.replace(final_destination)
    task_path = task_root / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["stages"]["candidate_selected"] = "completed"
    task["stages"]["candidate_frozen"] = "completed"
    task["completed_stage"] = "candidate_frozen"
    task["control"] = "paused"
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"task_id": task_id, "candidate": frozen.__dict__, "path": str(final_destination)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_best(args.project_root, args.state_root, args.task_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
