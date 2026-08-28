from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import replace
from pathlib import Path

from omegaconf import OmegaConf

from gate_a.budget import BudgetLedger
from gate_a.config import load_config
from gate_a.pipeline import select_live_models
from gate_a.provider import ProviderError, ZhipuProvider
from gate_a.runner import DockerRunner

from .autonomous import (
    AIScientistExperimentRunner,
    AutonomousTaskWorkspace,
    has_valid_generated_node,
)
from .autonomous_preflight import _load_spec
from .autonomous_acceptance import require_task
from .research_contract import load_contract
from .execution_control import task_lock
from .tuning_evidence import select_verified_tuning


def run_paid(project_root: Path, state_root: Path, task_id: str, *, repair_dynamic: bool = False) -> dict[str, object]:
    workspace = AutonomousTaskWorkspace.create(state_root.resolve(), task_id)
    with task_lock(workspace.root, reentrant=True):
        return _run_paid_locked(project_root, state_root, task_id, repair_dynamic=repair_dynamic)


def _run_paid_locked(project_root: Path, state_root: Path, task_id: str, *, repair_dynamic: bool = False) -> dict[str, object]:
    project_root, state_root = project_root.resolve(), state_root.resolve()
    workspace = AutonomousTaskWorkspace.create(state_root, task_id)
    task_path = workspace.root / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    resumable_stages = {
        "sandbox_prechecked",
        "initial_implementation_completed",
        "baseline_tuning_completed",
        "creative_research_completed",
        "ablations_completed",
    }
    if task.get("schema_version") != 2 or task.get("completed_stage") not in resumable_stages:
        raise RuntimeError("Paid AgentManager run requires a prepared, preflighted research task")
    require_task(workspace.root, "sandbox_prechecked")
    contract = load_contract(workspace.root, require_approved=True)
    from .stage_policy import check_upstream_compatibility

    check_upstream_compatibility(contract)
    spec = _load_spec(workspace.dataset / "dataset_profile.json")

    provider_config = load_config(project_root / "configs/gate_a_llm.yaml")
    budget_limit = float(task.get("budget_limit_usd", 8.0))
    if budget_limit != 8.0:
        budget_limit = 8.0
        task["budget_limit_usd"] = budget_limit
    ledger = BudgetLedger.open_or_upgrade(workspace.root / "budget.json", budget_limit)
    selected = select_live_models(provider_config)
    provider = ZhipuProvider(provider_config, selected, ledger, workspace.research / "responses")
    runner_config = replace(
        provider_config.runner,
        image="path-scientist-pathmnist-runner:0.1",
        timeout_seconds=3600,
        cpus=4.0,
        memory="8g",
        pids_limit=256,
    )
    docker_runner = DockerRunner(runner_config, gpus="all", shm_size="2g", stream_output=True)
    docker_runner.cancel_active(workspace.experiment_workspace)
    scientist = AIScientistExperimentRunner(
        project_root, provider, docker_runner, require_dynamic_audit=repair_dynamic
    )

    cfg = OmegaConf.load(project_root / "vendor/AI-Scientist-v2/bfts_config.yaml")
    cfg.generate_report = False
    # Worker includes code generation, a 3600s sandbox, and evidence validation.
    cfg.exec.timeout = 5400
    cfg.agent.contract_metric = contract["metrics"]["primary"]["name"]
    try:
        import torch
        gpu_count = int(torch.cuda.device_count())
    except (ImportError, RuntimeError):
        gpu_count = 1
    cfg.agent.num_workers = max(1, min(4, gpu_count))
    cfg.agent.steps = 20
    cfg.agent.search.num_drafts = 3
    cfg.agent.search.max_debug_depth = 3
    cfg.agent.search.debug_prob = 1.0
    cfg.agent.multi_seed_eval.num_seeds = int(contract["repeat_plan"]["count"])
    cfg.agent.stages.stage1_max_iters = 20
    cfg.agent.stages.stage2_max_iters = 12
    cfg.agent.stages.stage3_max_iters = 12
    cfg.agent.stages.stage4_max_iters = 18
    cfg.agent.code.model = selected["experiment_code"].model_id
    cfg.agent.feedback.model = selected["experiment_code"].model_id
    cfg.agent.vlm_feedback.model = selected["experiment_code"].model_id
    cfg.report.model = selected["paper_writer"].model_id

    task["control"] = "running"
    task["stages"]["sandbox_prechecked"] = "completed"
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        manager = scientist.run(task_id, task["research_direction"], spec, workspace, cfg)
    except BaseException as error:
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["control"] = "interrupted"
        task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        failures = workspace.experiment_logs / "interruptions"
        failures.mkdir(exist_ok=True)
        (failures / f"{uuid.uuid4().hex}.json").write_text(json.dumps({
            "status": "interrupted", "error_type": type(error).__name__,
            "reason": str(error), "completed_stage": task.get("completed_stage"),
        }, indent=2), encoding="utf-8")
        raise

    stage_map = {
        1: "initial_implementation_completed",
        2: "baseline_tuning_completed",
        3: "creative_research_completed",
        4: "ablations_completed",
    }
    completed_numbers = set()
    for name, journal in manager.journals.items():
        try:
            number = int(name.split("_", 1)[0])
        except ValueError:
            continue
        if number == 1 and journal.good_nodes:
            completed_numbers.add(number)
        elif number == 2 and select_verified_tuning(
            journal, workspace.root, cfg.agent.contract_metric
        )[0] is not None:
            completed_numbers.add(number)
        elif number in {3, 4} and has_valid_generated_node(journal):
            completed_numbers.add(number)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    from .stage_policy import enabled_stages
    enabled = set(enabled_stages(contract))
    for number, stage in stage_map.items():
        task["stages"][stage] = (
            "not_applicable" if number not in enabled else "completed" if number in completed_numbers else "waiting"
        )
    task["completed_stage"] = (
        "ablations_completed" if enabled <= completed_numbers else stage_map[max(completed_numbers)]
        if completed_numbers
        else "sandbox_prechecked"
    )
    task["control"] = "paused" if enabled <= completed_numbers else "interrupted"
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    budget = ledger.snapshot()
    return {
        "task_id": task_id,
        "completed_agent_stages": sorted(completed_numbers),
        "journal_count": len(manager.journals),
        "good_nodes": sum(len(journal.good_nodes) for journal in manager.journals.values()),
        "buggy_nodes": sum(len(journal.buggy_nodes) for journal in manager.journals.values()),
        "spent_usd": budget.spent_usd,
        "available_usd": budget.available_usd,
        "checkpoint": str(workspace.experiment_logs / "manager.pkl"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--confirm-paid", action="store_true")
    args = parser.parse_args()
    if not args.confirm_paid:
        parser.error("--confirm-paid is required")
    try:
        print(json.dumps(run_paid(args.project_root, args.state_root, args.task_id), indent=2))
        return 0
    except ProviderError as error:
        # Network/provider outages are resumable workflow interruptions. Keep
        # the checkpoint and emit machine-readable status instead of a long
        # traceback that suggests the experiment itself failed.
        print(json.dumps({"status": "interrupted", "reason": "provider_network", "error": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
