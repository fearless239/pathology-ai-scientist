from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from .autonomous import _has_indexed_high_subset
from .autonomous_paid import run_paid
from gate_a.provider import ProviderError


REPAIR_STAGE = "3_creative_research_2_hard_routing_repair"


def prepare_repair(project_root: Path, state_root: Path, task_id: str) -> bool:
    """Append a corrective AI-Scientist stage without discarding prior journals."""
    project_root, state_root = project_root.resolve(), state_root.resolve()
    vendor = project_root / "vendor/AI-Scientist-v2"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from ai_scientist.treesearch.agent_manager import Stage, StageTransition
    from ai_scientist.treesearch.journal import Journal

    checkpoint = state_root / task_id / "experiment_logs/manager.pkl"
    with checkpoint.open("rb") as handle:
        state = pickle.load(handle)
    if REPAIR_STAGE in state["journals"]:
        repair_journal = state["journals"][REPAIR_STAGE]
        if getattr(state.get("current_stage"), "name", None) == REPAIR_STAGE:
            return False
        audited = any(
            node.is_buggy is False
            and node.parent is not None
            and _has_indexed_high_subset(node.code)
            for node in repair_journal.nodes
        )
        if audited:
            return False
        # A previous attempt exhausted the repair sub-stage without producing
        # an audited child and the upstream manager advanced to an empty stage 4.
        # Reopen the repair stage, retaining all nodes and trimming only the
        # in-memory transition tail so the next run can continue debugging it.
        repair_index = next(i for i, stage in enumerate(state["stages"]) if stage.name == REPAIR_STAGE)
        state["stages"] = state["stages"][: repair_index + 1]
        state["stage_history"] = [
            transition
            for transition in state["stage_history"]
            if transition.to_stage == REPAIR_STAGE or transition.from_stage != REPAIR_STAGE
        ]
        repair_stage = state["stages"][repair_index]
        repair_stage.max_iterations = max(repair_stage.max_iterations, len(repair_journal.nodes) + 3)
        state["current_stage"] = repair_stage
        temporary = checkpoint.with_suffix(".repair.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(state, handle)
        temporary.replace(checkpoint)
        return True

    previous = state["stages"][-1]
    repair = Stage(
        name=REPAIR_STAGE,
        description="hard_routing_repair",
        goals=(
            "Repair the autonomously generated dynamic-resolution method using the prior failed evidence. "
            "The previous implementation always evaluated both resolution branches and its router collapsed "
            "to an extreme. Generate and execute a genuinely conditional per-sample hard-routing method: "
            "compute low-resolution inference first, select a non-trivial subset for high-resolution execution, "
            "and scatter only that subset's results back. Demonstrate a validation high-resolution fraction "
            "strictly between 0.05 and 0.95, zero samples executing both final branches, exact executed sample "
            "counts, fixed-low and fixed-high controls, validation accuracy, and measured inference time. "
            "Use the existing generated code as evidence to debug; do not substitute a fixed experiment template."
        ),
        max_iterations=3,
        num_drafts=0,
        stage_number=max(stage.stage_number for stage in state["stages"]) + 1,
    )
    state["stages"].append(repair)
    state["journals"][REPAIR_STAGE] = Journal()
    state["stage_history"].append(
        StageTransition(
            from_stage=previous.name,
            to_stage=repair.name,
            reason="Automated evidence audit rejected soft dual execution and route collapse",
            config_adjustments={"dynamic_audit": "required"},
        )
    )
    state["current_stage"] = repair
    temporary = checkpoint.with_suffix(".repair.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(state, handle)
    temporary.replace(checkpoint)
    return True


def run_repair(project_root: Path, state_root: Path, task_id: str) -> dict[str, object]:
    prepared = prepare_repair(project_root, state_root, task_id)
    result = run_paid(project_root, state_root, task_id, repair_dynamic=True)
    return {"repair_stage_prepared": prepared, **result}


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
        print(json.dumps(run_repair(args.project_root, args.state_root, args.task_id), indent=2))
        return 0
    except ProviderError as error:
        print(json.dumps({"status": "interrupted", "reason": "provider_network", "error": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
