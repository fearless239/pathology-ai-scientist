from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .research_stages import RESEARCH_STAGES
from .autonomous_acceptance import require_task, validate_task


class OrchestrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class StageHandler:
    name: str
    validate_inputs: Callable[[], None]
    run: Callable[[], dict[str, Any]]
    validate_outputs: Callable[[], None]

    def execute(self) -> dict[str, Any]:
        self.validate_inputs()
        value = self.run()
        self.validate_outputs()
        return value


class ResearchOrchestrator:
    """Single public control plane for the formal research workflow."""

    def __init__(self, project_root: Path, state_root: Path, task_id: str):
        self.project_root = project_root.resolve()
        self.state_root = state_root.resolve()
        self.task_id = task_id
        self.task_root = self.state_root / task_id

    def _task(self) -> dict[str, Any]:
        try:
            value = json.loads((self.task_root / "task.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OrchestrationError(f"Invalid or missing research task: {self.task_id}") from exc
        if value.get("schema_version") != 2:
            raise OrchestrationError("Legacy tasks cannot run through the research orchestrator")
        return value

    def status(self) -> dict[str, Any]:
        task = self._task()
        completed = str(task.get("completed_stage"))
        report = validate_task(self.task_root, completed) if completed in RESEARCH_STAGES else None
        next_stage = None
        if completed in RESEARCH_STAGES and completed != RESEARCH_STAGES[-1]:
            next_stage = RESEARCH_STAGES[RESEARCH_STAGES.index(completed) + 1]
        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "completed_stage": completed,
            "next_stage": next_stage,
            "control": task.get("control"),
            "valid": bool(report and report.passed),
            "errors": list(report.errors) if report else ["invalid completed_stage"],
            "data_integrity": report.data_integrity if report else "not_evaluated",
            "sealed_test_integrity": report.sealed_test_integrity if report else "not_evaluated",
            "metric_integrity": report.metric_integrity if report else "not_evaluated",
            "research_contract_integrity": report.research_contract_integrity if report else "not_evaluated",
            "comparison_integrity": report.comparison_integrity if report else "not_evaluated",
            "repeat_integrity": report.repeat_integrity if report else "not_evaluated",
            "statistical_integrity": report.statistical_integrity if report else "not_evaluated",
            "publication_mode": task.get("publication_mode") or (report.publication_mode if report else "not_applicable"),
        }

    def resume(self) -> dict[str, Any]:
        status = self.status()
        if not status["valid"]:
            raise OrchestrationError("Cannot resume an invalid task: " + "; ".join(status["errors"]))
        task = self._task()
        task["control"] = "paused"
        temporary = (self.task_root / "task.json").with_suffix(".tmp")
        temporary.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.task_root / "task.json")
        return self.status()

    def approve_test(self) -> dict[str, Any]:
        from .autonomous_test import approve

        require_task(self.task_root, "candidate_frozen")
        return approve(self.project_root, self.state_root, self.task_id)

    def approve_research_contract(self) -> dict[str, Any]:
        from .research_contract import approve_contract

        require_task(self.task_root, "research_contract_generated")
        return approve_contract(self.task_root)

    def revise_research_contract(self, feedback: str) -> dict[str, Any]:
        from .autonomous_research import regenerate_contract

        return regenerate_contract(self.state_root, self.task_id, feedback, project_root=self.project_root)

    def _handler(self, *, allow_paid: bool, allow_test: bool, allow_pdf: bool) -> StageHandler:
        stage = self._task().get("completed_stage")

        def accepted(name: str) -> Callable[[], None]:
            return lambda: require_task(self.task_root, name)

        if stage == "dataset_validated":
            from .autonomous_research import prepare_research
            return StageHandler("research", accepted("dataset_validated"), lambda: prepare_research(self.state_root, self.task_id, project_root=self.project_root), accepted("research_contract_generated"))
        if stage == "research_contract_generated":
            raise OrchestrationError("The generated research contract requires explicit user approval")
        if stage == "experiment_spec_validated":
            from .autonomous_preflight import run_preflight
            return StageHandler("preflight", accepted(stage), lambda: run_preflight(self.project_root, self.state_root, self.task_id), accepted("sandbox_prechecked"))
        if stage in {"sandbox_prechecked", "initial_implementation_completed", "baseline_tuning_completed", "creative_research_completed"}:
            if not allow_paid:
                raise OrchestrationError("The next stage uses a paid LLM; rerun with --allow-paid")
            from .autonomous_paid import run_paid
            return StageHandler("experiments", accepted(stage), lambda: run_paid(self.project_root, self.state_root, self.task_id), accepted("ablations_completed"))
        if stage == "ablations_completed":
            from .autonomous_export import export_journals
            from .autonomous_freeze import freeze_best
            from .research_contract import evaluate_fulfillment, review_implementation_semantics
            def export_and_freeze() -> dict[str, Any]:
                exported = export_journals(self.project_root, self.state_root, self.task_id)
                semantic_review = review_implementation_semantics(self.project_root, self.task_root)
                fulfillment = evaluate_fulfillment(self.task_root, require_semantic_review=True)
                if not fulfillment["passed"]:
                    diagnosis_root = self.task_root / "paper/failure_diagnosis"
                    diagnosis_root.mkdir(parents=True, exist_ok=True)
                    diagnosis = {
                        "schema_version": 1,
                        "publication_mode": "failure_diagnosis",
                        "reason": "research_contract_incomplete",
                        "errors": fulfillment["errors"],
                        "evidence": "research/contract_fulfillment.json",
                    }
                    (diagnosis_root / "diagnosis.json").write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    raise OrchestrationError("Research contract is incomplete: " + "; ".join(fulfillment["errors"]))
                return {"export": exported, "semantic_review": semantic_review, "fulfillment": fulfillment, "freeze": freeze_best(self.project_root, self.state_root, self.task_id)}
            return StageHandler("freeze", accepted(stage), export_and_freeze, accepted("candidate_frozen"))
        if stage == "candidate_frozen":
            raise OrchestrationError("The frozen candidate requires the separate approve-test action")
        if stage == "test_evaluation_approved":
            if not allow_test:
                raise OrchestrationError("The sealed test requires explicit approval and --allow-test")
            from .autonomous_test import evaluate
            return StageHandler("test", accepted("test_evaluation_approved"), lambda: evaluate(self.project_root, self.state_root, self.task_id), accepted("test_evaluated"))
        if stage in {"test_evaluated", "analysis_completed", "figures_generated", "paper_written", "review_completed", "revision_completed"}:
            if not allow_paid:
                raise OrchestrationError("Paper writing/review uses a paid LLM; rerun with --allow-paid")
            from .autonomous_postprocess import run_postprocess
            return StageHandler("paper", accepted("test_evaluated"), lambda: run_postprocess(self.project_root, self.state_root, self.task_id), accepted("translation_completed"))
        if stage == "translation_completed":
            if not allow_pdf:
                raise OrchestrationError("PDF compilation requires Docker; rerun with --allow-pdf")
            from .autonomous_pdf import build_pdfs
            return StageHandler("pdf", accepted(stage), lambda: build_pdfs(self.project_root, self.state_root, self.task_id, allow_paid=allow_paid), lambda: require_task(self.task_root, "archived", require_pdf=True))
        if stage == "archived":
            raise OrchestrationError("Task is already archived")
        raise OrchestrationError(f"No formal transition is available from {stage!r}")

    def run(self, *, allow_paid: bool = False, allow_test: bool = False, allow_pdf: bool = False, until_blocked: bool = True) -> dict[str, Any]:
        from .execution_control import task_lock
        with task_lock(self.task_root):
            return self._run_locked(allow_paid=allow_paid, allow_test=allow_test,
                                    allow_pdf=allow_pdf, until_blocked=until_blocked)

    def _run_locked(self, *, allow_paid: bool, allow_test: bool, allow_pdf: bool, until_blocked: bool) -> dict[str, Any]:
        transitions: list[dict[str, Any]] = []
        while True:
            try:
                handler = self._handler(allow_paid=allow_paid, allow_test=allow_test, allow_pdf=allow_pdf)
            except OrchestrationError as exc:
                if transitions and until_blocked:
                    return {"task_id": self.task_id, "transitions": transitions, "status": self.status(), "blocked_reason": str(exc)}
                raise
            task_path = self.task_root / "task.json"
            before = task_path.read_bytes()
            try:
                result = handler.execute()
            except Exception:
                # Roll back the uncommitted stage, not durable failure diagnostics.
                try:
                    failed = self._task()
                except (OSError, ValueError):
                    failed = {}
                restored = json.loads(before)
                try:
                    if validate_task(self.task_root, failed['completed_stage']).passed:
                        restored = failed.copy()
                except (ValueError, KeyError, RuntimeError):
                    pass
                for key in ("publication_mode", "failure", "interruption"):
                    if key in failed:
                        restored[key] = failed[key]
                restored["control"] = "interrupted"
                temporary = task_path.with_suffix(".rollback.tmp")
                temporary.write_text(json.dumps(restored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                temporary.replace(task_path)
                raise
            transitions.append({"handler": handler.name, "result": result})
            if not until_blocked or self.status()["completed_stage"] == "archived":
                return {"task_id": self.task_id, "transitions": transitions, "status": self.status()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Path-AI Scientist research workflow")
    parser.add_argument("action", choices=("init", "run", "resume", "status", "approve-contract", "approve-test"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--state-root", type=Path, default=Path("state/workflow"))
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--direction")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--allow-pdf", action="store_true")
    parser.add_argument("--one-stage", action="store_true")
    args = parser.parse_args()
    if args.action == "init":
        if args.dataset_path is None or not (args.direction or "").strip():
            parser.error("init requires --dataset-path and --direction")
        from .cli import _autonomous_init
        result = _autonomous_init(argparse.Namespace(
            state_root=args.state_root,
            task_id=args.task_id,
            dataset_path=args.dataset_path,
            direction=args.direction,
            seed=args.seed,
            resume=False,
        ))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    orchestrator = ResearchOrchestrator(args.project_root, args.state_root, args.task_id)
    if args.action == "status":
        result = orchestrator.status()
    elif args.action == "resume":
        result = orchestrator.resume()
    elif args.action == "approve-contract":
        result = orchestrator.approve_research_contract()
    elif args.action == "approve-test":
        result = orchestrator.approve_test()
    else:
        result = orchestrator.run(allow_paid=args.allow_paid, allow_test=args.allow_test, allow_pdf=args.allow_pdf, until_blocked=not args.one_stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Pre-release compatibility for local tasks and downstream imports.
AutonomousOrchestrator = ResearchOrchestrator
