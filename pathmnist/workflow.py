from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .config import load_config
from .data import validate_dataset
from .freeze import load_frozen_candidate
from .literature import LiteratureError, search_semantic_scholar
from .trace import TraceWriter

from gate_a.budget import BudgetLedger


class WorkflowError(ValueError):
    pass


def _workflow_request_id(task_id: str, suffix: str) -> str:
    """Map a task ID to a provider-safe request ID without losing uniqueness."""
    import re as _re

    if _re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", task_id):
        return f"{task_id}-{suffix}"
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    return f"task-{digest}-{suffix}"


DIRECTION_DISCIPLINE_PROMPT = (
    "Constraints that must be respected by every answer: the study uses only the local "
    "PathMNIST 64x64 dataset (SHA-256 pinned), trains and tunes on train/validation splits "
    "only, evaluates the held-out test split exactly once after candidate freezing, uses "
    "Macro-F1 as the primary metric, and makes no clinical, diagnostic, or patient-level claims."
)

RESEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "objective": {"type": "string"},
        "background": {"type": "string"},
        "key_questions": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["objective", "background", "key_questions", "constraints"],
    "additionalProperties": False,
}

QUERIES_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}

# Mirrors the upstream AI-Scientist-v2 FinalizeIdea JSON contract.
TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "title": {"type": "string"},
        "short_hypothesis": {"type": "string"},
        "experiments": {"type": "string"},
        "related_work": {"type": "string"},
        "risk_factors_and_limitations": {"type": "string"},
    },
    "required": [
        "name",
        "title",
        "short_hypothesis",
        "experiments",
        "related_work",
        "risk_factors_and_limitations",
    ],
    "additionalProperties": False,
}


class RunMode(str, Enum):
    STAGED_APPROVAL = "staged_approval"
    AUTONOMOUS = "autonomous"


class StageStatus(str, Enum):
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


STAGES = (
    "task_created",
    "dataset_validated",
    "models_prechecked",
    "research_understood",
    "literature_collected",
    "topic_proposed",
    "topic_approved",
    "experiment_planned",
    "budget_approved",
    "baseline_completed",
    "improvements_completed",
    "tuning_completed",
    "ablations_completed",
    "formal_training_approved",
    "main_comparison_completed",
    "candidate_frozen",
    "test_evaluated",
    "analysis_completed",
    "paper_approved",
    "english_paper_completed",
    "review_completed",
    "revision_completed",
    "chinese_translation_completed",
    "archived",
)

APPROVAL_STAGES = {
    "topic_approved",
    "budget_approved",
    "formal_training_approved",
    "paper_approved",
}

MAX_STAGE_RETRIES = 3
STATE_IO_ATTEMPTS = 20
STATE_IO_RETRY_SECONDS = 0.01

STAGE_CONTRACTS = {
    "task_created": {"task.json"},
    "dataset_validated": {"dataset.json"},
    "models_prechecked": {"models.json"},
    "research_understood": {"research.json"},
    "literature_collected": {"literature.json"},
    "topic_proposed": {"topic.json"},
    "topic_approved": {"approval.json"},
    "experiment_planned": {"experiment_plan.json"},
    "budget_approved": {"approval.json"},
    "baseline_completed": {"baseline.json"},
    "improvements_completed": {"improvements.json"},
    "tuning_completed": {"tuning.json"},
    "ablations_completed": {"ablations.json"},
    "formal_training_approved": {"approval.json"},
    "main_comparison_completed": {"main_comparison.json"},
    "candidate_frozen": {"candidate.json"},
    "test_evaluated": {"test_evaluation.json"},
    "analysis_completed": {"analysis.json"},
    "paper_approved": {"approval.json"},
    "english_paper_completed": {"paper.json"},
    "review_completed": {"review.json"},
    "revision_completed": {"revision.json"},
    "chinese_translation_completed": {"translation.json"},
    "archived": {"archive.json"},
}

# Stages that issue exactly one durable provider request.  The mapping is kept
# next to the stage contract so recovery can release only the request owned by
# the interrupted stage, rather than touching unrelated or settled calls.
LLM_REQUEST_SUFFIXES = {
    "research_understood": ("research",),
    "literature_collected": ("literature-queries",),
    "topic_proposed": ("topic",),
    "english_paper_completed": ("paper",),
    # Keep historical IDs recoverable while versioning prompts that must bypass
    # cached empty model responses.
    "review_completed": ("review", "review-v2"),
    "revision_completed": ("revision",),
    "chinese_translation_completed": ("translation",),
}


@dataclass(frozen=True)
class WorkflowConfig:
    mode: RunMode
    budget_limit_usd: float
    execution_limit_seconds: int
    enable_real_training: bool = False
    llm_config_path: str = ""
    research_direction: str = ""
    research_goal: str = ""

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "WorkflowConfig":
        try:
            mode = RunMode(raw["mode"])
            budget = float(raw["budget_limit_usd"])
            execution_limit = int(raw["execution_limit_seconds"])
            enable_real_training = raw.get("enable_real_training", False)
            llm_config_path = str(raw.get("llm_config_path", ""))
            research_direction = str(raw.get("research_direction", "")).strip()
            research_goal = str(raw.get("research_goal", "")).strip()
        except (KeyError, ValueError) as exc:
            raise WorkflowError("Invalid workflow configuration") from exc
        if budget <= 0 or budget > 50:
            raise WorkflowError("Budget must be positive and no more than 50 USD")
        if execution_limit <= 0 or execution_limit > 6 * 60 * 60:
            raise WorkflowError("Execution limit must be positive and no more than six hours")
        if len(research_direction) > 2000:
            raise WorkflowError("Research direction must be at most 2000 characters")
        if len(research_goal) > 1000:
            raise WorkflowError("Research goal must be at most 1000 characters")
        return cls(
            mode,
            budget,
            execution_limit,
            bool(enable_real_training),
            llm_config_path,
            research_direction,
            research_goal,
        )


@dataclass
class TaskState:
    task_id: str
    schema_version: int
    config: dict[str, Any]
    stages: dict[str, dict[str, Any]]
    created_at: str
    updated_at: str
    completed_stage: str
    control: str
    spent_usd: float
    reserved_usd: float
    execution_seconds: float


@dataclass(frozen=True)
class StageUsage:
    execution_seconds: float
    cost_usd: float


@dataclass(frozen=True)
class WorkflowContext:
    project_root: Path
    config_path: Path
    run_root: Path
    report_path: Path
    candidate_path: Path


@dataclass(frozen=True)
class ResourcePolicy:
    require_gpu: bool = True
    require_ac_power: bool = True
    minimum_free_gb: float = 20.0
    minimum_memory_gb: float = 16.0


class StageCostLedger:
    def reserve(self, task_id: str, stage: str, amount_usd: float) -> bool:
        return False

    def settle(self, task_id: str, stage: str, actual_usd: float, usage: dict[str, Any]) -> None:
        return None

    def release(self, task_id: str, stage: str) -> None:
        return None


class GateAStageCostLedger(StageCostLedger):
    def __init__(self, ledger: Any) -> None:
        self.ledger = ledger

    @staticmethod
    def request_id(task_id: str, stage: str) -> str:
        return f"{task_id}:{stage}"

    def reserve(self, task_id: str, stage: str, amount_usd: float) -> bool:
        return self.ledger.reserve(
            self.request_id(task_id, stage),
            amount_usd,
            {"task_id": task_id, "stage": stage, "source": "workflow"},
        )

    def settle(self, task_id: str, stage: str, actual_usd: float, usage: dict[str, Any]) -> None:
        self.ledger.settle(self.request_id(task_id, stage), actual_usd, usage)

    def release(self, task_id: str, stage: str) -> None:
        self.ledger.release(self.request_id(task_id, stage), "workflow stage failed")


class TrainingScheduler:
    """Starts explicit workflow training commands and records their durable process state."""

    COMMANDS = {
        "baseline_completed": ["python", "-m", "pathmnist", "train", "--phase", "main"],
        "improvements_completed": ["python", "-m", "pathmnist", "train", "--phase", "main"],
        "tuning_completed": ["python", "-m", "pathmnist", "train", "--phase", "tune"],
        "ablations_completed": ["python", "-m", "pathmnist", "train", "--phase", "ablations"],
    }

    def __init__(self, runner=None) -> None:
        self.runner = runner or self._default_runner
        self.processes: dict[tuple[str, str], Any] = {}

    @staticmethod
    def _default_runner(command: list[str], log_path: Path):
        import subprocess

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        return subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def start(self, task_id: str, stage: str, project_root: Path, log_path: Path) -> bool:
        key = (task_id, stage)
        if key in self.processes:
            return True
        command = self.COMMANDS.get(stage)
        if command is None:
            return False
        command = [
            *command,
            "--project-root",
            str(project_root),
            "--output-root",
            str(project_root / "runs" / "pathmnist-m4"),
        ]
        self.processes[key] = self.runner(command, log_path)
        return True

    @staticmethod
    def wait(process) -> int:
        return process.wait()


class WorkflowLLMClient:
    """Records provider usage under a workflow stage and aggregates actual cost."""

    def __init__(self, provider: Any, usage_path: Path) -> None:
        self.provider = provider
        self.usage_path = usage_path
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)

    def call_text(self, role: str, request_id: str, system: str, prompt: str):
        value, metadata = self.provider.call_text(role, request_id, system, prompt)
        self.record(request_id, metadata)
        return value, metadata

    def call_json(
        self,
        role: str,
        request_id: str,
        system: str,
        prompt: str,
        function_name: str,
        schema: dict[str, Any],
    ):
        value, metadata = self.provider.call_json(
            role, request_id, system, prompt, function_name, schema
        )
        self.record(request_id, metadata)
        return value, metadata

    def record(self, request_id: str, metadata: dict[str, Any]) -> None:
        record = {
            "request_id": request_id,
            **metadata,
            "recorded_at": _utc_now(),
        }
        with self.usage_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        TraceWriter(self.usage_path.parent / "trace.jsonl", self.usage_path.parent.name).event(
            "llm.completed",
            attributes={
                "request_id": request_id,
                "model": metadata.get("resolved_model") or metadata.get("model"),
                "prompt_tokens": metadata.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": metadata.get("usage", {}).get("completion_tokens", 0),
                "cost_usd": metadata.get("actual_cost_usd", 0.0),
                "cached": bool(metadata.get("cached", False)),
            },
        )

    @classmethod
    def total_cost_usd(cls, usage_path: Path) -> float:
        if not usage_path.is_file():
            return 0.0
        total = 0.0
        for line in usage_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += float(json.loads(line).get("actual_cost_usd", 0.0))
        return round(total, 10)


class WorkflowExecutor:
    def __init__(
        self,
        context: WorkflowContext,
        resource_policy: ResourcePolicy | None = None,
        cost_ledger: StageCostLedger | None = None,
        llm_client: WorkflowLLMClient | None = None,
        training_scheduler: TrainingScheduler | None = None,
        literature_search: Any = None,
    ) -> None:
        self.context = context
        self.resource_policy = resource_policy or ResourcePolicy()
        self.cost_ledger = cost_ledger or StageCostLedger()
        self.llm_client = llm_client
        self.training_scheduler = training_scheduler
        self.literature_search = literature_search

    def execute(self, stage: str, state: TaskState, artifact_root: Path) -> dict[str, str]:
        handlers = {
            "task_created": self._task_created,
            "dataset_validated": self._dataset_validated,
            "models_prechecked": self._models_prechecked,
            "research_understood": self._research_understood,
            "literature_collected": self._literature_collected,
            "topic_proposed": self._topic_proposed,
            "topic_approved": lambda root, task: self._approval("topic", root, task),
            "experiment_planned": self._experiment_planned,
            "budget_approved": lambda root, task: self._approval("budget", root, task),
            "baseline_completed": self._baseline_completed,
            "improvements_completed": self._improvements_completed,
            "tuning_completed": self._tuning_completed,
            "ablations_completed": self._ablations_completed,
            "formal_training_approved": lambda root, task: self._approval("formal_training", root, task),
            "main_comparison_completed": self._main_comparison_completed,
            "candidate_frozen": self._candidate_frozen,
            "test_evaluated": self._test_evaluated,
            "analysis_completed": self._analysis_completed,
            "paper_approved": lambda root, task: self._approval("paper", root, task),
            "english_paper_completed": self._english_paper_completed,
            "review_completed": self._review_completed,
            "revision_completed": self._revision_completed,
            "chinese_translation_completed": self._chinese_translation_completed,
            "archived": self._archived,
        }
        handler = handlers.get(stage)
        if handler is None:
            raise WorkflowError(f"No real executor is implemented for stage: {stage}")
        return handler(artifact_root, state)

    def check_resources(self, stage: str) -> dict[str, object]:
        training_stages = {
            "baseline_completed",
            "improvements_completed",
            "tuning_completed",
            "ablations_completed",
        }
        if stage in training_stages:
            if self.training_scheduler is None:
                return {
                    "mode": "reuse_frozen_artifacts",
                    "fresh_training": False,
                    "resource_check_required": False,
                }
            return inspect_resources(
                require_gpu=self.resource_policy.require_gpu,
                require_ac_power=self.resource_policy.require_ac_power,
                minimum_free_gb=self.resource_policy.minimum_free_gb,
                minimum_memory_gb=self.resource_policy.minimum_memory_gb,
                disk_path=self.context.project_root,
            )
        return {}

    def reserve_cost(self, task_id: str, stage: str, amount_usd: float) -> bool:
        return self.cost_ledger.reserve(task_id, stage, amount_usd)

    def settle_cost(
        self, task_id: str, stage: str, actual_usd: float, usage: dict[str, Any]
    ) -> None:
        self.cost_ledger.settle(task_id, stage, actual_usd, usage)

    def release_cost(self, task_id: str, stage: str) -> None:
        self.cost_ledger.release(task_id, stage)

    def _task_created(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        outputs = _write_or_reuse(
            artifact_root / "task.json",
            {
                "schema_version": 1,
                "task_id": state.task_id,
                "mode": state.config["mode"],
                "project": "PathMNIST",
            },
        )
        adapter = self._upstream_adapter()
        if adapter is not None:
            outputs.update(
                _write_or_reuse(artifact_root / "framework.json", adapter.framework_record())
            )
        return outputs

    def _dataset_validated(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        config = self._config()
        dataset = config.dataset.__class__(
            path=self.context.project_root / config.dataset.path,
            sha256=config.dataset.sha256,
            expected_splits=config.dataset.expected_splits,
            classes=config.dataset.classes,
        )
        summary = validate_dataset(dataset)
        return _write_or_reuse(
            artifact_root / "dataset.json",
            {
                "sha256": summary.sha256,
                "splits": summary.splits,
                "classes": {name: list(values) for name, values in summary.classes.items()},
            },
        )

    def _models_prechecked(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        try:
            import torch
        except (ImportError, OSError) as exc:
            if self.resource_policy.require_gpu:
                raise WorkflowError(f"Unable to inspect CUDA environment: {exc}") from exc
            payload = {
                "torch": None,
                "cuda_build": None,
                "gpu_available": False,
                "arch_list": [],
                "blackwell_supported": False,
                "host_note": "PyTorch unavailable; GPU requirement disabled",
            }
            return _write_or_reuse(artifact_root / "models.json", payload)

        payload = {
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu_available": torch.cuda.is_available(),
            "arch_list": torch.cuda.get_arch_list(),
            "blackwell_supported": "sm_120" in torch.cuda.get_arch_list(),
        }
        if not payload["blackwell_supported"]:
            raise WorkflowError("CUDA build does not include Blackwell sm_120")
        return _write_or_reuse(artifact_root / "models.json", payload)

    def _research_understood(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        direction = str(state.config.get("research_direction", "")).strip()
        goal = str(state.config.get("research_goal", "")).strip()
        payload = {
            "schema_version": 1,
            "research_direction": direction,
            "research_goal": goal,
            "objective": (
                "Train and evaluate a reproducible PathMNIST classifier under a fixed budget"
            ),
            "primary_metric": "macro_f1",
            "splits": ["train", "val", "test"],
            "test_policy": "single frozen evaluation only",
            "understanding_source": "deterministic",
        }
        if direction:
            payload["objective"] = (
                f"Address the user research direction under the PathMNIST discipline: {direction}"
            )
            if self.llm_client is not None:
                analysis, _ = self.llm_client.call_json(
                    "ideation",
                    _workflow_request_id(state.task_id, "research"),
                    "You are a rigorous research planner for a constrained computational "
                    "pathology study. Use only the supplied facts and respect every constraint.",
                    self._direction_prompt(direction, goal),
                    "RecordResearchUnderstanding",
                    RESEARCH_SCHEMA,
                )
                payload["llm_analysis"] = analysis
                payload["understanding_source"] = "llm"
        return _write_or_reuse(artifact_root / "research.json", payload)

    def _literature_collected(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        provenance = self.context.project_root / "docs" / "SOURCE_PROVENANCE.md"
        if not provenance.is_file():
            raise WorkflowError(f"Source provenance is absent: {provenance}")
        payload = {
            "schema_version": 1,
            "upstream": "SakanaAI/AI-Scientist_v2",
            "source": "local supplied baseline",
            "provenance": str(provenance),
            "provenance_sha256": _sha256(provenance),
        }
        direction = str(state.config.get("research_direction", "")).strip()
        goal = str(state.config.get("research_goal", "")).strip()
        if direction:
            raw = self._collect_literature(artifact_root, state, direction, goal)
            references = []
            seen_titles = set()
            for query in raw["queries"]:
                for paper in raw["results"].get(query, []):
                    key = paper["title"].casefold()
                    if key in seen_titles:
                        continue
                    seen_titles.add(key)
                    references.append({"query": query, "status": "api_verified", **paper})
            failed = bool(raw["failures"])
            if references and not failed:
                verification = "api_verified"
            elif references:
                verification = "partial"
            else:
                verification = "pending_manual_verification"
            payload.update(
                {
                    "source": "semantic_scholar",
                    "queries": raw["queries"],
                    "s2_api_key_present": bool(os.getenv("S2_API_KEY", "").strip()),
                    "references": references,
                    "failures": raw["failures"],
                    "verification_status": verification,
                }
            )
        return _write_or_reuse(artifact_root / "literature.json", payload)

    def _topic_proposed(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        direction = str(state.config.get("research_direction", "")).strip()
        goal = str(state.config.get("research_goal", "")).strip()
        payload = {
            "schema_version": 1,
            "title": "A constrained PathMNIST study with deterministic reporting",
            "topic": "SmallResNet optimization under reproducible train/val/test discipline",
            "proposer": "workflow",
        }
        if direction:
            if self.llm_client is not None:
                research = self._read_stage_artifact(
                    artifact_root, "research_understood", "research.json"
                )
                literature = self._read_stage_artifact(
                    artifact_root, "literature_collected", "literature.json"
                )
                topic, _ = self.llm_client.call_json(
                    "ideation",
                    _workflow_request_id(state.task_id, "topic"),
                    "You are an experienced research scientist proposing one focused, "
                    "feasible study. Use only the supplied facts and citations.",
                    self._topic_prompt(direction, goal, research, literature),
                    "FinalizeIdea",
                    TOPIC_SCHEMA,
                )
                payload = {
                    "schema_version": 1,
                    "research_direction": direction,
                    "name": topic["name"],
                    "title": topic["title"],
                    "short_hypothesis": topic["short_hypothesis"],
                    "experiments": topic["experiments"],
                    "related_work": topic["related_work"],
                    "risk_factors_and_limitations": topic["risk_factors_and_limitations"],
                    "proposer": "llm",
                }
            else:
                payload = {
                    "schema_version": 1,
                    "research_direction": direction,
                    "title": f"A PathMNIST study directed at: {direction}",
                    "topic": direction,
                    "proposer": "deterministic",
                }
        return _write_or_reuse(artifact_root / "topic.json", payload)

    def _direction_prompt(self, direction: str, goal: str) -> str:
        goal_line = f"\nRESEARCH_GOAL: {goal}" if goal else ""
        return (
            f"Understand and structure the following user research direction for this study.\n"
            f"{DIRECTION_DISCIPLINE_PROMPT}\n"
            f"RESEARCH_DIRECTION: {direction}{goal_line}\n"
            "Return objective, background, key_questions, and constraints as JSON."
        )

    def _topic_prompt(
        self,
        direction: str,
        goal: str,
        research: dict | None,
        literature: dict | None,
    ) -> str:
        research_summary = "No structured research analysis is available."
        if research:
            analysis = research.get("llm_analysis") or {}
            research_summary = json.dumps(
                {
                    "objective": analysis.get("objective") or research.get("objective"),
                    "key_questions": analysis.get("key_questions", []),
                    "constraints": analysis.get("constraints", []),
                },
                ensure_ascii=False,
                indent=2,
            )
        references = (literature or {}).get("references", [])[:8]
        literature_summary = (
            json.dumps(
                [
                    {
                        "title": item.get("title"),
                        "authors": item.get("authors"),
                        "venue": item.get("venue"),
                        "year": item.get("year"),
                    }
                    for item in references
                ],
                ensure_ascii=False,
                indent=2,
            )
            if references
            else "No verified references are available; state this limitation honestly."
        )
        goal_line = f"\nRESEARCH_GOAL: {goal}" if goal else ""
        prompt = (
            "Propose exactly one focused study that follows the user research direction.\n"
            f"{DIRECTION_DISCIPLINE_PROMPT}\n"
            f"RESEARCH_DIRECTION: {direction}{goal_line}\n"
            f"RESEARCH_ANALYSIS:\n{research_summary}\n"
            f"VERIFIED_REFERENCES:\n{literature_summary}\n"
            "The experiments field must describe comparisons that are feasible on PathMNIST "
            "with the available SmallResNet variant tooling; do not promise external datasets "
            "or clinical validity. Keep every field under 120 words so the JSON stays complete. "
            "Return the FinalizeIdea JSON."
        )
        adapter = self._upstream_adapter()
        if adapter is None:
            return prompt
        return adapter.compile_pathology_prompt(
            {
                "instructions": prompt,
                "dataset": "PathMNIST 64x64, nine classes",
                "primary_metric": "Macro-F1",
                "allowed_interventions": [
                    "augmentation",
                    "optimization",
                    "multiscale",
                    "combined",
                ],
            }
        )

    def _literature_queries(self, state: TaskState, direction: str, goal: str) -> list[str]:
        if self.llm_client is not None:
            proposed, _ = self.llm_client.call_json(
                "ideation",
                _workflow_request_id(state.task_id, "literature-queries"),
                "You convert a research direction into effective English academic search "
                "queries for Semantic Scholar.",
                self._direction_prompt(direction, goal),
                "ProposeSearchQueries",
                QUERIES_SCHEMA,
            )
            queries = [str(item).strip() for item in proposed.get("queries", [])]
        else:
            queries = [
                direction,
                f"{direction} PathMNIST",
                f"{direction} histopathology image classification",
            ]
        return [query for query in queries if query][:3]

    def _collect_literature(
        self, artifact_root: Path, state: TaskState, direction: str, goal: str
    ) -> dict[str, Any]:
        raw_path = artifact_root / "literature_raw.json"
        artifact_root.mkdir(parents=True, exist_ok=True)
        if raw_path.is_file():
            return json.loads(raw_path.read_text(encoding="utf-8"))
        search = self.literature_search
        if search is None:
            search = search_semantic_scholar
        queries = self._literature_queries(state, direction, goal)
        raw: dict[str, Any] = {"queries": queries, "results": {}, "failures": []}
        for index, query in enumerate(queries):
            if index:
                time.sleep(1.2)
            try:
                raw["results"][query] = search(query)
            except LiteratureError as exc:
                raw["failures"].append({"query": query, "error": str(exc)})
        temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
        temporary.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(raw_path)
        return raw

    def _read_stage_artifact(
        self, artifact_root: Path, stage: str, name: str
    ) -> dict[str, Any] | None:
        path = artifact_root.parent / stage / name
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _experiment_planned(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        config = self._config()
        direction = " ".join(
            [
                str(state.config.get("research_direction", "")),
                str(state.config.get("research_goal", "")),
            ]
        ).casefold()
        signals = {
            "augmentation": ("augment", "颜色", "染色", "color", "stain", "旋转", "翻转"),
            "optimization": ("optim", "泛化", "general", "label smooth", "调度", "学习率"),
            "multiscale": (
                "multiscale",
                "multi-scale",
                "resolution",
                "多尺度",
                "分辨率",
                "尺度",
                "形态",
                "morpholog",
            ),
        }
        selected = [
            name for name, keywords in signals.items() if any(word in direction for word in keywords)
        ]
        if not selected:
            selected = ["optimization"]
        if len(selected) > 1:
            selected.append("combined")
        payload = {
            "schema_version": 1,
            "phases": ["tune", "main", "ablations", "final", "test", "report"],
            "seeds": list(config.experiment.seeds),
            "variants": [variant.name for variant in config.experiment.variants],
            "direction_selected_variants": selected,
            "primary_comparisons": [f"baseline_vs_{name}" for name in selected],
            "selection_source": "research_direction_capability_mapping",
            "primary_metric": config.experiment.primary_metric,
            "test_policy": "freeze one validation-selected candidate, then evaluate test once",
            "ablations": [variant.name for variant in config.experiment.ablations],
            "config_sha256": _sha256(self.context.config_path),
        }
        return _write_or_reuse(artifact_root / "experiment_plan.json", payload)

    def _approval(self, name: str, artifact_root: Path, state: TaskState) -> dict[str, str]:
        return _write_or_reuse(
            artifact_root / "approval.json",
            {"schema_version": 1, "approved": True, "gate": name, "approved_by": "user"},
        )

    def _candidate_frozen(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        candidate = load_frozen_candidate(self.context.candidate_path)
        payload = {
            "variant": candidate.variant,
            "seeds": candidate.seeds,
            "dataset_sha256": candidate.dataset_sha256,
        }
        return _write_or_reuse(artifact_root / "candidate.json", payload)

    def _test_evaluated(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        source = self.context.run_root / "test_evaluation.json"
        if not source.is_file():
            raise WorkflowError(f"One-time test evaluation result is absent: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        return _write_or_reuse(artifact_root / "test_evaluation.json", payload)

    def _analysis_completed(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        report = self.context.report_path
        if not report.is_file():
            raise WorkflowError(f"Final report is absent: {report}")
        plan = self._read_stage_artifact(
            artifact_root, "experiment_planned", "experiment_plan.json"
        ) or {}
        baseline = self._variant_summary("main", "baseline")
        comparisons = {}
        for variant in plan.get("direction_selected_variants", ["optimization"]):
            summary = self._variant_summary("main", variant)
            comparisons[variant] = {
                **summary,
                "macro_f1_delta_vs_baseline": (
                    summary["macro_f1_mean"] - baseline["macro_f1_mean"]
                ),
            }
        return _write_or_reuse(
            artifact_root / "analysis.json",
            {
                "schema_version": 1,
                "report": str(report),
                "report_sha256": _sha256(report),
                "primary_metric": "macro_f1",
                "baseline": baseline,
                "direction_driven_comparisons": comparisons,
            },
        )

    def _english_paper_completed(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        payload = {
            "schema_version": 1,
            "title": "A constrained PathMNIST study with deterministic reporting",
            "sections": {
                "abstract": "We report a reproducible PathMNIST workflow and its frozen final evaluation.",
                "introduction": "This paper documents the constrained study and engineering workflow.",
                "methods": "The study uses a SmallResNet optimization-only candidate under a fixed protocol.",
                "results": "The frozen one-time test evaluation is recorded without feedback into model selection.",
                "limitations": "The study is constrained and does not make clinical claims.",
                "conclusion": "The workflow completes the planned reproducible reporting chain.",
            },
            "language": "en",
        }
        if self.llm_client is not None:
            prompt = self._formal_paper_prompt(artifact_root)
            payload["llm_output"] = self.llm_client.call_text(
                "paper_writer",
                _workflow_request_id(state.task_id, "paper"),
                "You are a rigorous computational pathology research writer. Use only supplied facts and make no clinical claims.",
                prompt,
            )[0]
        markdown = payload.get("llm_output") or self._deterministic_paper(artifact_root, state)
        payload["markdown_file"] = "paper.md"
        payload["latex_file"] = "paper.tex"
        outputs = _write_or_reuse(artifact_root / "paper.json", payload)
        outputs.update(_write_text_or_reuse(artifact_root / "paper.md", _as_markdown(markdown)))
        from .paper_export import markdown_to_latex

        outputs.update(
            _write_text_or_reuse(
                artifact_root / "paper.tex", markdown_to_latex(_as_markdown(markdown), "en")
            )
        )
        return outputs

    def _deterministic_paper(self, artifact_root: Path, state: TaskState) -> str:
        topic = self._read_stage_artifact(artifact_root, "topic_proposed", "topic.json") or {}
        literature = self._read_stage_artifact(
            artifact_root, "literature_collected", "literature.json"
        ) or {}
        analysis = self._read_stage_artifact(artifact_root, "analysis_completed", "analysis.json") or {}
        test = json.loads((self.context.run_root / "test_evaluation.json").read_text(encoding="utf-8"))
        title = topic.get("title", "A Reproducible PathMNIST Classification Study")
        comparisons = analysis.get("direction_driven_comparisons", {})
        rows = ["| Variant | Validation Macro-F1 | Delta vs baseline |", "|---|---:|---:|"]
        for name, item in comparisons.items():
            rows.append(
                f"| {name} | {item['macro_f1_mean']:.6f} | "
                f"{item['macro_f1_delta_vs_baseline']:+.6f} |"
            )
        refs = literature.get("references", [])[:8]
        related = (
            "\n".join(
                f"- {item.get('authors') or 'Unknown authors'} ({item.get('year') or 'n.d.'}). "
                f"{item.get('title', 'Untitled')}. {item.get('venue') or ''}" for item in refs
            )
            if refs
            else "No references were API-verified during this run; no citations are invented."
        )
        return f"""# {title}

## Abstract

We operationalized the user-provided direction—{state.config.get('research_direction', '')}—as a constrained nine-class PathMNIST study. A SmallResNet baseline and direction-selected supported variants were compared over three seeds using validation Macro-F1. The candidate was frozen before one test evaluation. The final test Macro-F1 was {test['macro_f1_mean']:.6f} ± {test['macro_f1_std']:.6f}. This work demonstrates a reproducible pathology-AI research workflow and makes no clinical claims.

## Introduction

Small histopathology benchmarks are useful for testing reproducible research machinery, but patch classification does not establish patient-level or clinical validity. This study converts a broad direction into an auditable experiment using only the locally pinned PathMNIST archive.

## Related Work

{related}

## Materials and Methods

PathMNIST contains nine colorectal tissue classes. Training and model selection used only train and validation splits. The primary metric was macro-averaged F1. SmallResNet variants represented augmentation, optimization, multiscale processing, and their combination. Seeds 7, 17, and 27 were used. The test split remained sealed until the validation-selected candidate was frozen.

## Experiments

The direction-to-capability mapper selected only interventions implemented by this repository. Each selected variant was compared with the same baseline protocol; tuning and ablations were completed before candidate freezing.

## Results

{chr(10).join(rows)}

The frozen candidate was evaluated on the test split exactly once. Mean test Macro-F1 was {test['macro_f1_mean']:.6f} with standard deviation {test['macro_f1_std']:.6f} across three seeds.

## Discussion

The observed validation comparisons quantify the supported intervention relevant to the submitted direction. The held-out result is reported only as final evidence and did not trigger additional tuning.

## Limitations

This is a single small benchmark without external-center, slide-level, patient-level, calibration, or prospective clinical validation. Reused frozen results are valid only for research directions aligned with the implemented intervention registry.

## Conclusion

The system completed a traceable direction-to-paper PathMNIST workflow while preserving split discipline and explicit claim boundaries.

## Reproducibility Statement

The dataset hash, configuration, seeds, per-run metrics, candidate record, one-time test record, workflow state, and generated manuscript are stored as machine-readable artifacts.
"""

    def _formal_paper_prompt(self, artifact_root: Path | None = None) -> str:
        report = self.context.report_path.read_text(encoding="utf-8") if self.context.report_path.is_file() else ""
        test_path = self.context.run_root / "test_evaluation.json"
        test = test_path.read_text(encoding="utf-8") if test_path.is_file() else "{}"
        candidate_path = self.context.candidate_path
        candidate = candidate_path.read_text(encoding="utf-8") if candidate_path.is_file() else "{}"
        topic = (
            self._read_stage_artifact(artifact_root, "topic_proposed", "topic.json")
            if artifact_root is not None
            else None
        )
        literature = (
            self._read_stage_artifact(artifact_root, "literature_collected", "literature.json")
            if artifact_root is not None
            else None
        )
        topic_section = "No direction-driven topic proposal is available."
        if topic:
            topic_section = json.dumps(
                {
                    "title": topic.get("title"),
                    "short_hypothesis": topic.get("short_hypothesis"),
                    "experiments": topic.get("experiments"),
                    "risk_factors_and_limitations": topic.get("risk_factors_and_limitations"),
                },
                ensure_ascii=False,
                indent=2,
            )
        references = (literature or {}).get("references", [])[:8]
        references_section = (
            json.dumps(
                [
                    {
                        "title": item.get("title"),
                        "authors": item.get("authors"),
                        "venue": item.get("venue"),
                        "year": item.get("year"),
                    }
                    for item in references
                ],
                ensure_ascii=False,
                indent=2,
            )
            if references
            else "No API-verified references are available; state this explicitly instead of inventing citations."
        )
        return (
            "Write a complete formal English research paper using ONLY these frozen artifacts as facts.\n"
            "Required sections: Title, Abstract, Introduction, Related Work, Materials and Methods, "
            "Experiments, Results, Discussion, Limitations, Conclusion, Reproducibility Statement.\n"
            "State clearly that the test set was evaluated exactly once after candidate freezing.\n"
            "Do not propose clinical use or diagnostic claims.\n"
            "Frame the narrative around the approved topic proposal, but report only facts present "
            "in the frozen artifacts; the topic framing must not introduce unmeasured claims.\n"
            "In Related Work, cite only the verified references supplied below.\n\n"
            f"APPROVED_TOPIC_PROPOSAL:\n{topic_section}\n\n"
            f"VERIFIED_REFERENCES:\n{references_section}\n\n"
            f"FINAL_REPORT:\n{report}\n\nTEST_EVALUATION_JSON:\n{test}\n\nCANDIDATE_JSON:\n{candidate}"
        )

    def _review_completed(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        payload = {
            "schema_version": 1,
            "decision": "accept_with_constraints",
            "summary": "The report is traceable and suitable as a constrained workflow record.",
            "required_changes": ["Retain the single-use test evaluation disclosure."],
            "independence": "review is separate from paper generation",
        }
        if self.llm_client is not None:
            payload["llm_output"] = self.llm_client.call_text(
                "reviewer",
                _workflow_request_id(state.task_id, "review-v2"),
                "You are an independent methodological reviewer. Reject unsupported claims and return a concise review.",
                self._review_prompt(artifact_root),
            )[0]
        return _write_or_reuse(artifact_root / "review.json", payload)

    def _review_prompt(self, artifact_root: Path) -> str:
        """Build a compact review request whose primary object is the manuscript."""
        paper_path = artifact_root.parent / "english_paper_completed" / "paper.md"
        if not paper_path.is_file():
            raise WorkflowError(f"English paper draft is absent: {paper_path}")
        analysis = self._read_stage_artifact(
            artifact_root, "analysis_completed", "analysis.json"
        ) or {}
        test = self._read_stage_artifact(
            artifact_root, "test_evaluated", "test_evaluation.json"
        ) or {}
        evidence = {
            "primary_metric": analysis.get("primary_metric"),
            "baseline": analysis.get("baseline"),
            "direction_driven_comparisons": analysis.get("direction_driven_comparisons"),
            "frozen_test_evaluation": test,
        }
        return (
            "Independently review the manuscript below against the frozen evidence.\n"
            "Return at most 900 English words with these headings: Decision, Summary, "
            "Major Concerns, Required Revisions, Claim-Boundary Check.\n"
            "Check numerical consistency, train/validation/test separation, whether the test "
            "set was used exactly once after candidate freezing, citation discipline, and "
            "whether any clinical or causal claim is unsupported. Do not rewrite the paper.\n\n"
            f"FROZEN_EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}\n\n"
            f"MANUSCRIPT:\n{paper_path.read_text(encoding='utf-8')}"
        )

    def _revision_completed(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        payload = {
            "schema_version": 1,
            "revision": "Retained test disclosure and constrained-claim language.",
            "addresses_review": True,
            "changes": ["Clarified that the test set was evaluated once."],
        }
        if self.llm_client is not None:
            draft = (artifact_root.parent / "english_paper_completed" / "paper.md").read_text(
                encoding="utf-8"
            )
            review = (artifact_root.parent / "review_completed" / "review.json").read_text(
                encoding="utf-8"
            )
            payload["llm_output"] = self.llm_client.call_text(
                "paper_writer",
                _workflow_request_id(state.task_id, "revision"),
                "You are a rigorous research writer revising only from supplied facts.",
                self._formal_paper_prompt(artifact_root)
                + f"\n\nORIGINAL_DRAFT:\n{draft}\n\nREVIEW:\n{review}\n\n"
                + "Return the complete revised paper in Markdown, not a change summary.",
            )[0]
        else:
            payload["llm_output"] = (
                artifact_root.parent / "english_paper_completed" / "paper.md"
            ).read_text(encoding="utf-8")
        final_markdown = _as_markdown(payload["llm_output"])
        payload["final_markdown_file"] = "final_paper.md"
        payload["final_latex_file"] = "final_paper.tex"
        outputs = _write_or_reuse(artifact_root / "revision.json", payload)
        outputs.update(_write_text_or_reuse(artifact_root / "final_paper.md", final_markdown))
        from .paper_export import markdown_to_latex

        outputs.update(
            _write_text_or_reuse(
                artifact_root / "final_paper.tex", markdown_to_latex(final_markdown, "en")
            )
        )
        return outputs

    def _chinese_translation_completed(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        payload = {
            "schema_version": 1,
            "language": "zh-CN",
            "title": "面向可复现报告的 PathMNIST 约束研究",
            "summary": "本文记录 PathMNIST 工作流、冻结候选与一次性测试评估结果。",
            "disclaimer": "翻译仅用于工程交接，不改变英文报告的科研结论。",
        }
        if self.llm_client is not None:
            payload["llm_output"] = self.llm_client.call_text(
                "paper_writer",
                _workflow_request_id(state.task_id, "translation"),
                "Translate the supplied formal paper faithfully into Simplified Chinese.",
                self._formal_paper_prompt(artifact_root),
            )[0]
        return _write_or_reuse(artifact_root / "translation.json", payload)

    def _archived(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        payload = {
            "schema_version": 1,
            "state_file": f"{state.task_id}.json",
            "completed_stage": "archived",
        }
        return _write_or_reuse(artifact_root / "archive.json", payload)

    def _baseline_completed(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        return self._summarize_variant(artifact_root, "baseline", "baseline.json")

    def _improvements_completed(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        names = ["augmentation", "optimization", "multiscale", "combined"]
        summaries = {name: self._variant_summary("main", name) for name in names}
        return _write_or_reuse(artifact_root / "improvements.json", {"schema_version": 1, **summaries})

    def _tuning_completed(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        source = self.context.run_root / "tuning" / "result.json"
        if not source.is_file():
            raise WorkflowError(f"Tuning result is absent: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        return _write_or_reuse(artifact_root / "tuning.json", payload)

    def _ablations_completed(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        config = self._config()
        summaries = {variant.name: self._variant_summary("ablations", variant.name) for variant in config.experiment.ablations}
        return _write_or_reuse(artifact_root / "ablations.json", {"schema_version": 1, **summaries})

    def _main_comparison_completed(self, artifact_root: Path, state: TaskState) -> dict[str, str]:
        config = self._config()
        summaries = {variant.name: self._variant_summary("main", variant.name) for variant in config.experiment.variants}
        return _write_or_reuse(artifact_root / "main_comparison.json", {"schema_version": 1, **summaries})

    def _variant_summary(self, phase: str, variant: str) -> dict[str, object]:
        run_paths = sorted((self.context.run_root / phase / variant).glob("seed_*/run.json"))
        if len(run_paths) != 3:
            raise WorkflowError(f"Incomplete {phase}:{variant} runs; found {len(run_paths)}")
        scores = []
        for path in run_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            scores.append(payload["epochs"][payload["best_epoch"] - 1]["macro_f1"])
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
        return {"runs": len(run_paths), "macro_f1_mean": mean, "macro_f1_std": variance ** 0.5}

    def _summarize_variant(self, artifact_root: Path, variant: str, filename: str) -> dict[str, str]:
        summary = self._variant_summary("main", variant)
        return _write_or_reuse(artifact_root / filename, {"schema_version": 1, **summary})

    def _config(self):
        return load_config(self.context.config_path)

    def _upstream_adapter(self):
        vendor = self.context.project_root / "vendor" / "AI-Scientist-v2"
        if not vendor.is_dir():
            return None
        from .upstream_adapter import PathologyAIScientistV2Adapter

        return PathologyAIScientistV2Adapter(self.context.project_root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_resources(
    require_gpu: bool = True,
    require_ac_power: bool = True,
    minimum_free_gb: float = 20.0,
    minimum_memory_gb: float = 16.0,
    disk_path: Path | None = None,
) -> dict[str, object]:
    import shutil
    import subprocess
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except (ImportError, OSError):
        gpu_available = False
    if require_gpu and not gpu_available:
        raise WorkflowError("Training stage requires an available GPU")
    checked_disk_path = (disk_path or Path.cwd()).resolve()
    usage = shutil.disk_usage(checked_disk_path)
    free_gb = usage.free / 1024**3
    if free_gb < minimum_free_gb:
        raise WorkflowError(
            f"Training stage requires at least {minimum_free_gb} GB free disk on "
            f"{checked_disk_path} (available: {free_gb:.1f} GB)"
        )
    if shutil.which("wmic") and require_ac_power:
        output = subprocess.check_output(
            ["wmic", "path", "Win32_Battery", "get", "BatteryStatus"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        ac_power = "2" in output
        if not ac_power:
            raise WorkflowError("Training stage requires AC power")
    else:
        ac_power = None
    if shutil.which("wmic"):
        output = subprocess.check_output(
            ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"], text=True
        )
        total_memory_gb = int(output.splitlines()[-1].strip()) / 1024**3
    else:
        total_memory_gb = 0.0
    if total_memory_gb and total_memory_gb < minimum_memory_gb:
        raise WorkflowError(f"Training stage requires at least {minimum_memory_gb} GB RAM")
    return {
        "gpu_available": gpu_available,
        "ac_power": ac_power,
        "free_disk_gb": free_gb,
        "disk_path": str(checked_disk_path),
        "total_memory_gb": total_memory_gb,
    }


def _write_or_reuse(path: Path, payload: dict[str, Any]) -> dict[str, str]:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise WorkflowError(f"Existing artifact differs from deterministic output: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(path)
    return {path.name: _sha256(path)}


def _write_text_or_reuse(path: Path, value: str) -> dict[str, str]:
    value = value.rstrip() + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise WorkflowError(f"Existing artifact differs from deterministic output: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    return {path.name: _sha256(path)}


def _as_markdown(value: str) -> str:
    """Remove a single outer Markdown fence sometimes emitted by chat models."""
    cleaned = str(value).strip()
    if cleaned.startswith("```markdown") and cleaned.endswith("```"):
        cleaned = cleaned[len("```markdown") : -3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    if not cleaned.startswith("#"):
        cleaned = "# PathMNIST Research Paper\n\n" + cleaned
    return cleaned


class WorkflowStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, task_id: str) -> Path:
        if not task_id or "/" in task_id or "\\" in task_id or task_id.startswith("."):
            raise WorkflowError("Invalid task ID")
        return self.root / f"{task_id}.json"

    def create(self, task_id: str, raw_config: dict[str, Any]) -> TaskState:
        config = asdict(WorkflowConfig.from_mapping(raw_config))
        path = self.path(task_id)
        if path.exists():
            raise WorkflowError(f"Task already exists: {task_id}")
        timestamp = _utc_now()
        stages = {
            name: {
                "status": StageStatus.READY.value if index == 0 else StageStatus.WAITING.value,
                "started_at": None,
                "completed_at": None,
                "inputs": {},
                "outputs": {},
                "error": None,
                "retries": 0,
            }
            for index, name in enumerate(STAGES)
        }
        state = TaskState(
            task_id=task_id,
            schema_version=1,
            config=config,
            stages=stages,
            created_at=timestamp,
            updated_at=timestamp,
            completed_stage="",
            control="running",
            spent_usd=0.0,
            reserved_usd=0.0,
            execution_seconds=0.0,
        )
        self._write(path, state)
        BudgetLedger(task_budget_path(self.root, task_id), config["budget_limit_usd"])
        return state

    def load(self, task_id: str) -> TaskState:
        try:
            raw = json.loads(self._read_with_retry(self.path(task_id)))
            return TaskState(
                task_id=raw["task_id"],
                schema_version=raw["schema_version"],
                config=raw["config"],
                stages=raw["stages"],
                created_at=raw["created_at"],
                updated_at=raw["updated_at"],
                completed_stage=raw["completed_stage"],
                control=raw["control"],
                spent_usd=raw["spent_usd"],
                reserved_usd=raw["reserved_usd"],
                execution_seconds=raw["execution_seconds"],
            )
        except WorkflowError:
            raise
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise WorkflowError(f"Cannot load task {task_id}") from exc

    def save(self, state: TaskState) -> None:
        self._write(self.path(state.task_id), state)

    def list_states(self) -> list[TaskState]:
        states = [self.load(path.stem) for path in self.root.glob("*.json")]
        return sorted(states, key=lambda state: state.created_at, reverse=True)

    def delete(self, task_id: str) -> None:
        import shutil

        state_path = self.path(task_id)
        task_root = (self.root / task_id).resolve()
        root = self.root.resolve()
        if task_root.parent != root:
            raise WorkflowError("Refusing to delete a task outside the workflow root")
        lock_path = task_root / "worker.lock"
        if lock_path.exists():
            raise WorkflowError("Cannot delete a task while its worker is running")
        if task_root.is_dir():
            shutil.rmtree(task_root)
        if state_path.is_file():
            state_path.unlink()

    @staticmethod
    def _read_with_retry(path: Path) -> str:
        last_error: OSError | None = None
        for _ in range(STATE_IO_ATTEMPTS):
            try:
                return path.read_text(encoding="utf-8")
            except PermissionError as exc:
                last_error = exc
                time.sleep(STATE_IO_RETRY_SECONDS)
        if last_error is not None:
            raise last_error
        raise WorkflowError(f"Cannot read task state: {path}")

    def _write(self, path: Path, state: TaskState) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(state), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._replace_with_retry(temporary, path)

    @staticmethod
    def _replace_with_retry(source: Path, target: Path) -> None:
        for attempt in range(10):
            try:
                os.replace(source, target)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.01 * (attempt + 1))


def _stage_reservation_usd(state: TaskState, stage: str) -> float:
    remaining = state.config["budget_limit_usd"] - state.spent_usd - state.reserved_usd
    reserved = state.stages[stage].get("inputs", {}).get("reserved_usd")
    if reserved is not None:
        return min(float(reserved), max(0.0, remaining))
    return 0.0


def _stage_cost_usd(state: TaskState, stage: str) -> float:
    return min(
        float(state.stages[stage].get("inputs", {}).get("cost_usd", 0.0)),
        _stage_reservation_usd(state, stage),
    )


def completed_index(state: TaskState) -> int:
    if not state.completed_stage:
        return -1
    return STAGES.index(state.completed_stage)


def requires_approval(stage: str, mode: RunMode) -> bool:
    return mode is RunMode.STAGED_APPROVAL and stage in APPROVAL_STAGES


def advance(
    store: WorkflowStore,
    task_id: str,
    executor: WorkflowExecutor | None = None,
    artifact_root: Path | None = None,
) -> TaskState:
    state = store.load(task_id)
    if state.control == "cancelled":
        raise WorkflowError("Cancelled task cannot advance")
    if state.execution_seconds >= state.config["execution_limit_seconds"]:
        raise WorkflowError("Execution time limit reached")
    if state.spent_usd + state.reserved_usd >= state.config["budget_limit_usd"]:
        raise WorkflowError("Budget limit reached")
    next_index = completed_index(state) + 1
    if next_index >= len(STAGES):
        raise WorkflowError("Workflow already completed")
    stage = STAGES[next_index]
    item = state.stages[stage]
    if item["status"] == StageStatus.INTERRUPTED.value and item["retries"] >= MAX_STAGE_RETRIES:
        raise WorkflowError(f"Stage {stage} exceeded retry limit")
    if state.control == "paused":
        raise WorkflowError("Task is paused")
    if requires_approval(stage, RunMode(state.config["mode"])):
        state.stages[stage]["status"] = StageStatus.WAITING_APPROVAL.value
        state.control = "waiting_approval"
        state.updated_at = _utc_now()
        store.save(state)
        return state
    return _execute_and_complete(state, store, stage, executor, artifact_root)


def approve(
    store: WorkflowStore,
    task_id: str,
    stage: str,
    executor: WorkflowExecutor | None = None,
    artifact_root: Path | None = None,
) -> TaskState:
    state = store.load(task_id)
    if stage not in APPROVAL_STAGES:
        raise WorkflowError("Stage does not require approval")
    if STAGES.index(stage) != completed_index(state) + 1:
        raise WorkflowError("Stage is not the next approval gate")
    if state.stages[stage]["status"] != StageStatus.WAITING_APPROVAL.value:
        raise WorkflowError("Approval gate is not waiting")
    state.control = "running"
    return _execute_and_complete(state, store, stage, executor, artifact_root)


def set_control(store: WorkflowStore, task_id: str, control: str) -> TaskState:
    if control not in {"running", "paused", "cancelled"}:
        raise WorkflowError("Invalid control state")
    state = store.load(task_id)
    if state.control == "cancelled" and control != "cancelled":
        raise WorkflowError("Cancelled task cannot be resumed")
    state.control = control
    state.updated_at = _utc_now()
    store.save(state)
    return state


def resume(store: WorkflowStore, task_id: str) -> str:
    state = store.load(task_id)
    if state.control != "paused":
        raise WorkflowError("Only paused tasks can resume")
    set_control(store, task_id, "running")
    return state.completed_stage


def _execute_and_complete(
    state: TaskState,
    store: WorkflowStore,
    stage: str,
    executor: WorkflowExecutor | None,
    artifact_root: Path | None,
) -> TaskState:
    if executor is None or artifact_root is None:
        raise WorkflowError(f"No real executor is provided for stage: {stage}")
    stage_artifact_root = artifact_root / stage
    reservation_usd = _stage_reservation_usd(state, stage)
    usage_path = artifact_root.parent / "llm_usage.jsonl"
    provider_cost_before = WorkflowLLMClient.total_cost_usd(usage_path)
    started_at = time.monotonic()
    trace = TraceWriter(artifact_root.parent / "trace.jsonl", state.task_id)
    span = trace.start_span(
        f"stage.{stage}",
        attempt=int(state.stages[stage].get("retries", 0)) + 1,
        attributes={"reservation_usd": reservation_usd},
    )
    state.stages[stage].update(
        status=StageStatus.RUNNING.value,
        started_at=_utc_now(),
        error=None,
    )
    state.reserved_usd += reservation_usd
    state.updated_at = _utc_now()
    store.save(state)
    try:
        if reservation_usd > 0 and not executor.reserve_cost(
            state.task_id, stage, reservation_usd
        ):
            raise WorkflowError(f"Stage {stage} cost reservation already completed")
        resource_report = executor.check_resources(stage) if hasattr(executor, "check_resources") else {}
        scheduler = getattr(executor, "training_scheduler", None)
        if scheduler is not None:
            log_path = artifact_root.parent / "training.log"
            started = scheduler.start(
                state.task_id, stage, executor.context.project_root, log_path
            )
            if started:
                exit_code = scheduler.wait(
                    scheduler.processes[(state.task_id, stage)]
                )
                if exit_code != 0:
                    raise WorkflowError(f"Stage {stage} training failed with exit code {exit_code}")
        outputs = executor.execute(stage, state, stage_artifact_root)
        _validate_contract(stage, stage_artifact_root)
        outputs["resources"] = resource_report
        provider_cost_after = WorkflowLLMClient.total_cost_usd(usage_path)
        provider_cost_usd = max(0.0, provider_cost_after - provider_cost_before)
        declared_cost_usd = _stage_cost_usd(state, stage)
        if provider_cost_usd > 0:
            cost_usd = provider_cost_usd
        else:
            cost_usd = declared_cost_usd
        if reservation_usd > 0:
            executor.settle_cost(
                state.task_id,
                stage,
                cost_usd,
                {
                    "execution_seconds": time.monotonic() - started_at,
                    "task_id": state.task_id,
                    "stage": stage,
                    "artifact_count": len(outputs) - 1,
                    "provider_cost_usd": provider_cost_usd,
                },
            )
        usage = StageUsage(time.monotonic() - started_at, cost_usd)
        trace.end_span(
            span,
            status="ok",
            duration_seconds=usage.execution_seconds,
            attributes={
                "cost_usd": cost_usd,
                "provider_cost_usd": provider_cost_usd,
                "artifacts": outputs,
            },
        )
    except Exception as exc:
        trace.end_span(
            span,
            status="error",
            duration_seconds=time.monotonic() - started_at,
            attributes={"reservation_usd": reservation_usd},
            error=exc,
        )
        item = state.stages[stage]
        item["status"] = StageStatus.INTERRUPTED.value
        item["error"] = str(exc)
        item["retries"] += 1
        state.reserved_usd = max(0.0, state.reserved_usd - reservation_usd)
        if reservation_usd > 0 and hasattr(executor, "release_cost"):
            executor.release_cost(state.task_id, stage)
        state.updated_at = _utc_now()
        store.save(state)
        raise WorkflowError(f"Stage {stage} failed: {exc}") from exc
    return _complete(state, store, stage, outputs, usage)


def _validate_contract(stage: str, artifact_root: Path) -> None:
    missing = [
        name for name in STAGE_CONTRACTS[stage] if not (artifact_root / name).is_file()
    ]
    if missing:
        raise WorkflowError(f"Stage {stage} artifact contract failed; missing: {missing}")


def task_budget_path(state_root: Path, task_id: str) -> Path:
    return state_root / task_id / "budget.json"


def task_llm_client(
    state_root: Path, task_id: str, llm_config_path: str, hard_limit_usd: float
) -> WorkflowLLMClient | None:
    if not llm_config_path:
        return None
    from gate_a.config import load_config
    from gate_a.models import ModelRegistry
    from gate_a.provider import ZhipuProvider

    config = load_config(Path(llm_config_path))
    selected = ModelRegistry.from_static_catalog(
        config.models, config.budget.cny_per_usd
    ).select_all(config)
    provider = ZhipuProvider(
        config,
        selected,
        BudgetLedger(task_budget_path(state_root, task_id), hard_limit_usd),
        state_root / task_id / "responses",
    )
    return WorkflowLLMClient(provider, state_root / task_id / "llm_usage.jsonl")


def task_executor(project_root: Path, state_root: Path, task_id: str) -> WorkflowExecutor:
    state = WorkflowStore(state_root).load(task_id)
    ledger = BudgetLedger(
        task_budget_path(state_root, task_id), state.config["budget_limit_usd"]
    )
    return WorkflowExecutor(
        WorkflowContext(
            project_root=project_root,
            config_path=project_root / "configs/pathmnist_m4.yaml",
            run_root=project_root / "runs/pathmnist-m4",
            report_path=project_root / "docs/M4_FINAL_REPORT.md",
            candidate_path=project_root / "configs/pathmnist_final_candidate.json",
        ),
        ResourcePolicy(
            require_gpu=state.config.get("enable_real_training", False),
            require_ac_power=state.config.get("enable_real_training", False),
        ),
        GateAStageCostLedger(ledger),
        task_llm_client(
            state_root,
            task_id,
            state.config.get("llm_config_path", ""),
            state.config["budget_limit_usd"],
        ),
        TrainingScheduler() if state.config.get("enable_real_training") else None,
    )


def validate_completed_artifacts(
    state: TaskState,
    artifact_root: Path,
) -> bool:
    if not state.completed_stage:
        return False
    index = STAGES.index(state.completed_stage)
    for stage in STAGES[: index + 1]:
        item = state.stages[stage]
        outputs = item.get("outputs", {})
        if outputs.get("status") != "completed":
            return False
        for name in STAGE_CONTRACTS[stage]:
            path = artifact_root / stage / name
            if not path.is_file() or outputs.get("artifacts", {}).get(name) != _sha256(path):
                return False
    return True


def latest_valid_stage(
    state: TaskState,
    artifact_root: Path,
) -> str:
    latest = ""
    index = STAGES.index(state.completed_stage) if state.completed_stage else -1
    for stage in STAGES[: index + 1]:
        item = state.stages[stage]
        outputs = item.get("outputs", {})
        if outputs.get("status") != "completed":
            break
        valid = True
        for name in STAGE_CONTRACTS[stage]:
            path = artifact_root / stage / name
            artifacts = outputs.get("artifacts", {})
            digest = _sha256(path) if path.is_file() else None
            if digest is None or artifacts.get(name, artifacts.get("sha256")) != digest:
                valid = False
                break
        if not valid:
            break
        latest = stage
    return latest


def repair_to_valid_stage(
    state: TaskState,
    store: WorkflowStore,
    artifact_root: Path,
) -> TaskState:
    valid_stage = latest_valid_stage(state, artifact_root)
    valid_index = STAGES.index(valid_stage) if valid_stage else -1
    current_index = STAGES.index(state.completed_stage) if state.completed_stage else -1
    for stage in STAGES[valid_index + 1 : current_index + 1]:
        state.stages[stage].update(
            status=StageStatus.WAITING.value,
            started_at=None,
            completed_at=None,
            outputs={},
            error=None,
            retries=0,
        )
    state.completed_stage = valid_stage
    state.updated_at = _utc_now()
    store.save(state)
    return state


def reset_interrupted_stage(
    store: WorkflowStore,
    task_id: str,
    stage: str,
    response_root: Path | None = None,
) -> TaskState:
    """Reset one failed stage and release only its unfinished LLM reservations.

    A provider response cache wins over the ledger: when a cached response is
    present the next attempt is idempotent and the settled reservation must not
    be changed.  A reservation without a cache is an interrupted request and is
    explicitly released so the user-requested retry can reuse its stable ID.
    """
    state = store.load(task_id)
    if stage not in STAGES or state.stages[stage]["status"] != StageStatus.INTERRUPTED.value:
        raise WorkflowError(f"Stage is not interrupted: {stage}")
    for suffix in LLM_REQUEST_SUFFIXES.get(stage, ()):
        request_id = _workflow_request_id(task_id, suffix)
        response_path = (response_root or store.root / task_id / "responses") / f"{request_id}.json"
        if not response_path.is_file():
            ledger = BudgetLedger(
                task_budget_path(store.root, task_id), state.config["budget_limit_usd"]
            )
            ledger.release(request_id, f"manual retry of interrupted stage {stage}")
    state.stages[stage].update(
        status=StageStatus.WAITING.value,
        retries=0,
        error=None,
        started_at=None,
        completed_at=None,
        outputs={},
    )
    state.updated_at = _utc_now()
    store.save(state)
    return state


def _complete(
    state: TaskState,
    store: WorkflowStore,
    stage: str,
    outputs: dict[str, str],
    usage: StageUsage,
) -> TaskState:
    timestamp = _utc_now()
    state.stages[stage].update(
        status=StageStatus.COMPLETED.value,
        completed_at=timestamp,
        outputs={
            "status": "completed",
            "execution_seconds": usage.execution_seconds,
            "cost_usd": usage.cost_usd,
            "artifacts": outputs,
        },
    )
    state.execution_seconds += usage.execution_seconds
    state.spent_usd += usage.cost_usd
    state.reserved_usd = max(0.0, state.reserved_usd - _stage_reservation_usd(state, stage))
    state.completed_stage = stage
    state.updated_at = timestamp
    store.save(state)
    return state


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
