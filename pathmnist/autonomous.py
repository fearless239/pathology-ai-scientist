from __future__ import annotations

import ast
import json
import hashlib
import os
import pickle
import re
import sys
import copy
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from gate_a.provider import ChatProvider
from gate_a.policy import CodePolicyError, validate_generated_code
from gate_a.runner import DockerRunner, RunnerError, SandboxCleanupError

from .dataset_adapter import DatasetSpec, materialize_split_view, research_view_interface
from .code_response import query_program
from .tuning_evidence import select_verified_tuning, validate_tuning_record
from .stage_policy import ExperimentExecutionBudget, POLICIES, stage_policy
from .training_budget import training_workload
from .experiment_manifest import ManifestError, TRAINING_POLICY_PROMPT, load_manifest
from .method_spec import (
    attach_method_spec,
    extract_method_spec,
    infer_method_spec,
    parse_method_spec,
    semantic_report,
    classify_requirements,
)
from .scientific_integrity import IntegrityError, record_trusted_evaluation, validate_no_synthetic_dataset
from .autonomous_stages import V2_STAGES as V2_STAGES
from .autonomous_evidence import metric_rows, snapshot_evidence, verified_metrics


class AutonomousExperimentError(RuntimeError):
    pass


def _check_repeated_preflight_failures(journal):
    """Stop unchanged pre-execution failures before more paid generation."""
    recent = journal.nodes[-3:]
    if len(recent) < 3:
        return
    reasons = []
    for node in recent:
        output = getattr(node, "_term_out", []) or []
        text = output if isinstance(output, str) else "".join(output)
        prefix = "Generated experiment rejected before execution: "
        if node.is_buggy is not True or not text.startswith(prefix):
            return
        reasons.append(text[len(prefix):].split(":", 1)[0].strip())
    if len(set(reasons)) == 1:
        raise AutonomousExperimentError(
            "REPEATED_PREFLIGHT_BLOCKED: three consecutive pre-execution failures: "
            + reasons[0] + "; review the contract/code before resuming; no further automatic retries"
        )


def _write_agent_progress(root, stage, journal, history):
    progress = {
        "stage": stage.name, "node_count": len(journal.nodes),
        "good_nodes": len(journal.good_nodes),
        "completed_agent_stages": sorted({item.from_stage for item in history}),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    target = root / "agent_progress.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


DEFAULT_EXPERIMENT_BUDGET = ExperimentExecutionBudget()
SINGLE_CONDITION_BUDGET = ExperimentExecutionBudget(
    max_conditions=1, max_total_epochs=15
)
EXECUTION_POLICY_REVISION = "host-metrics-purpose-boundary-v2"
# Compatibility aliases for callers added during the beta hardening cycle.
TuningExecutionBudget = ExperimentExecutionBudget
DEFAULT_TUNING_BUDGET = DEFAULT_EXPERIMENT_BUDGET


def has_valid_generated_node(journal: Any) -> bool:
    """Return whether a stage journal contains a successful generated child."""
    return any(
        node.is_buggy is False
        and node.parent is not None
        and not node.is_seed_node
        and not node.is_seed_agg_node
        for node in journal.nodes
    )


def _grant_policy_repair_window(manager: Any, attempts: int = 3) -> bool:
    """Grant one bounded retry window when a stricter policy exhausts a stage."""
    active = getattr(manager, "current_stage", None)
    if active is None or not str(active.name).startswith(("3_", "4_")):
        return False
    journal = manager.journals[active.name]
    if has_valid_generated_node(journal) or len(journal.nodes) < active.max_iterations:
        return False
    repairs = dict(getattr(manager, "_path_ai_policy_repairs", {}))
    if repairs.get(active.name) == EXECUTION_POLICY_REVISION:
        return False
    active.max_iterations = len(journal.nodes) + attempts
    repairs[active.name] = EXECUTION_POLICY_REVISION
    manager._path_ai_policy_repairs = repairs
    return True


def _expected_contract_role(stage_name: str) -> str | None:
    policy = stage_policy(stage_name)
    return policy.role if policy else None


def _allowed_contract_roles(stage_name: str) -> set[str]:
    expected = _expected_contract_role(stage_name)
    roles = {expected} if expected else set()
    if stage_name.startswith("2_"):
        roles.add("hyperparameter_tuning")
    return roles


def _validate_stage_semantics(code: str, stage_name: str, signals: list[str]) -> None:
    """Reject contract-role leakage before an expensive sandbox execution."""
    role = _expected_contract_role(stage_name)
    method_spec = extract_method_spec(code)
    evidence = semantic_report(code, signals, method_spec)
    implemented = [signal for signal, lines in evidence.get("signals", {}).items() if lines]
    generic_preprocessing = {"transform", "transforms", "augmentation", "augment",
                             "conv1", "conv2", "conv3", "cross_entropy", "num_classes"}
    baseline_interventions = [
        signal for signal in implemented if signal.casefold() not in generic_preprocessing
    ]
    if role == "baseline" and baseline_interventions:
        raise IntegrityError(
            "Baseline stage implements intervention signals: "
            f"{sorted(baseline_interventions)}"
        )
    if role == "proposed_method" and not evidence.get("passed"):
        if evidence.get("status") == "needs_review":
            raise IntegrityError(
                "Proposed-method code needs review for unknown components: "
                f"{evidence.get('unknown', [])}"
            )
        raise IntegrityError(
            "Proposed-method stage does not implement every approved intervention signal; "
            f"required={evidence.get('required', [])}, "
            f"detected={evidence.get('detected', [])}, "
            f"missing={evidence.get('missing', [])}"
        )
    if stage_name.startswith(("2_", "3_", "4_")):
        _validate_experiment_execution_budget(code, stage_name)
    if stage_name.startswith("4_"):
        if re.search(r"multi[-_ ]dataset|cross[-_ ]dataset", code, flags=re.IGNORECASE):
            raise IntegrityError(
                "Single-dataset contract forbids multi-dataset ablation or generalization"
            )
        # A valid ablation can disable the only component. Positive-method
        # detection deliberately excludes zero/False values; requiring it here
        # would reject exactly the removed-component experiment we requested.
        disabled = set()
        normalized_signals = {re.sub(r'[^a-z0-9]', '', signal.casefold()): signal for signal in signals}
        for item in ast.walk(ast.parse(code)):
            if isinstance(item, ast.keyword) and item.arg and isinstance(item.value, ast.Constant):
                key = re.sub(r'[^a-z0-9]', '', item.arg.casefold())
                if key in normalized_signals and item.value.value in (0, False, None):
                    disabled.add(normalized_signals[key])
        if signals and not implemented and not disabled:
            raise IntegrityError(
                "Component-ablation stage does not reference an approved intervention component"
            )


def _literal_assignments(code: str) -> dict[str, Any]:
    """Return safe module-level literal assignments from generated code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        try:
            values[targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return values


def _assigned_sequence_length(code: str, names: tuple[str, ...]) -> int | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if (
            len(targets) == 1
            and isinstance(targets[0], ast.Name)
            and targets[0].id in names
            and isinstance(value, (ast.List, ast.Tuple))
        ):
            return len(value.elts)
    return None


def _validate_experiment_execution_budget(
    code: str,
    stage_name: str,
    budget: ExperimentExecutionBudget = DEFAULT_EXPERIMENT_BUDGET,
) -> None:
    """Reject an obviously oversized generated training program before GPU use."""
    if stage_name.startswith(("3_", "4_")) and budget == DEFAULT_EXPERIMENT_BUDGET:
        budget = stage_policy(stage_name).budget
    values = _literal_assignments(code)
    sequence_names = (
        ("learning_rates", "candidate_values", "candidates")
        if stage_name.startswith("2_")
        else (
            "ablation_configs",
            "experiment_configs",
            "conditions",
            "variants",
            "candidate_values",
            "candidates",
        )
    )
    condition_count = _assigned_sequence_length(code, sequence_names)
    seed_count = _assigned_sequence_length(
        code, ("seeds", "random_seeds", "experiment_seeds", "repeat_seeds")
    )
    epochs = values.get("num_epochs", values.get("epochs_per_candidate"))
    if condition_count is not None and condition_count > budget.max_conditions:
        raise IntegrityError(
            "Generated experiment exceeds the per-sandbox condition limit: "
            f"{condition_count} > {budget.max_conditions}"
        )
    if condition_count is not None and isinstance(epochs, int):
        total_epochs = condition_count * epochs
        if total_epochs > budget.max_total_epochs:
            raise IntegrityError(
                "Generated experiment exceeds the per-sandbox epoch budget: "
                f"{total_epochs} > {budget.max_total_epochs}"
            )
    if stage_name.startswith(("3_", "4_")):
        if seed_count is not None and seed_count > 1:
            raise IntegrityError(
                "Generated experiment must consume exactly one host-injected seed per "
                f"sandbox; found an internal sequence of {seed_count} seeds"
            )
        labels = _generated_training_condition_labels(code)
        forbidden = "baseline" if stage_name.startswith("3_") else "full"
        if any(forbidden in label for label in labels):
            raise IntegrityError(
                f"{stage_name.split('_', 1)[0]} stage must reuse the trusted "
                f"{forbidden} artifact instead of retraining it"
            )
        # Helper structure is not a reliable training-launch contract. The runtime
        # validates explicit training_runs; this estimate is diagnostic only.
        try:
            bounded_count, bounded_epochs = training_workload(code)
        except IntegrityError:
            bounded_count, bounded_epochs = None, None
        if len(labels) > budget.max_conditions:
            raise IntegrityError(
                "Generated experiment invokes too many training conditions in one sandbox: "
                f"{sorted(labels)}"
            )
        return {'estimated_launches': bounded_count, 'estimated_epochs': bounded_epochs}


def _generated_training_condition_labels(code: str) -> set[str]:
    """Find labels passed to generated helper functions that perform training."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    training_functions: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in {"backward", "step"}
            for child in ast.walk(node)
        ):
            training_functions.add(node.name)
    labels: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            name = child.func.id if isinstance(child.func, ast.Name) else None
            if name not in training_functions:
                continue
            strings = [
                item.value.casefold()
                for item in [*child.args, *(kw.value for kw in child.keywords)]
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            ]
            labels.update(strings or {name.casefold()})
    return labels


def _generated_training_invocation_count(code: str) -> int:
    """Count module-level launches of generated training logic."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    training_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in {"backward", "step"}
            for child in ast.walk(node)
        )
    }
    count = 0
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        helper_calls = [
            child for child in ast.walk(statement)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id in training_functions
        ]
        if helper_calls:
            count += len(helper_calls)
            continue
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in {"backward", "step"}
            for child in ast.walk(statement)
        ):
            count += 1
    return count


def _validate_tuning_execution_budget(
    code: str, budget: TuningExecutionBudget = DEFAULT_TUNING_BUDGET
) -> None:
    """Backward-compatible entry point for the stage-2 budget guard."""
    _validate_experiment_execution_budget(code, "2_baseline_tuning", budget)


def _preserve_host_injected_seed(code: str) -> str:
    """Neutralize duplicate seed assignments without emptying suites or editing locals."""
    marker = "# Set random seed"
    if marker not in code:
        return code
    tree = ast.parse(code)
    assignments = []

    class Seeds(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            pass

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'seed' and isinstance(node.value, ast.Constant)
                    and type(node.value.value) is int):
                assignments.append(node)

    Seeds().visit(tree)
    assignments.sort(key=lambda node: (node.lineno, node.col_offset))
    lines = code.encode('utf-8').splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    result = code.encode('utf-8')
    for node in reversed(assignments[1:]):
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        result = result[:start] + b'pass' + result[end:]
    normalized = result.decode('utf-8')
    compile(normalized, '<seed-normalized>', 'exec')
    return normalized


def _normalize_tuning_manifest(path: Path) -> bool:
    """Map explicit per-run aliases to required singular v2 fields."""
    value = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    if "learning_rate" not in value and isinstance(value.get("best_learning_rate"), (int, float)):
        value["learning_rate"] = value["best_learning_rate"]
        changed = True
    if "epochs" not in value and not ({"max_epochs", "early_stopping"} & value.keys()):
        for alias in ("epochs_per_candidate", "epochs_per_condition"):
            if isinstance(value.get(alias), int) and value[alias] > 0:
                value["epochs"] = value[alias]
                changed = True
                break
    if changed:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def _has_indexed_high_subset(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript):
            if any(isinstance(target, ast.Name) and target.id == "selected_high_inputs" for target in node.targets):
                return True
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "selected_high_inputs"
            and isinstance(node.value, ast.Subscript)
        ):
            return True
    return False


@dataclass(frozen=True)
class AutonomousTaskWorkspace:
    root: Path
    dataset: Path
    research: Path
    experiment_workspace: Path
    experiment_logs: Path
    candidates: Path
    final_evaluation: Path
    paper: Path

    @classmethod
    def create(cls, state_root: Path, task_id: str) -> "AutonomousTaskWorkspace":
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", task_id):
            raise AutonomousExperimentError("Task ID must be filesystem-safe")
        root = (state_root / task_id).resolve()
        if state_root.resolve() not in root.parents:
            raise AutonomousExperimentError("Task workspace escapes the workflow root")
        values = [root / name for name in ("dataset", "research", "experiment_workspace", "experiment_logs", "candidates", "final_evaluation", "paper")]
        for path in values:
            path.mkdir(parents=True, exist_ok=True)
        return cls(root, *values)

    def prepare_research_dataset(self, spec: DatasetSpec) -> Path:
        view = self.dataset / "research_view"
        if view.exists() and any(view.iterdir()):
            raise AutonomousExperimentError("Research dataset view already exists; refusing to mix task data")
        return materialize_split_view(spec, view, {"train", "validation"})


def pathology_task_description(
    direction: str, spec: DatasetSpec, contract: dict[str, Any] | None = None,
    *, research_view: Path | None = None,
) -> str:
    """Build the upstream AgentManager idea contract from data facts, not interventions."""
    if not direction.strip():
        raise AutonomousExperimentError("Research direction is required")
    profile = {
        "source_type": spec.source_type,
        "image_shape": spec.image_shape,
        "channels": spec.channels,
        "classes": spec.classes,
        "split_counts": {key: value for key, value in spec.split_counts.items() if key != "test"},
        "recommended_metrics": spec.recommended_metrics,
        "warnings": spec.warnings,
    }
    title = "Agent-designed supervised pathology image classification study"
    abstract = direction.strip()
    short_hypothesis = direction.strip()
    contract_experiments = None
    if contract:
        dataset_name = (
            "PathMNIST"
            if "pathmnist" in f"{spec.name} {direction}".casefold()
            else spec.name
        )
        baseline_name = str(contract["baseline"]["name"])
        intervention = contract["interventions"][0]
        signal_names = {
            "color_jitter": "color perturbation",
            "rotation": "rotation",
            "flip": "horizontal/vertical flipping",
            "confidence": "confidence-based hard-example selection",
            "hard": "hard-example mining",
            "contrastive": "supervised contrastive learning",
            "temperature": "temperature-scaled contrastive loss",
        }
        components = [
            signal_names[signal]
            for signal in intervention.get("implementation_signals", [])
            if signal in signal_names
        ]
        method_summary = ", ".join(dict.fromkeys(components)) or str(intervention["name"])
        primary = str(contract["metrics"]["primary"]["name"])
        criterion = contract["success_criteria"][0]
        threshold = criterion.get("minimum_delta")
        threshold_text = (
            f" by at least {float(threshold) * 100:.1f} percentage points"
            if isinstance(threshold, (int, float))
            else ""
        )
        seeds = contract["repeat_plan"]["seeds"]
        title = f"{dataset_name} classification with {method_summary}"
        short_hypothesis = (
            f"Relative to the fixed {baseline_name} baseline, adding {method_summary} "
            f"will improve validation {primary}{threshold_text} under identical data splits, "
            "training controls, and paired seeds."
        )
        abstract = (
            f"This study tests data augmentation for {dataset_name} {len(spec.classes)}-class "
            f"classification using {baseline_name}. The intervention combines {method_summary}. "
            f"Baseline and intervention results are compared on the validation split using paired "
            f"seeds {seeds}, with {primary} as the primary metric. Candidate selection never accesses "
            "the sealed test split; one approved held-out evaluation is reserved for final confirmation."
        )
        contract_experiments = [
            f"Trusted baseline reference [{contract['baseline']['id']}]: {str(contract['baseline']['description']).rstrip('.')}. It is executed in the baseline stage and must not be retrained by proposed-method or ablation sandboxes.",
            f"Required intervention [{intervention['id']}]: {method_summary}. {POLICIES[3].prompt()}",
            *[
                f"Required host-side comparison [{item['id']}]: pair trusted {item['baseline_id']} artifacts with {item['intervention_id']} artifacts by seed; do not train both conditions in one sandbox"
                for item in contract["comparisons"] if item.get("required", True)
            ],
            f"Use primary metric {contract['metrics']['primary']['name']} with scope {contract['metrics']['primary']['scope']}.",
            f"Run the required final comparison for seeds {contract['repeat_plan']['seeds']} on the fixed split.",
        ]
    return json.dumps(
        {
            "Title": title,
            "Abstract": abstract,
            "Short Hypothesis": short_hypothesis,
            "Experiments": contract_experiments or [
                "Generate and execute a reproducible baseline using only /dataset train and validation.",
                "Tune the baseline without changing the split or selection metric.",
                "Design a research method that directly tests the submitted hypothesis.",
                "Ablate the generated method's essential components.",
            ],
            "Risk Factors and Limitations": [
                "The test split is not mounted and must never be requested or reconstructed.",
                "Do not download or use another dataset.",
                "Do not make clinical claims from patch-level classification.",
            ],
            "Dataset Profile": profile,
            "Image preprocessing": {
                "source_image_shape": spec.image_shape,
                "instruction": "The source resolution may differ from the required model input. Resize consistently in train, validation and inference preprocessing. Preserve original dataset identity and report source resolution, model input resolution and interpolation; never describe resized data as originally acquired at the target resolution.",
            },
            "Approved Research Execution Contract": contract,
            "Requirement classification": (
                {item['id']: classify_requirements(item.get('implementation_signals', []))
                 for item in contract.get('interventions', [])} if contract else {}
            ),
            "Research View Interface": research_view_interface(spec, research_view),
        },
        ensure_ascii=False,
    )


class GatewayQueryAdapter:
    """Translate all upstream code/feedback/summary queries to the Gate A provider ledger."""

    def __init__(self, provider: ChatProvider, task_id: str):
        self.provider = provider
        self.task_id = task_id
        self.sequence = 0
        ledger_path = getattr(getattr(provider, "ledger", None), "path", None)
        if ledger_path is not None and Path(ledger_path).is_file():
            try:
                ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
                prefix = f"{task_id}-agent-v2-"
                self.sequence = max(
                    (
                        int(match.group(1))
                        for request_id in ledger.get("requests", {})
                        if request_id.startswith(prefix)
                        and (match := re.search(r"-call(\d+)$", request_id))
                    ),
                    default=0,
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                self.sequence = 0

    def __call__(self, system_message=None, user_message=None, model="", max_tokens=None, func_spec=None, **kwargs):
        if getattr(func_spec, "name", "") == "parse_metrics":
            output = _prompt_text(system_message)
            rows = []
            for name, raw in re.findall(
                r"(?:'|\")?([A-Za-z][A-Za-z0-9 _-]{1,80})(?:'|\")?\s*:\s*(-?\d+(?:\.\d+)?)",
                output,
            ):
                metric_name = name.strip().replace("_", " ")
                rows.append({
                    "metric_name": metric_name,
                    "lower_is_better": "loss" in metric_name.casefold(),
                    "description": "Host-validated structured experiment metric",
                    "data": [{"dataset_name": "validation", "final_value": float(raw), "best_value": float(raw)}],
                })
            return {"valid_metrics_received": bool(rows), "metric_names": rows}
        self.sequence += 1
        system = _prompt_text(system_message) or "You are the AI Scientist v2 experiment agent."
        prompt = _prompt_text(user_message)
        fingerprint = hashlib.sha256(
            (system + "\n" + prompt + "\n" + str(getattr(func_spec, "name", "text"))).encode("utf-8")
        ).hexdigest()[:20]
        role = "experiment_code" if func_spec is None else "ideation"
        role_config = getattr(getattr(self.provider, "config", None), "roles", {}).get(role)
        output_limit = getattr(role_config, "max_output_tokens", "default")
        # The occurrence suffix prevents upstream's extraction retries from
        # replaying one malformed cached completion. Including the configured
        # output ceiling invalidates responses truncated under an older limit.
        request_id = (
            f"{self.task_id}-agent-v2-{fingerprint}-out{output_limit}-call{self.sequence}"
        )
        if func_spec is not None and hasattr(self.provider, "call_json"):
            value, _ = self.provider.call_json(
                role,
                request_id,
                system,
                prompt,
                func_spec.name,
                func_spec.json_schema,
            )
            return value
        value, _ = self.provider.call_text(role, request_id, system, prompt)
        return value


def _prompt_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


class AIScientistExperimentRunner:
    """Run the pinned upstream AgentManager with documented reliability patches."""

    def __init__(self, project_root: Path, provider: ChatProvider, docker_runner: DockerRunner, *, require_dynamic_audit: bool = False):
        self.project_root = project_root.resolve()
        self.provider = provider
        self.docker_runner = docker_runner
        self.require_dynamic_audit = require_dynamic_audit
        self.vendor_root = self.project_root / "vendor" / "AI-Scientist-v2"
        if not self.vendor_root.is_dir():
            raise AutonomousExperimentError("Vendored AI-Scientist-v2 is missing")

    def run(self, task_id: str, direction: str, spec: DatasetSpec, workspace: AutonomousTaskWorkspace, cfg: Any) -> Any:
        if str(self.vendor_root) not in sys.path:
            sys.path.insert(0, str(self.vendor_root))
        research_view = workspace.dataset / "research_view"
        if not research_view.is_dir():
            workspace.prepare_research_dataset(spec)
        cfg.workspace_dir = str(workspace.experiment_workspace)
        cfg.log_dir = workspace.experiment_logs
        cfg.data_dir = str(research_view)
        cfg.copy_data = False
        cfg.agent.num_workers = 1
        cfg.experiment.num_syn_datasets = 1
        from .research_contract import load_contract

        contract = load_contract(workspace.root, require_approved=True)
        cfg.agent.contract_metric = contract["metrics"]["primary"]["name"]
        from .stage_policy import enabled_stages
        cfg.agent.enabled_stages = enabled_stages(contract)
        cfg.agent.repeat_seeds = contract['repeat_plan']['seeds']
        task_desc = pathology_task_description(direction, spec, contract, research_view=research_view)
        query_adapter = GatewayQueryAdapter(self.provider, task_id)
        manager_class, agent_class, minimal_agent_class, interpreter_class = self._runtime_classes(
            research_view,
            validation_count=spec.split_counts["validation"],
            intervention_signals=tuple(
                contract["interventions"][0].get("implementation_signals", [])
            ),
        )
        checkpoint = workspace.experiment_logs / "manager.pkl"
        # The lightweight control-plane image deliberately has neither CUDA nor
        # nvidia-smi.  GPU capability belongs to the nested experiment sandbox,
        # which is verified separately before paid execution.  Tell upstream's
        # worker scheduler about the single validated sandbox GPU so it does not
        # incorrectly switch its scheduling metadata to CPU-only mode.
        with _patched_upstream(
            query_adapter,
            agent_class,
            minimal_agent_class,
            interpreter_class,
            sandbox_gpu_count=1,
        ):
            manager = self._load_or_create_manager(checkpoint, manager_class, task_desc, cfg, workspace)
            def checkpoint_and_progress(stage, journal):
                self._checkpoint(checkpoint, manager)
                _write_agent_progress(workspace.root, stage, journal, manager.stage_history)
            manager.run(exec_callback=lambda *args, **kwargs: interpreter_class(*args, **kwargs), step_callback=checkpoint_and_progress)
            self._checkpoint(checkpoint, manager)
        return manager

    def _runtime_classes(
        self,
        research_view: Path,
        validation_count: int | None = None,
        intervention_signals: tuple[str, ...] = (),
    ):
        # This method is also used by offline preflight and checkpoint recovery,
        # which may run before ``run`` has had a chance to prepare sys.path.
        if str(self.vendor_root) not in sys.path:
            sys.path.insert(0, str(self.vendor_root))
        from ai_scientist.treesearch.agent_manager import AgentManager
        from ai_scientist.treesearch.interpreter import ExecutionResult
        from ai_scientist.treesearch.parallel_agent import MinimalAgent, ParallelAgent

        docker_runner = self.docker_runner
        require_dynamic_audit = self.require_dynamic_audit

        class PathologyPromptMixin:
            def plan_and_code_query(self, prompt, retries=3):
                from ai_scientist.treesearch import parallel_agent
                metric_lock = research_view.parent.parent / 'research/locked_metric.json'
                if metric_lock.is_file():
                    prompt = {'Task': prompt, 'Locked validation metric subgroup': json.loads(metric_lock.read_text(encoding='utf-8'))}
                return query_program(
                    parallel_agent.query, prompt, model=self.cfg.agent.code.model,
                    temperature=self.cfg.agent.code.temp, retries=retries,
                )

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                os.environ["PATH_AI_SCIENTIST_AGENT_STAGE"] = str(self.stage_name or "")

            @property
            def _prompt_environment(self):
                return {"Environment": "A single local supervised image dataset is mounted read-only at /dataset. The container has no network. Use only preinstalled packages and write only beneath the current workspace."}

            @property
            def _prompt_hyperparam_tuning_resp_fmt(self):
                base = super()._prompt_hyperparam_tuning_resp_fmt
                budget = DEFAULT_TUNING_BUDGET
                base["Tuning execution and recovery contract"] = (
                    f"Evaluate exactly two candidate values and no more than {budget.max_conditions}; include the "
                    f"baseline value. Candidate count multiplied by epochs per candidate must not exceed "
                    f"{budget.max_total_epochs}. Keep all non-tuned training conditions unchanged. After every "
                    "completed candidate, atomically replace working/tuning_progress.json with completed "
                    "candidates, validation metrics, and the next candidate index. Flush every epoch's console "
                    "output. A partial progress file is diagnostic only; write the final experiment manifest "
                    "only after the complete search succeeds."
                )
                base['Tuning evidence schema'] = (
                    'After BOTH candidates finish, write working/tuning_evidence.json with schema_version=1, '
                    'complete=true, seed, selection_metric, selected_learning_rate, and candidates=[{learning_rate, '
                    'validation_metric, history:[{epoch:1, train_loss, validation_loss, validation_metric}, ...]}, ...]. '
                    'History epochs are consecutive from 1. Each candidate also declares selected_epoch, chosen by '
                    'checkpoint_selection (minimum validation_loss or maximum validation_metric). Its validation_metric '
                    'is the PRIMARY metric at that selected epoch, not necessarily its maximum across epochs. '
                    'Restore that checkpoint; select learning rate by the highest candidate PRIMARY metric. '
                    'Include the baseline learning rate. Preserve the model class definitions, constructor, '
                    'dataset, optimizer, max_epochs, early_stopping, batch size, input resolution and seed. '
                    + TRAINING_POLICY_PROMPT + ' No improvement is required '
                    'to complete tuning; a valid negative result is acceptable. Progress files alone are not completion evidence.'
                )
                base["MethodSpec"] = (
                    "Begin the natural-language section with `METHOD_SPEC: ` followed by one compact JSON "
                    "object containing hypothesis, components (id, category, implementation_symbols), changes, "
                    "and preserved fields. The following Python remains free-form."
                )
                return base

            @property
            def _prompt_ablation_resp_fmt(self):
                base = super()._prompt_ablation_resp_fmt
                base["Single-component ablation contract"] = (
                    "Use the sole mounted dataset only. Train exactly one condition: the accepted method with "
                    "exactly one approved intervention component removed or disabled. The host pairs this result "
                    "with the trusted full-method artifact for the same seed; never retrain the full method. "
                    "Do not propose multi-dataset or cross-dataset generalization. Keep architecture, split, seeds, "
                    "optimizer, learning rate, epoch count, and metrics fixed. "
                    f"Use no more than {SINGLE_CONDITION_BUDGET.max_total_epochs} total training epochs in this sandbox. Save "
                    "complete structured evidence and the component name; partial output is never success."
                )
                base["MethodSpec"] = (
                    "Begin with `METHOD_SPEC: ` and a compact JSON object describing the component removed, "
                    "its implementation symbols, changed fields, and preserved controls."
                )
                return base

            @property
            def _prompt_debug_resp_fmt(self):
                base = super()._prompt_debug_resp_fmt
                stage_name = str(getattr(self, "stage_name", ""))
                if stage_name.startswith("2_"):
                    base["Tuning evidence schema"] = self._prompt_hyperparam_tuning_resp_fmt["Tuning evidence schema"]
                if stage_name.startswith("3_"):
                    base["Proposed-method repair boundary"] = POLICIES[3].prompt()
                if stage_name.startswith("4_"):
                    base["Ablation repair boundary"] = (
                        "Repair this as a single-dataset, single-component ablation. Train exactly one variant with "
                        "one approved component disabled. Never retrain the trusted full method, retain a "
                        "multi-dataset study, or exceed 15 epochs."
                    )
                base["MethodSpec repair"] = (
                    "Begin with `METHOD_SPEC: ` and a compact JSON object whose components and "
                    "implementation_symbols match the repaired free-form code."
                )
                return base

            @property
            def _prompt_impl_guideline(self):
                base = super()._prompt_impl_guideline
                base["Dataset and evidence constraints"] = [
                    "Keep intervention methods separate from architecture, initialization, data strategy and evaluation requirements. test_accuracy belongs to held-out evaluation only, never research training. Do not invent calls or rename metadata to satisfy a symbol checker.",
                    "Prefer torch.nn.CrossEntropyLoss(label_smoothing=...) or torch.nn.functional.cross_entropy over a custom label-smoothing implementation. Custom losses require sandbox value, gradient and backward verification. Standard metadata categories such as cnn_architecture, data_loading and classification_metrics are valid; never relabel them as rotation or flip.",
                    "Use only /dataset/dataset.npz or /dataset/manifest.json.",
                    "Use the exact exported keys in Research View Interface; source NPZ aliases do not apply to the mounted view. Inspect data.files before indexing.",
                    "Unconditional top-level imports are prechecked in the actual sandbox before any training. Use exact installed API names; an IMPORT_PREFLIGHT_FAILED diagnostic includes similar available names. Repair the API name, never replace the approved architecture or silently disable pretrained weights.",
                    "Only train and validation exist; never synthesize, infer, download, or search for test data.",
                    "Do not install packages or invoke subprocesses.",
                    "Write working/experiment_result.json with the standard metrics and test_data_accessed=false.",
                    "The result JSON must include metrics, predictions, targets, and sample_ids for the complete validation split; probabilities are optional. Load NPZ sample IDs from validation_sample_ids or preserve manifest IDs in evaluation order.",
                    "Every program must support training and inference-only execution. For NPZ input define exactly `HAS_TRAIN_SPLIT = \"train_images\" in data.files` (or use an equivalent manifest split check), and never require train arrays when it is false.",
                    "When HAS_TRAIN_SPLIT is true, train normally and save the final selected model to working/model_checkpoint.pt. When false, load /workspace/model_checkpoint.pt, skip all optimizer/training code, evaluate validation_* arrays only, and still export the complete per-sample result contract.",
                    "The launcher checks actual image tensors at model entry against approved input sizes. Resize in preprocessing; reporting a different resolution in a manifest is not sufficient.",
                    "Inference is fail-closed: the required checkpoint must exist, torch.load must succeed and load_state_dict must completely restore the model before any image forward. Never fall back to random weights, suppress loading errors or continue after a MODEL_CONTRACT_FAILED error.",
                    "Write working/experiment_manifest.json with schema_version=1 and exact dataset, model, optimizer, learning_rate, epochs, batch_size, seed, input_resolutions, selection_metric, and hardware fields.",
                    "Write working/contract_execution.json binding this run to the approved contract role and comparison ID, with the actual training seed. Do not claim a method component that is absent from the code.",
                    "Make the program seed-injectable: use an already-defined global `seed` when present and only default it when absent. Never overwrite the seed prepended by the multi-seed evaluator, and report that effective seed in every output manifest/result.",
                    "For tuning runs, the singular learning_rate is the validation-selected numeric value. " + TRAINING_POLICY_PROMPT,
                    "The model, loss, optimizer, routing, and ablations must be generated from the research hypothesis, not selected from a registry.",
                    "Each execution has a 60-minute sandbox ceiling. Use convergence-aware training and early stopping; avoid duplicate training when a checkpoint or evaluation-only control is scientifically equivalent.",
                ]
                base["MethodSpec protocol"] = [
                    "Begin the response plan with `METHOD_SPEC: ` and one compact JSON object.",
                    "The object must contain hypothesis, components, changes, and preserved fields.",
                    "Each component must contain id, category, and implementation_symbols used by the free-form code.",
                    "For standard components use canonical categories color_perturbation, rotation, and flip; novel categories remain allowed but require review.",
                    "MethodSpec describes intent and evidence; it does not restrict the implementation to a template.",
                ]
                stage_name = str(getattr(self, "stage_name", ""))
                expected_role = _expected_contract_role(stage_name)
                if expected_role:
                    base["Approved contract role"] = [
                        f"This stage has the exact contract_role `{expected_role}`; write that exact value in contract_execution.json.",
                        "Baseline stages must not implement or evaluate the proposed intervention; proposed-method stages must implement every approved intervention signal.",
                    ]
                if stage_name.startswith("3_"):
                    base["Approved intervention requirement"] = [POLICIES[3].prompt()]
                    from .comparison_policy import read_policy
                    locked_policy = read_policy(research_view.parent.parent)
                    if locked_policy:
                        base['Locked final training plan'] = (
                            f'Trusted baseline final policy: {json.dumps(locked_policy)}. '
                            'Declare a top-level literal FINAL_TRAINING_PLAN containing exactly this max_epochs '
                            'and early_stopping plus search_epochs: a list of per-candidate search caps. '
                            'Reserve the final max_epochs FIRST, then fit search within the remaining total. '
                            'For example, a 12-epoch final cap and [5,5,5] search cost 27 epochs. '
                            'Use this plan in actual training and manifest export. Never shorten the final cap '
                            'to accommodate search. Actual completed epochs may differ only through the fixed stopping policy.')
                if stage_name.startswith("2_"):
                    budget = DEFAULT_TUNING_BUDGET
                    base["Tuning evidence schema"] = self._prompt_hyperparam_tuning_resp_fmt["Tuning evidence schema"]
                    base["Tuning execution budget"] = [
                        f"Evaluate at most {budget.max_conditions} candidate values in this sandbox.",
                        f"Candidate count multiplied by epochs per candidate must not exceed {budget.max_total_epochs}.",
                        "Keep the baseline epoch count and every non-tuned training condition unchanged.",
                        "After each candidate finishes, atomically replace working/tuning_progress.json with completed candidates, metrics, and the next candidate index.",
                        "Flush progress and console output after every epoch; partial progress is diagnostic evidence, never a successful experiment manifest.",
                    ]
                if stage_name.startswith("4_"):
                    base["Approved ablation requirement"] = [
                        "Use the sole mounted dataset; multi-dataset and cross-dataset studies are forbidden.",
                        "Train exactly one variant that removes one approved intervention component; the host pairs it with the trusted full method.",
                        "Never retrain the accepted full method in the ablation sandbox.",
                        "Do not rebuild an unrelated method from the baseline parent.",
                        "Keep split, seeds, architecture, training budget, and evaluation metrics fixed.",
                    ]
                if require_dynamic_audit:
                    base["Mandatory dynamic-resolution repair constraints"] = [
                        "The previous soft router was invalid because it always executed both 28x28 and 56x56 branches.",
                        "At validation inference use a hard per-sample decision and execute the high-resolution feature extractor only on the selected tensor subset; never compute high-resolution features for the whole batch.",
                        "For auditable generated code, assign that indexed tensor subset to a variable named selected_high_inputs before high-resolution feature extraction.",
                        "Prevent route collapse: the validation high-resolution fraction must be strictly between 0.05 and 0.95.",
                        "Report numeric validation_accuracy, high_resolution_fraction, executed_high_resolution_samples, executed_low_resolution_samples, and both_branches_executed_samples in experiment_result.json metrics.",
                        "The two executed sample counts must sum to the validation size and both_branches_executed_samples must be zero.",
                        "Evaluate compact fixed-low and fixed-high controls with the same trained weights; do not add separate control training loops. Report measured validation inference elapsed seconds and do not present expected soft probabilities as actual FLOPs savings.",
                    ]
                return base

            def _analyze_plots_with_vlm(self, node):
                # Experiment nodes are executed by MinimalAgent instances inside
                # ProcessPoolExecutor workers, so this guard belongs on the
                # shared mixin (not only on the outer ParallelAgent subclass).
                # Formal Beta figures are generated later from structured
                # evidence and validated through FigureManifest.
                node.plot_analyses = []
                node.is_buggy_plots = False
                node.vlm_feedback_summary = (
                    "Optional VLM plot critique skipped; use formal figure validation."
                )
                return None

        class PathologyMinimalAgent(PathologyPromptMixin, MinimalAgent):
            def _generate_plotting_code(self, *args, **kwargs):
                return "# Formal evidence figures are generated by the host after candidate selection."

            def _generate_seed_node(self, parent_node):
                node = super()._generate_seed_node(parent_node)
                node.code = _preserve_host_injected_seed(node.code)
                return node

            def parse_exec_result(self, node, exec_result, workspace):
                # Sandbox and host-side contract validation already provide the
                # authoritative success decision. Do not make a valid 30-minute
                # training run depend on a second provider call.
                node.absorb_exec_result(exec_result)
                node.is_buggy = node.exc_type is not None
                node.analysis = "" if not node.is_buggy else "Sandbox or experiment-contract validation failed."

            def plan_and_code_query(self, prompt, retries=3):
                if "analyzing experimental results stored in numpy files" in str(prompt):
                    code = """import json
from pathlib import Path
result = json.loads(Path('working/experiment_result.json').read_text(encoding='utf-8'))
for name, value in result['metrics'].items():
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        print(f'{name}: {value}')
"""
                    return "Read the host-validated structured result directly.", code
                plan, code = super().plan_and_code_query(prompt, retries=retries)
                spec = parse_method_spec(plan) or infer_method_spec(
                    code, list(intervention_signals)
                )
                if extract_method_spec(code) is None:
                    code = attach_method_spec(code, spec)
                return plan, code

        class PathologyParallelAgent(PathologyPromptMixin, ParallelAgent):
            def _define_global_metrics(self):
                name = self.cfg.agent.contract_metric
                return json.dumps([{"name": name, "maximize": True,
                                    "description": "Primary validation metric fixed by the approved contract"}])

            def cleanup(self):
                # Stop submitters before scanning sandbox receipts, so a worker
                # cannot launch another container during cleanup.
                for process in list((self.executor._processes or {}).values()):
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
                    if process.is_alive():
                        raise SandboxCleanupError("Worker termination could not be confirmed")
                if hasattr(docker_runner, "cancel_active"):
                    docker_runner.cancel_active(Path(self.cfg.workspace_dir))
                return super().cleanup()

            def _generate_hyperparam_tuning_idea(self):
                idea = super()._generate_hyperparam_tuning_idea()
                if idea is not None and any(
                    term in idea.name.casefold() for term in ("epoch", "learning rate")
                ):
                    idea.name = "learning rate"
                    idea.description = (
                        "Tune learning rate with validation-only selection and convergence-aware early stopping, "
                        "while keeping architecture, data split, seed, and all other settings unchanged. "
                        "Evaluate exactly two learning-rate candidates including the baseline value, keep at most "
                        "15 epochs per candidate, atomically save tuning_progress.json after each candidate, and "
                        "finish within the 60-minute sandbox ceiling. " + TRAINING_POLICY_PROMPT
                    )
                return idea

            def _generate_ablation_idea(self):
                idea = super()._generate_ablation_idea()
                if idea is not None:
                    invalid = re.search(
                        r"multi[-_ ]dataset|cross[-_ ]dataset|generalization",
                        f"{idea.name} {idea.description}",
                        flags=re.IGNORECASE,
                    )
                    component = next(iter(intervention_signals), "approved intervention")
                    if invalid:
                        idea.name = f"Remove {component}"
                        idea.description = (
                            f"On the sole mounted dataset, compare the accepted full method against exactly one "
                            f"variant with the approved `{component}` component disabled."
                        )
                    idea.description += (
                        " Use only the sole mounted dataset and train exactly one condition with one component removed. "
                        "The host compares it with the trusted full-method artifact. Keep the split, seeds, architecture, optimizer, learning rate, epoch count, "
                        "training budget, and contract metrics fixed. Multi-dataset evaluation is forbidden."
                    )
                return idea

            def _run_multi_seed_evaluation(self, node):
                count = int(self.cfg.agent.multi_seed_eval.num_seeds)
                if count <= 1:
                    return []
                from ai_scientist.treesearch.journal import Node

                # Upstream mutates and reuses one node_data dictionary while
                # submitting all futures. Process-pool serialization can then
                # observe a later seed value. Submit one independent deep copy
                # at a time so seed identity and single-GPU scheduling are exact.
                parent_data = node.to_dict()
                seed_nodes = []
                parent_hash = hashlib.sha256(node.code.encode('utf-8')).hexdigest()
                root = research_view.parent.parent
                parent_manifest = load_manifest(root / 'experiment_logs/evidence' / parent_hash / 'experiment_manifest.json')
                for seed in self.cfg.agent.get('repeat_seeds', list(range(count))):
                    if parent_manifest['seed'] == seed:
                        verified_metrics(root / 'experiment_logs/evidence' / parent_hash,
                                         research_view / 'dataset_profile.json', parent_hash)
                        seed_nodes.append(node)
                        continue
                    reusable = []
                    for existing in self.journal.nodes:
                        if not existing.is_seed_node or existing.is_seed_agg_node or existing.is_buggy is not False:
                            continue
                        digest = hashlib.sha256(existing.code.encode('utf-8')).hexdigest()
                        directory = root / 'experiment_logs/evidence' / digest
                        try:
                            saved = load_manifest(directory / 'experiment_manifest.json')
                            if not saved.get('repeat_parent_code_sha256'):
                                raise AutonomousExperimentError(
                                    'Legacy successful seed lacks parent identity; explicit migration required before further training')
                            if saved.get('repeat_parent_code_sha256') == parent_hash and saved['seed'] == seed:
                                verified_metrics(directory, research_view / 'dataset_profile.json', digest)
                                reusable.append(existing)
                        except (OSError, ValueError, ManifestError, IntegrityError) as error:
                            raise AutonomousExperimentError(
                                'Successful seed evidence cannot be verified; refusing to retrain it automatically') from error
                    if len(reusable) > 1:
                        raise AutonomousExperimentError('Ambiguous completed repeat identity')
                    if reusable:
                        seed_nodes.append(reusable[0])
                        continue
                    seed_data = copy.deepcopy(parent_data)
                    repeat = {'parent_code_sha256': parent_hash, 'learning_rate': parent_manifest['learning_rate'],
                              'parameters': parent_manifest.get('selected_parameters', {})}
                    seed_data["code"] = (
                        '# PATH_AI_REPEAT: ' + json.dumps(repeat) + '\nPATH_AI_REPEAT = ' + repr(repeat) + '\n'
                        +
                        "# Set random seed\nimport random\nimport numpy as np\nimport torch\n\n"
                        f"seed = {seed}\nrandom.seed(seed)\nnp.random.seed(seed)\n"
                        "torch.manual_seed(seed)\nif torch.cuda.is_available():\n"
                        "    torch.cuda.manual_seed(seed)\n\n"
                        + node.code
                    )
                    future = self.executor.submit(
                        self._process_node_wrapper,
                        seed_data,
                        self.task_desc,
                        self.cfg,
                        0,
                        "",
                        self.evaluation_metrics,
                        self.stage_name,
                        None,
                        None,
                        None,
                        None,
                        None,
                        True,
                    )
                    result_data = future.result(timeout=self.timeout)
                    result_node = Node.from_dict(result_data, self.journal)
                    self.journal.append(result_node)
                    if hasattr(self, '_on_seed_completed'):
                        self._on_seed_completed()
                    if result_node.is_buggy is not False:
                        raise AutonomousExperimentError(f'Repeat seed {seed} failed; completed seeds were preserved')
                    seed_nodes.append(self.journal.get_node_by_id(result_node.id))
                return seed_nodes

            def _run_plot_aggregation(self, node, seed_nodes):
                # Upstream aggregation indexes exactly three seed nodes even when
                # cfg.agent.multi_seed_eval.num_seeds is explicitly one. A fast
                # single-seed run remains reproducible; skip only the incompatible
                # aggregate plot while preserving the seed node and its metrics.
                if len(seed_nodes) < 3:
                    return node
                return super()._run_plot_aggregation(node, seed_nodes)

        class PathologyAgentManager(AgentManager):
            def _create_agent_for_stage(self, stage):
                _write_agent_progress(
                    research_view.parent.parent, stage, self.journals[stage.name], self.stage_history
                )
                return super()._create_agent_for_stage(stage)

            def _get_task_desc_str(self):
                description = super()._get_task_desc_str()
                for name in ("Research View Interface", "Approved Research Execution Contract", "Requirement classification", "Image preprocessing"):
                    if self.task_desc.get(name) is not None:
                        description += "\n" + name + ":\n" + json.dumps(
                            self.task_desc[name], ensure_ascii=False
                        ) + "\n"
                return description

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.main_stage_goals = {number: policy.prompt() for number, policy in POLICIES.items()}
                self.current_stage.goals = self.main_stage_goals[1]

            def _check_substage_completion(self, current_substage, journal):
                if current_substage.name.startswith(("1_", "2_", "3_", "4_")):
                    return self._check_stage_completion(current_substage)
                return super()._check_substage_completion(current_substage, journal)

            def _check_stage_completion(self, stage):
                journal = self.journals[stage.name]
                if stage.name.startswith('3_') and journal.nodes:
                    root = research_view.parent.parent
                    if (root / 'research/research_contract.json').exists():
                        from .comparison_policy import bind_policy
                        bind_policy(root, journal.nodes[0].code)
                for existing in journal.nodes:
                    if existing.is_buggy is True and 'needs review for unknown components:' in str(getattr(existing, 'term_out', '')):
                        raise AutonomousExperimentError('SEMANTIC_REVIEW_REQUIRED: classify unsupported components before further generation; do not rename metadata to bypass review. ' + str(existing.term_out)[-800:])
                    if existing.is_buggy is True and 'ARTIFACT_REVIEW_REQUIRED:' in str(getattr(existing, 'term_out', '')):
                        raise AutonomousExperimentError(
                            'ARTIFACT_REVIEW_REQUIRED: completed training outputs were preserved in '
                            'experiment_logs/raw_executions; inspect and revalidate artifacts before authorizing '
                            'another training execution. ' + str(existing.term_out)[-800:])
                _check_repeated_preflight_failures(journal)
                if stage.name.startswith('2_'):
                    node, errors = select_verified_tuning(journal, research_view.parent.parent, self.cfg.agent.contract_metric)
                    if node is not None:
                        return True, 'Host verified tuning controls, histories and validation predictions'
                    if any('Missing immutable' in error or 'legacy evidence' in error for error in errors) or len(journal.nodes) >= stage.max_iterations:
                        raise AutonomousExperimentError('TUNING_EVIDENCE_BLOCKED: ' + '; '.join(errors or ['No verified tuning candidate']))
                    for candidate in journal.nodes[1:]:
                        reasons = [error for error in errors if error.startswith(candidate.id + ':')]
                        if reasons:
                            candidate.is_buggy = True
                            candidate.analysis = '\n'.join(reasons)
                            candidate._term_out = [candidate.analysis]
                    return False, 'No complete verified tuning candidate yet'
                for node in journal.nodes:
                    if node.is_buggy is False and node.is_buggy_plots is None:
                        node.is_buggy_plots = False
                if stage.stage_number == 1 and journal.good_nodes:
                    return True, "Found working implementation"
                if stage.name.startswith(("3_", "4_")):
                    if has_valid_generated_node(journal):
                        return True, "Found contract-valid generated implementation"
                    if len(journal.nodes) >= stage.max_iterations:
                        raise AutonomousExperimentError(
                            f"Refusing to complete {stage.name} without a valid generated child"
                        )
                    return False, "No contract-valid generated implementation yet"
                if require_dynamic_audit and stage.stage_number >= 5 and not any(
                    node.is_buggy is False
                    and node.parent is not None
                    and _has_indexed_high_subset(node.code)
                    for node in journal.nodes
                ):
                    return False, "No audited non-collapsed conditional hard-routing implementation yet"
                return super()._check_stage_completion(stage)

            def _get_best_implementation(self, stage_name):
                if stage_name.startswith('2_'):
                    node, errors = select_verified_tuning(self.journals[stage_name], research_view.parent.parent, self.cfg.agent.contract_metric)
                    if node is None:
                        raise AutonomousExperimentError('TUNING_EVIDENCE_BLOCKED: ' + '; '.join(errors))
                    copied = copy.deepcopy(node)
                    copied.parent = None
                    copied.children = set()
                    return copied
                stage = next((item for item in self.stages if item.name == stage_name), None)
                if stage_name.startswith(("3_", "4_")):
                    from ai_scientist.treesearch.journal import Journal

                    proposed = Journal()
                    for node in self.journals[stage_name].nodes:
                        if (
                            node.is_buggy is False
                            and node.parent is not None
                            and not node.is_seed_node
                            and not node.is_seed_agg_node
                        ):
                            proposed.append(node)
                    best = proposed.get_best_node(cfg=self.cfg)
                    if best is None:
                        return None
                    copied = copy.deepcopy(best)
                    copied.parent = None
                    copied.children = set()
                    return copied
                if not require_dynamic_audit or stage is None or stage.stage_number < 5:
                    return super()._get_best_implementation(stage_name)
                from ai_scientist.treesearch.journal import Journal

                audited = Journal()
                for node in self.journals[stage_name].nodes:
                    if (
                        node.is_buggy is False
                        and node.parent is not None
                        and _has_indexed_high_subset(node.code)
                    ):
                        audited.append(node)
                best = audited.get_best_node(cfg=self.cfg)
                if best is None:
                    return None
                copied = copy.deepcopy(best)
                copied.parent = None
                copied.children = set()
                return copied

            def _create_next_main_stage(self, current_substage, journal):
                if current_substage.name.startswith('2_'):
                    complete, reason = self._check_stage_completion(current_substage)
                    if not complete:
                        raise AutonomousExperimentError('TUNING_EVIDENCE_BLOCKED: ' + reason)
                if current_substage.name.startswith("3_") and not has_valid_generated_node(
                    journal
                ):
                    raise AutonomousExperimentError(
                        "Refusing to advance to ablations: proposed-method stage has no valid generated child"
                    )
                if "hard_routing_repair" not in current_substage.name:
                    following = super()._create_next_main_stage(current_substage, journal)
                    enabled = self.cfg.agent.get('enabled_stages', [1, 2, 3, 4])
                    while following is not None and int(following.name.split('_', 1)[0]) not in enabled:
                        following = super()._create_next_main_stage(following, journal)
                    return following
                if not any(
                    node.is_buggy is False
                    and node.parent is not None
                    and _has_indexed_high_subset(node.code)
                    for node in journal.nodes
                ):
                    raise AutonomousExperimentError(
                        "Refusing to advance to ablations: hard-routing repair has no audited child"
                    )
                from ai_scientist.treesearch.agent_manager import Stage

                return Stage(
                    name="4_ablation_studies_2_hard_routing_ablation",
                    description="hard_routing_ablation",
                    goals=(
                        self.main_stage_goals[4]
                        + " Preserve genuine conditional execution, non-collapsed routing, and all mandatory dynamic audit metrics."
                    ),
                    max_iterations=self._get_max_iterations(4),
                    num_drafts=0,
                    stage_number=current_substage.stage_number + 1,
                )

            def _save_checkpoint(self):
                if self.current_stage is None:
                    return
                save_dir = (
                    Path(self.workspace_dir).parent
                    / "logs"
                    / Path(self.workspace_dir).name
                    / f"stage_{self.current_stage.name}"
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                return super()._save_checkpoint()

        class DockerInterpreter:
            def __init__(self, working_dir, timeout=3600, format_tb_ipython=False, agent_file_name="runfile.py", env_vars=None):
                self.working_dir = Path(working_dir).resolve()
                self.working_dir.mkdir(parents=True, exist_ok=True)
                self.agent_file_name = agent_file_name
                self.stage_name = os.environ.get("PATH_AI_SCIENTIST_AGENT_STAGE", "")

            def trusted_metrics_for_node(self, code):
                return metric_rows(verified_metrics(
                    self.working_dir / "working", research_view / "dataset_profile.json",
                    hashlib.sha256(code.encode("utf-8")).hexdigest(),
                ))

            def run_for_purpose(self, code, *, purpose):
                if purpose == "training":
                    return self.run(code)
                if purpose == "plotting":
                    # Do not execute auxiliary generated code with training privileges.
                    return ExecutionResult(["Optional tree plots deferred to host figure validation.\n"], 0.0, None, None, None)
                raise ValueError(f"Unsupported execution purpose: {purpose}")

            def run(self, code: str, reset_session=True):
                result_path = self.working_dir / "working" / "experiment_result.json"
                manifest_path = self.working_dir / "working" / "experiment_manifest.json"
                previous_mtime = result_path.stat().st_mtime_ns if result_path.exists() else None
                try:
                    validate_generated_code(code)
                    validate_no_synthetic_dataset(code)
                    _validate_stage_semantics(
                        code, self.stage_name, list(intervention_signals)
                    )
                    from .comparison_policy import read_policy, validate_final_plan
                    comparison_policy = read_policy(research_view.parent.parent) if self.stage_name.startswith('3_') else None
                    if comparison_policy:
                        validate_final_plan(code, comparison_policy, POLICIES[3].budget)
                    # A new run must not inherit an old worker's tuning receipt.
                    for name in ("tuning_evidence.json", "tuning_progress.json"):
                        (result_path.parent / name).unlink(missing_ok=True)
                    runtime_options = {}
                    if _expected_contract_role(self.stage_name) == 'proposed_method':
                        from .method_spec import semantic_report
                        report = semantic_report(code, list(intervention_signals))
                        if ('label_smoothing' in report['required']
                                and report['runtime_checks']['standard_smoothing_required']):
                            runtime_options['require_standard_smoothing'] = True
                    result = docker_runner.run_python(
                        code,
                        self.working_dir,
                        self.agent_file_name,
                        dataset_mount=research_view,
                        **runtime_options,
                    )
                except SandboxCleanupError:
                    raise
                except (CodePolicyError, RunnerError, IntegrityError, OSError) as error:
                    # Invalid or policy-rejected LLM code is an experiment-node
                    # failure, not an orchestration failure. Returning it through
                    # the upstream interpreter contract lets the agent debug the
                    # saved attempt on its next step.
                    return ExecutionResult(
                        [f"Generated experiment rejected before execution: {error}\n"],
                        0.0,
                        type(error).__name__,
                        {"exit_code": None, "pre_execution_rejection": True},
                        None,
                    )
                exc_type = None if result.succeeded else ("TimeoutError" if result.timed_out else "RuntimeError")
                if result.exit_code == 86 and "IMPORT_PREFLIGHT_FAILED" in result.stderr:
                    exc_type = "ImportPreflightError"
                output = (result.stdout + ("\n" if result.stdout and result.stderr else "") + result.stderr).splitlines(keepends=True)
                if result.succeeded and result_path.is_file() and result_path.stat().st_mtime_ns != previous_mtime:
                    from .autonomous_evidence import preserve_unvalidated_execution
                    preserve_unvalidated_execution(result_path.parent,
                        research_view.parent.parent / 'experiment_logs/raw_executions', code)
                if result.succeeded and (not result_path.exists() or previous_mtime == result_path.stat().st_mtime_ns):
                    return ExecutionResult(output + ['Missing fresh experiment_result.json\n'], result.elapsed_seconds,
                                           'ExperimentContractError', None, None)
                if result.succeeded and result_path.exists():
                    current_mtime = result_path.stat().st_mtime_ns
                    if previous_mtime != current_mtime:
                        try:
                            raw = json.loads(result_path.read_text(encoding="utf-8"))
                            expected_role = _expected_contract_role(self.stage_name)
                            if expected_role:
                                execution_path = (
                                    self.working_dir / "working" / "contract_execution.json"
                                )
                                execution = json.loads(
                                    execution_path.read_text(encoding="utf-8")
                                )
                                allowed_roles = _allowed_contract_roles(self.stage_name)
                                if execution.get("contract_role") not in allowed_roles:
                                    raise ValueError(
                                        f"contract_role must be one of {sorted(allowed_roles)!r} "
                                        f"for stage {self.stage_name}"
                                    )
                            metrics = raw.get("metrics")
                            if not isinstance(metrics, dict) or not metrics:
                                raise ValueError("metrics must be a non-empty object")
                            _normalize_tuning_manifest(manifest_path)
                            manifest = load_manifest(manifest_path, require_training_policy=True)
                            if comparison_policy:
                                from .comparison_policy import validate_final_manifest
                                validate_final_manifest(manifest, comparison_policy)
                            from .experiment_manifest import metric_policy
                            from .research_contract import load_contract
                            contract_path = research_view.parent.parent / 'research/research_contract.json'
                            primary = (load_contract(research_view.parent.parent, require_approved=True)['metrics']['primary']['name']
                                       if contract_path.is_file() else manifest.get('primary_metric', manifest['selection_metric']))
                            metric_policy(manifest, primary)
                            from .experiment_manifest import validate_execution_budget
                            policy = stage_policy(self.stage_name)
                            if policy:
                                validate_execution_budget(manifest, policy.budget, repeat='# PATH_AI_REPEAT:' in code)
                            if raw.get("seed") is not None and int(raw["seed"]) != int(manifest["seed"]):
                                raise ValueError("result and manifest seeds disagree")
                            raw["seed"] = int(manifest["seed"])
                            injected_seed = re.search(r"# Set random seed[\s\S]{0,240}?\bseed\s*=\s*(\d+)", code)
                            if injected_seed:
                                expected_seed = int(injected_seed.group(1))
                                reported_seed = raw.get("seed")
                                if reported_seed is not None and int(reported_seed) != expected_seed:
                                    raise ValueError("result misreported the host-injected seed")
                                if int(manifest["seed"]) != expected_seed:
                                    raise ValueError("manifest misreported the host-injected seed")
                                if expected_role and int(execution.get("training_seed", -1)) != expected_seed:
                                    raise ValueError("contract execution misreported the host-injected seed")
                                raw["seed"] = expected_seed
                            predictions = raw.get("predictions")
                            targets = raw.get("targets")
                            sample_ids = raw.get("sample_ids")
                            probabilities = raw.get("probabilities")
                            checkpoint_path = self.working_dir / "working" / "model_checkpoint.pt"
                            required_inference_tokens = ("HAS_TRAIN_SPLIT", "model_checkpoint.pt", "torch.load")
                            if not all(token in code for token in required_inference_tokens):
                                raise ValueError("generated code lacks the required checkpoint-backed inference-only branch")
                            if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
                                raise ValueError("generated code did not write working/model_checkpoint.pt")
                            if not all(isinstance(value, list) for value in (predictions, targets, sample_ids)):
                                raise ValueError("predictions, targets, and sample_ids must be JSON arrays")
                            if require_dynamic_audit:
                                if not _has_indexed_high_subset(code):
                                    raise ValueError("generated code does not expose indexed conditional high-resolution execution")
                                required = {
                                    "validation_accuracy",
                                    "high_resolution_fraction",
                                    "executed_high_resolution_samples",
                                    "executed_low_resolution_samples",
                                    "both_branches_executed_samples",
                                    "validation_inference_seconds",
                                }
                                missing = sorted(required - metrics.keys())
                                if missing:
                                    raise ValueError(f"dynamic audit metrics missing: {missing}")
                                high_fraction = float(metrics["high_resolution_fraction"])
                                high_count = int(metrics["executed_high_resolution_samples"])
                                low_count = int(metrics["executed_low_resolution_samples"])
                                both_count = int(metrics["both_branches_executed_samples"])
                                expected_count = int(validation_count or 0)
                                if not 0.05 < high_fraction < 0.95:
                                    raise ValueError(f"routing collapsed: high_resolution_fraction={high_fraction}")
                                if high_count + low_count != expected_count:
                                    raise ValueError(
                                        f"executed route counts {high_count}+{low_count} do not equal validation size {expected_count}"
                                    )
                                if both_count != 0:
                                    raise ValueError("high and low branches were both executed for some validation samples")
                            code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
                            raw = {
                                "schema_version": 1,
                                "status": "completed",
                                "method_name": str(raw.get("method_name") or "agent_generated_method"),
                                "parent_experiment_id": raw.get("parent_experiment_id"),
                                "code_sha256": code_hash,
                                "seed": int(manifest["seed"]),
                                "split": "validation",
                                "metrics": metrics,
                                "resource_usage": {**raw.get("resource_usage", {}), "elapsed_seconds": result.elapsed_seconds},
                                "artifacts": {**raw.get("artifacts", {}), "experiment_data": "working/experiment_data.npy"},
                                "test_data_accessed": bool(raw.get("test_data_accessed", False)),
                                "predictions": predictions,
                                "targets": targets,
                                "sample_ids": [str(value) for value in sample_ids],
                                "probabilities": probabilities,
                                "class_names": raw.get("class_names"),
                            }
                            if raw["test_data_accessed"]:
                                raise ValueError("research node reported test access")
                            record_trusted_evaluation(
                                profile_path=research_view / "dataset_profile.json",
                                split="validation", sample_ids=raw["sample_ids"], targets=targets,
                                predictions=predictions, probabilities=probabilities,
                                code_sha256=code_hash, output_dir=result_path.parent,
                                reported_metrics=metrics,
                            )
                            from .metrics import add_contract_metric
                            trusted = json.loads((result_path.parent / 'trusted_metrics.json').read_text(encoding='utf-8'))['metrics']
                            resolved = add_contract_metric(research_view.parent.parent, trusted,
                                baseline_hash=code_hash if self.stage_name.startswith('1_') else None)
                            raw['metrics'] = {**metrics, **resolved}
                            repeat_match = re.search(r'^# PATH_AI_REPEAT: (.+)$', code, re.MULTILINE)
                            if repeat_match:
                                repeat = json.loads(repeat_match.group(1))
                                if manifest['learning_rate'] != repeat['learning_rate']:
                                    raise IntegrityError('Repeat changed the selected learning rate')
                                if manifest.get('selected_parameters', {}) != repeat['parameters']:
                                    raise IntegrityError('Repeat changed the selected method parameters')
                                manifest['repeat_parent_code_sha256'] = repeat['parent_code_sha256']
                                manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
                            if self.stage_name.startswith('2_') and not repeat_match:
                                record = json.loads((result_path.parent / 'tuning_evidence.json').read_text(encoding='utf-8'))
                                value = resolved.get(primary)
                                if value is None:
                                    raise IntegrityError(f'Trusted primary metric is missing: {primary}')
                                validate_tuning_record(record, manifest, value, primary=primary)
                            result_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                            snapshots = self.working_dir.parent.parent / "experiment_logs" / "generated_code"
                            snapshots.mkdir(parents=True, exist_ok=True)
                            snapshot = snapshots / f"{code_hash}.py"
                            if not snapshot.exists():
                                snapshot.write_text(code, encoding="utf-8")
                            manifest_snapshot = snapshots / f"{code_hash}.manifest.json"
                            manifest_snapshot.write_bytes(manifest_path.read_bytes())
                            snapshot_evidence(
                                result_path.parent, snapshots.parent / "evidence" / code_hash
                            )
                        except (
                            OSError,
                            TypeError,
                            KeyError,
                            ValueError,
                            json.JSONDecodeError,
                            ManifestError,
                            IntegrityError,
                        ) as error:
                            exc_type = "ExperimentContractError"
                            output.append(f"Invalid experiment contract: {error}\n")
                            if isinstance(error, (ManifestError, KeyError)):
                                output.append('ARTIFACT_REVIEW_REQUIRED: training finished but metadata validation failed; do not retrain to repair metadata.\n')
                return ExecutionResult(output, result.elapsed_seconds, exc_type, {"exit_code": result.exit_code} if exc_type else None, None)

            def cleanup_session(self):
                return None

        # ProcessPoolExecutor serializes worker instances even on Linux. Publish
        # runtime subclasses as stable module globals while keeping vendor code untouched.
        for name, runtime_class in {
            "PathologyMinimalAgent": PathologyMinimalAgent,
            "PathologyParallelAgent": PathologyParallelAgent,
            "PathologyAgentManager": PathologyAgentManager,
            "DockerInterpreter": DockerInterpreter,
        }.items():
            runtime_class.__name__ = name
            runtime_class.__qualname__ = name
            runtime_class.__module__ = __name__
            globals()[name] = runtime_class
        return PathologyAgentManager, PathologyParallelAgent, PathologyMinimalAgent, DockerInterpreter

    @staticmethod
    def _checkpoint(path: Path, manager: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(manager.__dict__, handle)
        temporary.replace(path)

    @staticmethod
    def _load_or_create_manager(path: Path, manager_class: type, task_desc: str, cfg: Any, workspace: AutonomousTaskWorkspace):
        if path.is_file():
            with path.open("rb") as handle:
                state = pickle.load(handle)
            manager = manager_class.__new__(manager_class)
            manager.__dict__.update(state)
            if Path(manager.workspace_dir).resolve() != workspace.experiment_workspace.resolve():
                raise AutonomousExperimentError("Checkpoint belongs to another workspace")
            # Dataset discovery may refine interface facts between interrupted
            # attempts. Refresh only the immutable research contract and runtime
            # config; journals and generated nodes remain untouched.
            manager.task_desc = json.loads(task_desc)
            manager.cfg = cfg
            # Pickled instances bypass __init__; refresh policy on resume as well
            # as new runs so a stale goal cannot contradict the current validator.
            manager.main_stage_goals = {number: policy.prompt() for number, policy in POLICIES.items()}
            for stage in manager.stages:
                policy = stage_policy(stage.name)
                if policy:
                    stage.goals = policy.prompt()
            for journal in manager.journals.values():
                seen = set()
                unique = []
                for node in journal.nodes:
                    if node.id not in seen:
                        seen.add(node.id)
                        unique.append(node)
                journal.nodes[:] = unique
            from ai_scientist.treesearch.utils.metric import MetricValue

            profile_path = workspace.dataset / "research_view/dataset_profile.json"
            recovery = []
            for journal in manager.journals.values():
                for node in journal.nodes:
                    if node.is_buggy is not True or node.exc_type is not None:
                        continue
                    if getattr(node, "parse_exc_type", None) != "IntegrityError":
                        continue
                    code_hash = hashlib.sha256(node.code.encode("utf-8")).hexdigest()
                    sources = [workspace.experiment_logs / "evidence" / code_hash]
                    sources.extend(workspace.experiment_workspace.glob("*/working"))
                    for source in sources:
                        # Legacy mutable workspaces have no execution-time weight
                        # receipt. Do not manufacture checkpoint provenance now.
                        if not (source / "artifact_hashes.json").is_file():
                            continue
                        try:
                            metrics = verified_metrics(source, profile_path, code_hash)
                        except (OSError, ValueError, KeyError, IntegrityError, ManifestError):
                            continue
                        saved = workspace.experiment_logs / "evidence" / code_hash
                        snapshot_evidence(source, saved)
                        node.metric = MetricValue(value={"metric_names": metric_rows(metrics)})
                        node.is_buggy = False
                        node.is_buggy_plots = False
                        node.exp_results_dir = str(saved)
                        recovery.append({"node_id": node.id, "code_sha256": code_hash,
                                         "reason": "Recovered verified training after auxiliary-parser rejection"})
                        break
            if recovery:
                (workspace.experiment_logs / "parser_recovery.json").write_text(
                    json.dumps(recovery, indent=2), encoding="utf-8"
                )
            for journal in manager.journals.values():
                for node in journal.nodes:
                    if node.is_buggy is False and node.is_buggy_plots is None:
                        node.is_buggy_plots = False
            return manager
        return manager_class(task_desc=task_desc, cfg=cfg, workspace_dir=workspace.experiment_workspace)


@contextmanager
def _patched_upstream(
    query_adapter: GatewayQueryAdapter,
    agent_class: type,
    minimal_agent_class: type,
    interpreter_class: type,
    *,
    sandbox_gpu_count: int = 0,
) -> Iterator[None]:
    targets = (
        "ai_scientist.treesearch.agent_manager.query",
        "ai_scientist.treesearch.parallel_agent.query",
        "ai_scientist.treesearch.journal.query",
        "ai_scientist.treesearch.journal2report.query",
    )
    with ExitStack() as stack:
        for target in targets:
            stack.enter_context(patch(target, query_adapter))
        stack.enter_context(patch("ai_scientist.treesearch.agent_manager.ParallelAgent", agent_class))
        stack.enter_context(patch("ai_scientist.treesearch.parallel_agent.MinimalAgent", minimal_agent_class))
        stack.enter_context(patch("ai_scientist.treesearch.interpreter.Interpreter", interpreter_class))
        if sandbox_gpu_count > 0:
            stack.enter_context(
                patch(
                    "ai_scientist.treesearch.parallel_agent.get_gpu_count",
                    return_value=sandbox_gpu_count,
                )
            )
        yield
