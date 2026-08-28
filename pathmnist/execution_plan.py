"""Versioned, bounded experiment DAGs and crash-safe host execution.

Callbacks must return host-verified evidence; this module never trusts generated
completion flags or chooses parameters from test-set measurements.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Callable


class ExecutionPlanError(RuntimeError):
    pass


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ExecutionPlanError("Execution plan schema must be 1")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 64:
        raise ExecutionPlanError("Execution plan needs 1..64 bounded steps")
    seen = {}
    total_epochs = 0
    for step in steps:
        name = step.get("id") if isinstance(step, dict) else None
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name)
            or name in seen
        ):
            raise ExecutionPlanError("Step IDs must be unique safe identifiers")
        dependencies = step.get("depends_on", [])
        if (
            not isinstance(dependencies, list)
            or len(set(dependencies)) != len(dependencies)
            or any(d not in seen for d in dependencies)
        ):
            raise ExecutionPlanError(f"{name}: dependencies must precede the step (no cycles)")
        kind = step.get("kind")
        if kind == "train":
            for key, ceiling in [("epochs", 15), ("timeout_seconds", 3600), ("max_attempts", 3)]:
                if type(step.get(key)) is not int or not 1 <= step[key] <= ceiling:
                    raise ExecutionPlanError(f"{name}: invalid {key}")
            fraction = step.get("train_fraction")
            if type(fraction) not in (int, float) or not 0 < fraction <= 1:
                raise ExecutionPlanError(f"{name}: invalid training fraction")
            if type(step.get("seed")) is not int or not isinstance(step.get("parameters"), dict):
                raise ExecutionPlanError(f"{name}: seed and explicit parameters required")
            if step.get("role") not in {"baseline", "proposed_method", "ablation"}:
                raise ExecutionPlanError(f"{name}: unknown role")
            source = step.get("parameters_from")
            if source is not None and (
                source not in dependencies or seen[source]["kind"] != "select"
            ):
                raise ExecutionPlanError(f"{name}: parameters_from must depend on a selection step")
            if source is not None:
                candidates = [seen[d] for d in seen[source]["depends_on"]]
                if any(c["role"] != step["role"] for c in candidates):
                    raise ExecutionPlanError(f"{name}: selection belongs to another role")
                if any(set(c["parameters"]) & set(step["parameters"]) for c in candidates):
                    raise ExecutionPlanError(f"{name}: selected parameters cannot be overridden")
            total_epochs += step["epochs"] * step["max_attempts"]
        elif kind == "select":
            if len(dependencies) < 2 or any(seen[d]["kind"] != "train" for d in dependencies):
                raise ExecutionPlanError(
                    f"{name}: select requires at least two training candidates"
                )
            if (
                step.get("metric") not in {"accuracy", "macro_f1", "weighted_f1"}
                or step.get("split") != "validation"
            ):
                raise ExecutionPlanError(
                    f"{name}: only supported validation metrics can select candidates"
                )
            candidates = [seen[d] for d in dependencies]
            for field in ("seed", "role", "train_fraction", "epochs"):
                if any(c[field] != candidates[0][field] for c in candidates):
                    raise ExecutionPlanError(f"{name}: candidate controls differ: {field}")
        else:
            raise ExecutionPlanError(f"{name}: unsupported step kind")
        seen[name] = step
    maximum = plan.get("max_total_epochs")
    if type(maximum) is not int or not 1 <= maximum <= 960 or total_epochs > maximum:
        raise ExecutionPlanError("Worst-case epoch budget exceeds approved plan ceiling")
    fingerprint(plan)  # Reject non-JSON / non-finite parameters.
    return plan


def requires_multistep(contract):
    text = "\n".join(
        [
            str(contract.get("research_question", "")),
            *(
                str(item.get("description", ""))
                for item in contract.get("interventions", [])
                if isinstance(item, dict)
            ),
        ]
    ).casefold()
    return bool(
        re.search(
            r"子集|subset|分阶段|多阶段|multi.stage|\d+\s*%[^。；\n]{0,20}(?:选参|确定参数|tuning)",
            text,
        )
    )


def require_compatible_execution(contract, *, supports_dag=False):
    plan = contract.get("execution_plan")
    if plan is None:
        if requires_multistep(contract):
            raise ExecutionPlanError(
                "EXECUTION_PLAN_REQUIRED: multi-step research requires an explicit approved execution plan before paid execution"
            )
        return None
    validate_plan(plan)
    if not supports_dag:
        raise ExecutionPlanError(
            "EXECUTION_BACKEND_UNSUPPORTED: this entry point cannot execute approved DAG steps; refusing legacy single-stage fallback"
        )
    return plan


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class StepExecutor:
    """Run under the task's exclusive lock; callbacks never run on the select step.

    execute(step, selected_parameters, output_dir) -> evidence
    verify(step, evidence, output_dir) -> host-recomputed validation metrics
    The verifier must bind dataset, code, checkpoint, training controls and seed.
    """

    def __init__(self, root: Path, plan: dict, contract_hash: str, dataset_hash: str):
        self.plan = validate_plan(plan)
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "execution_state.json"
        self.identity = fingerprint(
            {"plan": plan, "contract": contract_hash, "dataset": dataset_hash}
        )
        self.state = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists()
            else {"identity": self.identity, "steps": {}}
        )
        if self.state.get("identity") != self.identity:
            raise ExecutionPlanError(
                "Execution identity changed; do not reuse another contract/dataset/plan checkpoint"
            )

    def run(self, execute: Callable, verify: Callable, *, should_stop=lambda: False):
        metrics = {}
        by_id = {s["id"]: s for s in self.plan["steps"]}
        for step in self.plan["steps"]:
            if should_stop():
                return {"status": "paused", "steps": self.state["steps"]}
            name = step["id"]
            record = self.state["steps"].get(name, {"attempts": 0})
            dependencies = {
                d: fingerprint(self.state["steps"][d]) for d in step.get("depends_on", [])
            }
            parameters = dict(step.get("parameters", {}))
            source = step.get("parameters_from")
            if source:
                parameters.update(self.state["steps"][source]["evidence"]["parameters"])
            effective_step = {**step, "parameters": parameters}
            if record.get("status") == "completed":
                if record.get("dependencies") != dependencies:
                    raise ExecutionPlanError(f"{name}: dependency evidence changed")
                if step["kind"] == "train":
                    directory = (self.root / record["directory"]).resolve()
                    if not directory.is_relative_to(self.root.resolve()):
                        raise ExecutionPlanError(f"{name}: evidence path escapes execution root")
                    metrics[name] = verify(effective_step, record["evidence"], directory)
                else:
                    expected = self._selection(step, metrics, by_id)
                    if record["evidence"] != expected:
                        raise ExecutionPlanError(
                            f"{name}: saved selection differs from verified validation results"
                        )
                continue
            if step["kind"] == "select":
                evidence = self._selection(step, metrics, by_id)
                self.state["steps"][name] = {
                    "status": "completed",
                    "attempts": 0,
                    "dependencies": dependencies,
                    "evidence": evidence,
                }
                _write(self.path, self.state)
                continue
            # A crashed attempt counts against the budget, and is never presumed complete.
            if record["attempts"] >= step["max_attempts"]:
                raise ExecutionPlanError(
                    f"{name}: retry budget exhausted; explicit recovery required"
                )
            attempt = record["attempts"] + 1
            output = self.root / name / str(attempt)
            output.mkdir(parents=True, exist_ok=False)
            record = {
                "status": "running",
                "attempts": attempt,
                "dependencies": dependencies,
                "directory": str(output.relative_to(self.root)),
            }
            self.state["steps"][name] = record
            _write(self.path, self.state)
            try:
                evidence = execute(effective_step, parameters, output)
                metrics[name] = verify(effective_step, evidence, output)
                record.update(status="completed", evidence=evidence)
            except BaseException as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
                _write(self.path, self.state)
                raise
            _write(self.path, self.state)
        return {"status": "completed", "steps": self.state["steps"]}

    @staticmethod
    def _selection(step, metrics, by_id):
        values = [metrics[d][step["metric"]] for d in step["depends_on"]]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ExecutionPlanError("Selection requires finite host-verified metrics")
        # Deterministic first-candidate tie handling; no improvement is required.
        selected = max(zip(step["depends_on"], values), key=lambda x: x[1])[0]
        return {
            "selected_step": selected,
            "parameters": by_id[selected]["parameters"],
            "metric": step["metric"],
            "value": metrics[selected][step["metric"]],
        }
