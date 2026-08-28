from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema

from .budget import BudgetLedger
from .config import AppConfig
from .models import ModelInfo, ModelRegistry, ModelSelectionError
from .paper import PAPER_SCHEMA, REVIEW_SCHEMA, render_latex, validate_pdf
from .policy import validate_generated_code
from .provider import (
    ChatProvider,
    FixtureProvider,
    OpenRouterProvider,
    ZhipuProvider,
    fetch_model_catalog,
    load_openrouter_key,
    load_provider_key,
)
from .runner import DockerRunner, ExecutionResult, RunnerError
from .upstream_bridge import UpstreamMinimalBFTS


IDEA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        key: {"type": "string", "minLength": 1}
        for key in [
            "Name",
            "Title",
            "Short Hypothesis",
            "Related Work",
            "Abstract",
            "Experiments",
            "Risk Factors and Limitations",
        ]
    },
    "required": [
        "Name",
        "Title",
        "Short Hypothesis",
        "Related Work",
        "Abstract",
        "Experiments",
        "Risk Factors and Limitations",
    ],
    "additionalProperties": False,
}


class PipelineError(RuntimeError):
    """Raised when a Gate A acceptance invariant is not met."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _execution_dict(result: ExecutionResult) -> dict[str, Any]:
    return {
        "command": list(result.command),
        "exit_code": result.exit_code,
        "stdout": result.stdout[-20000:],
        "stderr": result.stderr[-20000:],
        "elapsed_seconds": result.elapsed_seconds,
        "timed_out": result.timed_out,
        "succeeded": result.succeeded,
    }


def _event(path: Path, stage: str, status: str, **details: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"timestamp": _utc_now(), "stage": stage, "status": status, **details},
                sort_keys=True,
            )
            + "\n"
        )


def fixture_models(config: AppConfig) -> dict[str, ModelInfo]:
    selected: dict[str, ModelInfo] = {}
    for role_name, role in config.roles.items():
        selected[role_name] = ModelInfo.from_api(
            {
                "id": role.candidates[0],
                "context_length": 1_000_000,
                "pricing": {
                    "prompt": "0.0000001",
                    "completion": "0.0000002",
                    "request": "0",
                },
                "supported_parameters": [
                    "tools",
                    "tool_choice",
                    "max_tokens",
                    "temperature",
                ],
            }
        )
    if selected["paper_writer"].model_id == selected["reviewer"].model_id:
        raise ModelSelectionError(
            "Offline fixture config must keep writer and reviewer independent"
        )
    return selected


def select_live_models(config: AppConfig) -> dict[str, ModelInfo]:
    if config.provider.name != "openrouter":
        load_provider_key(config)
        registry = ModelRegistry.from_static_catalog(
            config.models, config.budget.cny_per_usd
        )
    else:
        key = load_openrouter_key()
        registry = ModelRegistry.from_api(fetch_model_catalog(config, key))
    selected = registry.select_all(config)
    maximum = sum(
        selected[role].maximum_cost(role_config, config.budget.reserve_margin)
        for role, role_config in config.roles.items()
    )
    if maximum > config.budget.hard_limit_usd:
        raise ModelSelectionError(
            f"Configured one-pass maximum ${maximum:.4f} exceeds Gate A limit "
            f"${config.budget.hard_limit_usd:.2f}"
        )
    return selected


def _validate_metrics(workspace: Path) -> dict[str, Any]:
    metrics_path = workspace / "working" / "metrics.json"
    data_path = workspace / "working" / "experiment_data.npy"
    if (
        not metrics_path.exists()
        or not data_path.exists()
        or data_path.stat().st_size == 0
    ):
        raise PipelineError(
            "Experiment did not produce both metrics.json and experiment_data.npy"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required = {"primary_metric", "value", "n"}
    if not required <= set(metrics):
        raise PipelineError("metrics.json is missing required fields")
    value = metrics["value"]
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PipelineError("Primary metric is not a finite number")
    if int(metrics["n"]) <= 0:
        raise PipelineError("Metric sample count must be positive")
    return metrics


def _find_plot(workspace: Path) -> Path:
    plots = sorted((workspace / "working").glob("*.png"))
    valid = [
        path
        for path in plots
        if path.stat().st_size > 1024 and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    ]
    if not valid:
        raise PipelineError("Plotting stage did not produce a valid PNG")
    return valid[0]


def _artifact_manifest(run_dir: Path) -> list[dict[str, Any]]:
    ignored_suffixes = {".aux", ".log", ".out"}
    values: list[dict[str, Any]] = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        if path.suffix in ignored_suffixes or path.name == "manifest.json":
            continue
        values.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return values


def run_pipeline(
    config: AppConfig,
    output_root: Path,
    provider_mode: str,
    vendor_root: Path,
) -> Path:
    if provider_mode not in {
        "fixture",
        "openrouter",
        "zhipu",
        "openai_compatible",
    }:
        raise ValueError(
            "provider_mode must be fixture, openrouter, zhipu, or openai_compatible"
        )

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "Z_"
        + uuid.uuid4().hex[:8]
    )
    run_dir = (output_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.jsonl"
    _event(events_path, "run", "started", provider_mode=provider_mode, run_id=run_id)

    ledger = BudgetLedger(run_dir / "budget.json", config.budget.hard_limit_usd)
    if provider_mode == "fixture":
        selected = fixture_models(config)
        provider: ChatProvider = FixtureProvider(selected)
    else:
        _event(events_path, "model_preflight", "started")
        selected = select_live_models(config)
        if provider_mode in {"zhipu", "openai_compatible"}:
            provider = ZhipuProvider(config, selected, ledger, run_dir / "responses")
        else:
            provider = OpenRouterProvider(config, selected, ledger, run_dir / "responses")
        _event(events_path, "model_preflight", "completed")

    selected_payload = {
        role: {
            "model_id": model.model_id,
            "context_length": model.context_length,
            "maximum_reserved_usd": model.maximum_cost(
                config.roles[role], config.budget.reserve_margin
            ),
        }
        for role, model in selected.items()
    }
    _write_json(run_dir / "selected_models.json", selected_payload)

    _event(events_path, "ideation", "started")
    idea_prompt = (
        "Propose exactly one tiny deterministic synthetic ML experiment for this topic:\n"
        f"{config.gate_a.topic}\n"
        "The experiment must run on CPU in under 60 seconds, require no downloads, and be explicitly described as an engineering smoke test."
    )
    idea, idea_meta = provider.call_json(
        "ideation",
        "ideation-000",
        "You are the ideation stage of a constrained AI-Scientist-v2 reproduction.",
        idea_prompt,
        "submit_idea",
        IDEA_SCHEMA,
    )
    jsonschema.validate(idea, IDEA_SCHEMA)
    _write_json(run_dir / "idea.json", idea)
    _event(events_path, "ideation", "completed", model=idea_meta["resolved_model"])

    runner = DockerRunner(config.runner)
    runner_image_id = runner.inspect_image_id()
    workspace = run_dir / "bfts" / "node_000" / "workspace"
    _event(events_path, "bfts", "started", nodes=config.gate_a.bfts_nodes)
    bridge = UpstreamMinimalBFTS(vendor_root, provider, config.runner.timeout_seconds)
    node, plotting_code = bridge.generate(idea, workspace)
    validate_generated_code(node.code)
    code_result = runner.run_python(node.code, workspace, "experiment.py")
    _write_json(
        run_dir / "bfts" / "node_000" / "execution.json", _execution_dict(code_result)
    )
    if not code_result.succeeded:
        raise RunnerError(
            f"Agent-generated experiment failed with exit code {code_result.exit_code}"
        )
    metrics = _validate_metrics(workspace)
    _write_json(
        run_dir / "bfts" / "node_000" / "node.json",
        {
            "node_id": str(node.id),
            "stage": "1_initial_implementation_1_preliminary",
            "plan": node.plan,
            "code_path": "workspace/experiment.py",
            "primary_metric": metrics,
            "upstream_component": "ai_scientist.treesearch.parallel_agent.MinimalAgent._draft",
        },
    )
    _event(events_path, "bfts", "completed", successful_nodes=1)

    _event(events_path, "plotting", "started")
    validate_generated_code(plotting_code)
    plot_result = runner.run_python(plotting_code, workspace, "plot.py")
    _write_json(
        run_dir / "bfts" / "node_000" / "plot_execution.json",
        _execution_dict(plot_result),
    )
    if not plot_result.succeeded:
        raise RunnerError(
            f"Agent-generated plotting failed with exit code {plot_result.exit_code}"
        )
    plot_path = _find_plot(workspace)
    _event(events_path, "plotting", "completed", plot=plot_path.name)

    _event(events_path, "paper", "started")
    paper_prompt = json.dumps(
        {
            "idea": idea,
            "experiment_plan": node.plan,
            "observed_metrics": metrics,
            "required_scope": [
                "English only",
                "preliminary engineering smoke test",
                "human review required",
                "no novelty, clinical, diagnostic, or real-world validity claim",
                "do not invent citations or unobserved results",
            ],
        },
        indent=2,
    )
    paper_content, paper_meta = provider.call_json(
        "paper_writer",
        "paper-000",
        "Write a concise and factual workshop-style engineering smoke paper from only the supplied evidence.",
        paper_prompt,
        "submit_paper",
        PAPER_SCHEMA,
    )
    jsonschema.validate(paper_content, PAPER_SCHEMA)
    paper_dir = run_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    figure_target = paper_dir / plot_path.name
    shutil.copyfile(plot_path, figure_target)
    tex_path = paper_dir / "paper.tex"
    tex_path.write_text(
        render_latex(paper_content, figure_target.name), encoding="utf-8"
    )
    latex_result = runner.compile_latex(paper_dir)
    _write_json(paper_dir / "compile.json", _execution_dict(latex_result))
    if not latex_result.succeeded:
        raise RunnerError(
            f"LaTeX compilation failed with exit code {latex_result.exit_code}"
        )
    pdf_path = paper_dir / "paper.pdf"
    validate_pdf(pdf_path)
    _write_json(paper_dir / "content.json", paper_content)
    _event(events_path, "paper", "completed", model=paper_meta["resolved_model"])

    _event(events_path, "review", "started")
    review_prompt = json.dumps(
        {
            "paper": paper_content,
            "observed_metrics": metrics,
            "disclosure_present": True,
        },
        indent=2,
    )
    review, review_meta = provider.call_json(
        "reviewer",
        "review-000",
        "Act as an independent reviewer. Penalize unsupported claims and check disclosure and artifact traceability.",
        review_prompt,
        "submit_review",
        REVIEW_SCHEMA,
    )
    jsonschema.validate(review, REVIEW_SCHEMA)
    _write_json(run_dir / "review.json", review)
    _event(events_path, "review", "completed", model=review_meta["resolved_model"])

    snapshot = ledger.snapshot()
    criteria = {
        "real_paid_provider": provider_mode
        in {"openrouter", "zhipu", "openai_compatible"},
        "nonempty_idea_and_plan": bool(idea and node.plan),
        "generated_code_executed_in_isolated_container": code_result.succeeded,
        "minimal_bfts_search_completed": True,
        "structured_metrics_present": bool(metrics),
        "valid_chart_present": plot_path.exists(),
        "english_latex_and_pdf_present": tex_path.exists() and pdf_path.exists(),
        "independent_model_review_present": (
            selected["paper_writer"].model_id != selected["reviewer"].model_id
            and bool(review)
        ),
        "no_unhandled_exception": True,
        "clean_runner_image_identified": bool(runner_image_id),
    }
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "provider_mode": provider_mode,
        "started_and_completed": True,
        "completed_at": _utc_now(),
        "runner_image_id": runner_image_id,
        "selected_models": selected_payload,
        "budget": snapshot.__dict__,
        "summary": {
            "primary_metric": metrics["primary_metric"],
            "value": metrics["value"],
            "bfts_nodes": 1,
            "plot": plot_path.name,
            "pdf": "paper/paper.pdf",
        },
        "criteria": criteria,
        "gate_a_passed": all(criteria.values()),
        "offline_ready": provider_mode == "fixture"
        and all(
            value for key, value in criteria.items() if key != "real_paid_provider"
        ),
    }
    _write_json(run_dir / "acceptance.json", report)
    _event(events_path, "run", "completed", gate_a_passed=report["gate_a_passed"])
    _write_json(run_dir / "manifest.json", {"artifacts": _artifact_manifest(run_dir)})
    return run_dir
