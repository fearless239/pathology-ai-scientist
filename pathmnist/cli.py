from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig, load_config
from .data import validate_dataset
from .freeze import (
    REQUIRED_APPROVAL,
    load_frozen_candidate,
    require_ready_checkpoints,
    write_once,
)
from .orchestration import aggregate_variant
from .reporting import generate_m4_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PathMNIST M4 experiment tooling")
    parser.add_argument("--config", type=Path, default=Path("configs/pathmnist_m4.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate dataset before paid work")
    validate.add_argument("--project-root", type=Path, default=Path("."))
    subparsers.add_parser("gpu-smoke", help="Run the minimal CUDA acceptance test")
    train = subparsers.add_parser("train", help="Run train/val experiments")
    train.add_argument("--project-root", type=Path, default=Path("."))
    train.add_argument("--output-root", type=Path, default=Path("runs/pathmnist-m4"))
    train.add_argument("--phase", choices=["tune", "main", "ablations", "all"], default="all")
    train.add_argument("--limit-epochs", type=int)
    train.add_argument("--smoke", action="store_true")
    prepare_final = subparsers.add_parser(
        "prepare-final", help="Retrain frozen candidate and save checkpoints"
    )
    prepare_final.add_argument("--project-root", type=Path, default=Path("."))
    prepare_final.add_argument("--output-root", type=Path, default=Path("runs/pathmnist-m4"))
    prepare_final.add_argument(
        "--candidate", type=Path, default=Path("configs/pathmnist_final_candidate.json")
    )
    evaluate_test = subparsers.add_parser(
        "evaluate-test", help="Perform the single frozen test evaluation"
    )
    evaluate_test.add_argument("--project-root", type=Path, default=Path("."))
    evaluate_test.add_argument("--output-root", type=Path, default=Path("runs/pathmnist-m4"))
    evaluate_test.add_argument(
        "--candidate", type=Path, default=Path("configs/pathmnist_final_candidate.json")
    )
    evaluate_test.add_argument("--approval", required=True)
    paper_smoke = subparsers.add_parser(
        "paper-smoke", help="Run a paid formal-paper smoke using frozen M4 artifacts"
    )
    paper_smoke.add_argument("--project-root", type=Path, default=Path("."))
    paper_smoke.add_argument("--state-root", type=Path, default=Path("state/workflow"))
    paper_smoke.add_argument("--task-id", default="formal-paper-paid-smoke")
    paper_smoke.add_argument("--budget-limit-usd", type=float, default=0.5)
    paper_smoke.add_argument("--confirm-paid", action="store_true")
    paper_export = subparsers.add_parser(
        "paper-export", help="Render final paper Markdown as LaTeX source"
    )
    paper_export.add_argument("--input", type=Path, required=True)
    paper_export.add_argument("--output", type=Path, required=True)
    paper_export.add_argument("--language", choices=["en", "zh"], default="en")
    archive = subparsers.add_parser(
        "archive", help="Build the lightweight M4/M5 evidence archive"
    )
    archive.add_argument("--project-root", type=Path, default=Path("."))
    archive.add_argument(
        "--output", type=Path, default=Path("runs/archives/pathmnist-m4-m5-archive.zip")
    )
    report = subparsers.add_parser("report", help="Generate the final deterministic M4 report")
    report.add_argument("--project-root", type=Path, default=Path("."))
    report.add_argument("--run-root", type=Path, default=Path("runs/pathmnist-m4"))
    report.add_argument("--candidate", type=Path, default=Path("configs/pathmnist_final_candidate.json"))
    report.add_argument("--output", type=Path, default=Path("docs/M4_FINAL_REPORT.md"))
    workflow_run = subparsers.add_parser(
        "workflow-run", help="Create or resume one complete pathology-AI research workflow"
    )
    workflow_run.add_argument("--project-root", type=Path, default=Path("."))
    workflow_run.add_argument("--state-root", type=Path, default=Path("state/workflow"))
    workflow_run.add_argument("--task-id", required=True)
    workflow_run.add_argument("--direction", required=True)
    workflow_run.add_argument("--goal", default="")
    workflow_run.add_argument("--budget-limit-usd", type=float, default=2.0)
    workflow_run.add_argument("--execution-hours", type=float, default=6.0)
    workflow_run.add_argument("--enable-llm", action="store_true")
    workflow_run.add_argument("--enable-real-training", action="store_true")
    framework_smoke = subparsers.add_parser(
        "framework-smoke", help="Verify real AI-Scientist-v2 imports in the container runtime"
    )
    framework_smoke.add_argument("--project-root", type=Path, default=Path("."))
    autonomous_init = subparsers.add_parser(
        "autonomous-init", help="Create a research task and sealed research dataset view"
    )
    autonomous_init.add_argument("--state-root", type=Path, default=Path("state/workflow"))
    autonomous_init.add_argument("--task-id", required=True)
    autonomous_init.add_argument("--dataset-path", type=Path, required=True)
    autonomous_init.add_argument("--direction", required=True)
    autonomous_init.add_argument("--seed", type=int, default=7)
    autonomous_init.add_argument(
        "--resume", action="store_true", help="Validate and repair metadata for an existing v2 task"
    )
    return parser


def _dataset(config: AppConfig, project_root: Path):
    return type(config.dataset)(
        path=project_root.resolve() / config.dataset.path,
        sha256=config.dataset.sha256,
        expected_splits=config.dataset.expected_splits,
        classes=config.dataset.classes,
    )


def gpu_smoke() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    inputs = torch.randn(64, 3, 64, 64, device=device)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 2),
    ).to(device)
    targets = torch.randint(0, 2, (64,), device=device)
    outputs = model(inputs)
    loss = torch.nn.CrossEntropyLoss()(outputs, targets)
    loss.backward()
    torch.cuda.synchronize(device)
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(device),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "loss_backward": True,
    }


def _limited(config: AppConfig, epochs: int) -> AppConfig:
    experiment = config.experiment
    return AppConfig(
        dataset=config.dataset,
        experiment=type(experiment)(
            primary_metric=experiment.primary_metric,
            seeds=experiment.seeds,
            epochs=epochs,
            batch_size=experiment.batch_size,
            num_workers=experiment.num_workers,
            early_stop_patience=experiment.early_stop_patience,
            baseline_learning_rate=experiment.baseline_learning_rate,
            learning_rates=experiment.learning_rates,
            weight_decays=experiment.weight_decays,
            variants=experiment.variants,
            ablations=experiment.ablations,
        ),
        raw=config.raw,
    )


def _train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from .datasets import load_archive
    from .train import run_single, tune

    original = load_config(args.config.resolve())
    validate_dataset(_dataset(original, args.project_root))
    config = _limited(original, args.limit_epochs) if args.limit_epochs else original
    if args.smoke and args.limit_epochs is None:
        raise RuntimeError("Smoke mode requires --limit-epochs")
    archive = load_archive(_dataset(config, args.project_root).path)
    if args.smoke:
        archive = {name: array[:512] for name, array in archive.items()}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_root = args.output_root.resolve()
    completed: dict[str, Any] = {}
    phases = [args.phase] if args.phase != "all" else ["tune", "main", "ablations"]
    for phase in phases:
        if phase == "tune":
            completed["tuning"] = str(tune(config, archive, output_root, device))
        else:
            variants = config.experiment.variants if phase == "main" else config.experiment.ablations
            for variant in variants:
                for seed in config.experiment.seeds:
                    run_single(config, variant, seed, archive, output_root / phase, device)
                completed[f"{phase}:{variant.name}"] = aggregate_variant(
                    output_root / phase / variant.name
                )
    return {
        "completed": completed,
        "smoke": args.smoke,
        "output_root": str(output_root),
        "python_version": platform.python_version(),
    }


def _prepare_final(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from .datasets import load_archive
    from .train import run_single

    config = load_config(args.config.resolve())
    candidate = load_frozen_candidate(args.candidate.resolve())
    dataset = _dataset(config, args.project_root)
    validate_dataset(dataset)
    if candidate.dataset_sha256 != config.dataset.sha256:
        raise RuntimeError("Frozen dataset hash differs from experiment config")
    archive = load_archive(dataset.path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_root = args.output_root.resolve()
    final_root = output_root / "final"
    if (output_root / "test_evaluation.json").exists():
        raise RuntimeError("Refusing final training after test evaluation")
    variant = next(item for item in config.experiment.variants if item.name == candidate.variant)
    completed = []
    for seed in candidate.seeds:
        run_dir = run_single(
            config,
            variant,
            seed,
            archive,
            final_root,
            device,
            candidate.learning_rate,
            candidate.weight_decay,
            save_checkpoint=True,
        )
        completed.append(str(run_dir))
    require_ready_checkpoints(candidate, config.dataset.sha256, final_root)
    return {"candidate": candidate.variant, "checkpoints": completed}


def _evaluate_test(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from .datasets import load_archive, make_loader
    from .models import SmallResNet
    from .train import evaluate

    if args.approval != REQUIRED_APPROVAL:
        raise RuntimeError("Test evaluation approval string is incorrect")
    config = load_config(args.config.resolve())
    candidate = load_frozen_candidate(args.candidate.resolve())
    dataset = _dataset(config, args.project_root)
    validate_dataset(dataset)
    output_root = args.output_root.resolve()
    final_root = output_root / "final"
    checkpoints = require_ready_checkpoints(candidate, config.dataset.sha256, final_root)
    result_path = output_root / "test_evaluation.json"
    if result_path.exists():
        raise RuntimeError("Test set has already been evaluated")

    archive = load_archive(dataset.path)
    loader = make_loader(
        archive, "test", config.experiment.batch_size, False, config.experiment.num_workers, 0
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    per_seed = []
    confusion = [[0 for _ in range(config.dataset.classes)] for _ in range(config.dataset.classes)]
    for seed, checkpoint_file in zip(candidate.seeds, checkpoints, strict=True):
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
        model = SmallResNet(
            classes=config.dataset.classes, multiscale=candidate.multiscale
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        loss, macro_f1, matrix = evaluate(model, loader, device)
        accuracy = sum(matrix[index][index] for index in range(config.dataset.classes)) / sum(
            sum(row) for row in matrix
        )
        per_seed.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_file),
                "best_epoch": int(checkpoint["best_epoch"]),
                "loss": loss,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "confusion_matrix": matrix,
            }
        )
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                confusion[row_index][column_index] += value
    scores = [item["macro_f1"] for item in per_seed]
    mean_score = sum(scores) / len(scores)
    payload = {
        "candidate": "optimization",
        "split": "test",
        "evaluation_count": 1,
        "per_seed": per_seed,
        "aggregate_confusion_matrix": confusion,
        "macro_f1_mean": mean_score,
        "macro_f1_std": (
            sum((score - mean_score) ** 2 for score in scores) / (len(scores) - 1)
        )
        ** 0.5,
        "environment": {
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
    }
    write_once(result_path, payload)
    return payload


def _report(args: argparse.Namespace) -> dict[str, str]:
    report_path = generate_m4_report(
        args.run_root.resolve(),
        args.candidate.resolve(),
        args.output.resolve(),
    )
    return {"report": str(report_path)}


def _paper_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_paid:
        raise RuntimeError("Paper smoke requires --confirm-paid")
    from .workflow import (
        STAGES,
        ResourcePolicy,
        WorkflowStore,
        advance,
        task_executor,
    )

    project_root = args.project_root.resolve()
    state_root = args.state_root.resolve()
    task_id = args.task_id
    store = WorkflowStore(state_root)
    if not (state_root / f"{task_id}.json").exists():
        store.create(
            task_id,
            {
                "mode": "autonomous",
                "budget_limit_usd": args.budget_limit_usd,
                "execution_limit_seconds": 3600,
                "llm_config_path": str(
                    project_root / "configs" / "pathmnist_paper_llm.yaml"
                ),
            },
        )
    state = store.load(task_id)
    read_only_executor = task_executor(project_root, state_root, task_id)
    read_only_executor.training_scheduler = None
    read_only_executor.resource_policy = ResourcePolicy(
        require_gpu=False,
        require_ac_power=False,
        minimum_free_gb=0,
        minimum_memory_gb=0,
    )
    analysis_index = STAGES.index("analysis_completed")
    while state.completed_stage != "analysis_completed":
        state = advance(
            store,
            task_id,
            read_only_executor,
            state_root / task_id / "artifacts",
        )
        if STAGES.index(state.completed_stage) > analysis_index:
            raise RuntimeError("Paper smoke unexpectedly advanced beyond analysis")
    for stage in STAGES[STAGES.index("paper_approved") :]:
        state = advance(
            store, task_id, read_only_executor, state_root / task_id / "artifacts"
        )
    return {
        "completed_stage": state.completed_stage,
        "budget_limit_usd": state.config["budget_limit_usd"],
        "artifacts": str(state_root / task_id / "artifacts"),
        "usage": str(state_root / task_id / "llm_usage.jsonl"),
    }


def _paper_export(args: argparse.Namespace) -> dict[str, str]:
    from .paper_export import markdown_to_latex

    markdown = args.input.resolve().read_text(encoding="utf-8")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown_to_latex(markdown, args.language), encoding="utf-8")
    return {"latex": str(output)}


def _archive(args: argparse.Namespace) -> dict[str, object]:
    from .archive import build_archive

    return build_archive(args.project_root.resolve(), args.output.resolve())


def _workflow_run(args: argparse.Namespace) -> dict[str, object]:
    from .workflow import (
        STAGES,
        WorkflowStore,
        advance,
        task_executor,
        task_budget_path,
        validate_completed_artifacts,
    )
    from gate_a.budget import BudgetLedger

    project_root = args.project_root.resolve()
    state_root = args.state_root.resolve()
    store = WorkflowStore(state_root)
    if not store.path(args.task_id).is_file():
        store.create(
            args.task_id,
            {
                "mode": "autonomous",
                "budget_limit_usd": args.budget_limit_usd,
                "execution_limit_seconds": int(args.execution_hours * 3600),
                "enable_real_training": args.enable_real_training,
                "llm_config_path": (
                    str(project_root / "configs" / "gate_a_llm.yaml")
                    if args.enable_llm
                    else ""
                ),
                "research_direction": args.direction.strip(),
                "research_goal": args.goal.strip(),
            },
        )
    state = store.load(args.task_id)
    if state.config.get("research_direction") != args.direction.strip():
        raise RuntimeError("Existing task research direction differs from --direction")
    artifacts = state_root / args.task_id / "artifacts"
    while state.completed_stage != STAGES[-1]:
        state = advance(
            store,
            args.task_id,
            task_executor(project_root, state_root, args.task_id),
            artifacts,
        )
    final_paper = artifacts / "revision_completed" / "final_paper.md"
    budget = BudgetLedger(
        task_budget_path(state_root, args.task_id), state.config["budget_limit_usd"]
    ).snapshot()
    return {
        "task_id": args.task_id,
        "completed_stage": state.completed_stage,
        "stage_count": len(STAGES),
        "artifacts_valid": validate_completed_artifacts(state, artifacts),
        "final_paper": str(final_paper),
        "final_paper_exists": final_paper.is_file(),
        "framework_record": str(artifacts / "task_created" / "framework.json"),
        "spent_usd": budget.spent_usd,
        "available_usd": budget.available_usd,
    }


def _framework_smoke(args: argparse.Namespace) -> dict[str, object]:
    from .upstream_adapter import PathologyAIScientistV2Adapter

    record = PathologyAIScientistV2Adapter(args.project_root.resolve()).framework_record()
    if not record["runtime_imported"]:
        raise RuntimeError(
            "AI-Scientist-v2 imports were deferred; run this check with Python >=3.11"
        )
    return {
        "framework_imported": True,
        "framework": record["framework"],
        "minimal_agent_module": record["minimal_agent_module"],
        "agent_manager_module": record["agent_manager_module"],
        "upstream_manifest_sha256": record["upstream_manifest_sha256"],
        "python_version": platform.python_version(),
    }


def _autonomous_init(args: argparse.Namespace) -> dict[str, object]:
    from .autonomous import AutonomousTaskWorkspace, V2_STAGES, pathology_task_description
    from .dataset_adapter import DatasetAdapter

    state_root = args.state_root.resolve()
    workspace = AutonomousTaskWorkspace.create(state_root, args.task_id)
    task_file = workspace.root / "task.json"
    existing = None
    if task_file.exists() and not args.resume:
        raise RuntimeError(f"Autonomous task already exists: {args.task_id}")
    if task_file.exists():
        existing = json.loads(task_file.read_text(encoding="utf-8"))
        if existing.get("schema_version") != 2:
            raise RuntimeError("--resume cannot convert a legacy task")
        if existing.get("research_direction") != args.direction.strip():
            raise RuntimeError("Existing task research direction differs from --direction")
    spec = DatasetAdapter(seed=args.seed).discover(
        args.dataset_path.resolve(), workspace.dataset / "dataset_profile.json"
    )
    research_view = workspace.dataset / "research_view"
    if not research_view.is_dir() or not any(research_view.iterdir()):
        research_view = workspace.prepare_research_dataset(spec)
    research_contract = pathology_task_description(args.direction, spec)
    (workspace.research / "agent_task.json").write_text(
        research_contract + "\n", encoding="utf-8"
    )
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        **(existing or {}),
        "schema_version": 2,
        "scientific_integrity_schema_version": 1,
        "task_type": "autonomous_experiment",
        "publication_backend": existing.get("publication_backend", "legacy_local") if existing else "upstream_v2",
        "task_id": args.task_id,
        "research_direction": args.direction.strip(),
        "budget_limit_usd": 8.0,
        "dataset_profile": "dataset/dataset_profile.json",
        "research_dataset": "dataset/research_view",
        "seed": args.seed,
        "stages": {stage: ("completed" if stage in {"task_created", "dataset_discovered", "dataset_validated"} else "waiting") for stage in V2_STAGES},
        "completed_stage": "dataset_validated",
        "control": "paused",
        "created_at": now,
        "updated_at": now,
    }
    temporary = task_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(task_file)
    return {
        "schema_version": 2,
        "task_id": args.task_id,
        "completed_stage": "dataset_validated",
        "dataset_name": spec.name,
        "source_type": spec.source_type,
        "classes": spec.classes,
        "split_counts": spec.split_counts,
        "content_sha256": spec.content_sha256,
        "research_view": str(research_view),
        "test_materialized_in_research_view": False,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.command == "gpu-smoke":
        result = gpu_smoke()
    elif args.command == "train":
        result = _train(args)
    elif args.command == "prepare-final":
        result = _prepare_final(args)
    elif args.command == "evaluate-test":
        result = _evaluate_test(args)
    elif args.command == "report":
        result = _report(args)
    elif args.command == "paper-smoke":
        result = _paper_smoke(args)
    elif args.command == "paper-export":
        result = _paper_export(args)
    elif args.command == "archive":
        result = _archive(args)
    elif args.command == "workflow-run":
        result = _workflow_run(args)
    elif args.command == "framework-smoke":
        result = _framework_smoke(args)
    elif args.command == "autonomous-init":
        result = _autonomous_init(args)
    else:
        summary = validate_dataset(_dataset(load_config(args.config.resolve()), args.project_root))
        result = {
            "dataset_valid": True,
            **summary.__dict__,
            "python_version": platform.python_version(),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
