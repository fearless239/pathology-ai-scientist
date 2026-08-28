from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from gate_a.config import RunnerConfig
from gate_a.runner import DockerRunner

from .autonomous import AIScientistExperimentRunner, AutonomousTaskWorkspace, pathology_task_description
from .autonomous_acceptance import require_task
from .dataset_adapter import DatasetSpec, SampleRecord
from .research_contract import load_contract


class NoInferenceProvider:
    def call_text(self, *args, **kwargs):
        raise AssertionError("Offline preflight must not call an LLM")


def _load_spec(path: Path) -> DatasetSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        key: raw[key]
        for key in (
            "schema_version", "name", "source_type", "source_path", "content_sha256",
            "image_shape", "channels", "classes", "label_mapping", "split_counts",
            "class_counts",
        )
    }
    return DatasetSpec(
        **fields,
        samples=[SampleRecord(**sample) for sample in raw["samples"]],
        has_group_ids=raw.get("has_group_ids", False),
        inference=raw.get("inference", []),
        warnings=raw.get("warnings", []),
        confidence=float(raw.get("confidence", 1.0)),
        recommended_metrics=raw.get("recommended_metrics", ["macro_f1", "accuracy"]),
    )


def run_preflight(project_root: Path, state_root: Path, task_id: str) -> dict[str, object]:
    project_root, state_root = project_root.resolve(), state_root.resolve()
    workspace = AutonomousTaskWorkspace.create(state_root, task_id)
    task = json.loads((workspace.root / "task.json").read_text(encoding="utf-8"))
    if task.get("schema_version") != 2:
        raise RuntimeError("Research preflight requires a current research task")
    require_task(workspace.root, "experiment_spec_validated")
    spec = _load_spec(workspace.dataset / "dataset_profile.json")
    contract = load_contract(workspace.root, require_approved=True)
    from .stage_policy import check_upstream_compatibility

    check_upstream_compatibility(contract)
    research_view = workspace.dataset / "research_view"
    from .dataset_adapter import research_view_interface
    interface = research_view_interface(spec, research_view)
    if (research_view / "dataset.npz").is_file():
        with __import__("numpy").load(research_view / "dataset.npz", allow_pickle=False) as view:
            mounted_arrays = sorted(view.files)
    else:
        rows = json.loads((research_view / "manifest.json").read_text(encoding="utf-8"))
        mounted_arrays = sorted({str(row.get("split")) for row in rows})
    if any(name.startswith("test") for name in mounted_arrays):
        raise RuntimeError("Research view contains a test array")

    docker_runner = DockerRunner(
        RunnerConfig(
            image="path-scientist-gate-a-runner:0.2",
            timeout_seconds=30,
            cpus=1.0,
            memory="1g",
            pids_limit=32,
            docker_command=("docker",),
        )
    )
    runner = AIScientistExperimentRunner(project_root, NoInferenceProvider(), docker_runner)
    cfg = OmegaConf.load(project_root / "vendor/AI-Scientist-v2/bfts_config.yaml")
    cfg.workspace_dir = str(workspace.experiment_workspace)
    cfg.log_dir = workspace.experiment_logs
    cfg.agent.num_workers = 1
    manager_class, _, _, _ = runner._runtime_classes(research_view)
    manager = manager_class(
        task_desc=pathology_task_description(task["research_direction"], spec, contract, research_view=research_view),
        cfg=cfg,
        workspace_dir=workspace.experiment_workspace,
    )
    goals = dict(manager.main_stage_goals)
    forbidden = ("huggingface", "two more", "three huggingface")
    if any(term in json.dumps(goals).casefold() for term in forbidden):
        raise RuntimeError("Pathology AgentManager retained an external-dataset goal")
    checkpoint = workspace.experiment_logs / "preflight-manager.pkl"
    runner._checkpoint(checkpoint, manager)
    restored = runner._load_or_create_manager(
        checkpoint,
        manager_class,
        pathology_task_description(task["research_direction"], spec, contract, research_view=research_view),
        cfg,
        workspace,
    )

    smoke_code = """import json, os
print(json.dumps({
    'dataset_entries': sorted(os.listdir('/dataset')),
    'docker_socket_visible': os.path.exists('/var/run/docker.sock'),
    'test_path_visible': os.path.exists('/dataset/test'),
}))
"""
    execution = docker_runner.run_python(
        smoke_code,
        workspace.experiment_workspace / "sandbox_preflight",
        dataset_mount=research_view,
        enforce_model_contract=False,
    )
    if not execution.succeeded:
        raise RuntimeError(f"Sandbox preflight failed: {execution.stderr}")
    sandbox = json.loads(execution.stdout.strip().splitlines()[-1])
    if sandbox["docker_socket_visible"] or sandbox["test_path_visible"]:
        raise RuntimeError("Generated-code sandbox can access a forbidden resource")
    report = {
        "schema_version": 2,
        "task_id": task_id,
        "agent_manager": f"{manager_class.__module__}.{manager_class.__name__}",
        "four_stage_goals": goals,
        "checkpoint_restored": restored.current_stage.name == manager.current_stage.name,
        "mounted_arrays": mounted_arrays,
        "dataset_interface": interface,
        "sandbox": sandbox,
        "llm_calls": 0,
        "passed": True,
    }
    (workspace.research / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    task = json.loads((workspace.root / "task.json").read_text(encoding="utf-8"))
    task["stages"]["sandbox_prechecked"] = "completed"
    task["completed_stage"] = "sandbox_prechecked"
    (workspace.root / "task.json").write_text(
        json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.project_root, args.state_root, args.task_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
