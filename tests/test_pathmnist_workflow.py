import json

import pytest

from pathmnist.workflow import (
    RunMode,
    STAGES,
    STAGE_CONTRACTS,
    WorkflowConfig,
    WorkflowContext,
    WorkflowError,
    WorkflowExecutor,
    repair_to_valid_stage,
    reset_interrupted_stage,
    WorkflowStore,
    advance,
    approve,
    resume,
    requires_approval,
    set_control,
)


def test_reset_interrupted_llm_stage_releases_unfinished_request(tmp_path):
    from gate_a.budget import BudgetLedger

    root = tmp_path / "state" / "workflow"
    store = WorkflowStore(root)
    store.create("retry-task", config())
    ledger = BudgetLedger(root / "retry-task" / "budget.json", 50)
    assert ledger.reserve("retry-task-topic", 0.5, {"role": "ideation"})
    state = store.load("retry-task")
    state.stages["topic_proposed"].update(
        status="interrupted", retries=3, error="provider connection lost"
    )
    store.save(state)

    reset = reset_interrupted_stage(store, "retry-task", "topic_proposed")

    assert reset.stages["topic_proposed"]["status"] == "waiting"
    assert reset.stages["topic_proposed"]["retries"] == 0
    assert ledger.snapshot().reserved_usd == 0
    assert ledger.reserve("retry-task-topic", 0.5, {"role": "ideation"})


def test_reset_interrupted_llm_stage_keeps_settled_cached_request(tmp_path):
    from gate_a.budget import BudgetLedger

    root = tmp_path / "state" / "workflow"
    store = WorkflowStore(root)
    store.create("cached-task", config())
    ledger = BudgetLedger(root / "cached-task" / "budget.json", 50)
    assert ledger.reserve("cached-task-topic", 0.5, {"role": "ideation"})
    ledger.settle("cached-task-topic", 0.1, {"prompt_tokens": 1})
    response = root / "cached-task" / "responses" / "cached-task-topic.json"
    response.parent.mkdir(parents=True)
    response.write_text("{}", encoding="utf-8")
    state = store.load("cached-task")
    state.stages["topic_proposed"].update(status="interrupted", retries=1, error="disk")
    store.save(state)

    reset_interrupted_stage(store, "cached-task", "topic_proposed")

    assert ledger.snapshot().spent_usd == 0.1
    assert ledger.reserve("cached-task-topic", 0.5, {}) is False


def config(mode="staged_approval"):
    return {
        "mode": mode,
        "budget_limit_usd": 50,
        "execution_limit_seconds": 21600,
    }


def test_config_bounds_and_mode_validation():
    assert WorkflowConfig.from_mapping(config()).mode is RunMode.STAGED_APPROVAL
    with pytest.raises(WorkflowError):
        WorkflowConfig.from_mapping({**config(), "budget_limit_usd": 51})
    with pytest.raises(WorkflowError):
        WorkflowConfig.from_mapping({**config(), "execution_limit_seconds": 21601})


def test_staged_approval_pauses_at_topic_gate(tmp_path):
    store = WorkflowStore(tmp_path)
    state = store.create("task-1", config())
    state = stub_advance(store, "task-1", tmp_path)
    assert state.completed_stage == "task_created"
    for _ in range(6):
        state = stub_advance(store, "task-1", tmp_path)
    assert state.completed_stage == "topic_proposed"
    state = stub_advance(store, "task-1", tmp_path)
    assert state.stages["topic_approved"]["status"] == "waiting_approval"
    state = approve(store, "task-1", "topic_approved", StubExecutor(), tmp_path / "artifacts")
    assert state.completed_stage == "topic_approved"
    assert state.control == "running"


def test_autonomous_mode_skips_normal_approval_gates(tmp_path):
    store = WorkflowStore(tmp_path)
    store.create("task-2", config("autonomous"))
    for _ in range(6):
        stub_advance(store, "task-2", tmp_path)
    state = stub_advance(store, "task-2", tmp_path)
    assert state.completed_stage == "topic_approved"
    assert state.control == "running"


def test_pause_resume_cancel_semantics(tmp_path):
    store = WorkflowStore(tmp_path)
    store.create("task-3", config("autonomous"))
    stub_advance(store, "task-3", tmp_path)
    set_control(store, "task-3", "paused")
    with pytest.raises(WorkflowError, match="paused"):
        stub_advance(store, "task-3", tmp_path)
    assert resume(store, "task-3") == "task_created"
    set_control(store, "task-3", "cancelled")
    with pytest.raises(WorkflowError, match="Cancelled"):
        set_control(store, "task-3", "running")


def test_limit_and_resume_from_last_completed_stage(tmp_path):
    store = WorkflowStore(tmp_path)
    store.create("task-4", config("autonomous"))
    state = store.load("task-4")
    state.execution_seconds = 21600
    store.save(state)
    with pytest.raises(WorkflowError, match="time limit"):
        stub_advance(store, "task-4", tmp_path)
    state.execution_seconds = 0
    state.spent_usd = 50
    store.save(state)
    with pytest.raises(WorkflowError, match="Budget"):
        stub_advance(store, "task-4", tmp_path)


def test_advance_rejects_without_executor(tmp_path):
    store = WorkflowStore(tmp_path)
    store.create("no-executor", config("autonomous"))
    with pytest.raises(WorkflowError, match="No real executor"):
        advance(store, "no-executor")


class StubExecutor:
    def execute(self, stage, state, artifact_root):
        path = artifact_root / list(STAGE_CONTRACTS[stage])[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return {"ok": True}


def stub_advance(store, task_id, tmp_path):
    return advance(store, task_id, StubExecutor(), tmp_path / "artifacts")


def real_executor(project_root):
    from pathmnist.workflow import ResourcePolicy

    return WorkflowExecutor(
        WorkflowContext(
            project_root=project_root,
            config_path=project_root / "configs/pathmnist_m4.yaml",
            run_root=project_root / "runs/pathmnist-m4",
            report_path=project_root / "docs/M4_FINAL_REPORT.md",
            candidate_path=project_root / "configs/pathmnist_final_candidate.json",
        ),
        ResourcePolicy(require_gpu=False, require_ac_power=False),
    )


def require_real_dataset(project_root):
    if not (project_root / "pathmnist_64.npz").is_file():
        pytest.skip("manual real-PathMNIST integration test; dataset is intentionally absent from Git")


def run_stages(store, task_id, count, project_root, artifact_root, executor=None):
    for _ in range(count):
        state = store.load(task_id)
        index = STAGES.index(state.completed_stage) + 1 if state.completed_stage else 0
        stage = STAGES[index]
        executor = executor or real_executor(project_root)
        if requires_approval(stage, RunMode(state.config["mode"])):
            approve(store, task_id, stage, executor, artifact_root)
        else:
            advance(store, task_id, executor, artifact_root)
    return store.load(task_id)


def test_research_and_planning_handlers(project_root, tmp_path):
    require_real_dataset(project_root)
    store = WorkflowStore(tmp_path)
    store.create("research-task", config("autonomous"))
    state = run_stages(store, "research-task", 8, project_root, tmp_path / "artifacts")
    assert state.completed_stage == "experiment_planned"
    for stage, filename in {
        "research_understood": "research.json",
        "literature_collected": "literature.json",
        "topic_proposed": "topic.json",
        "experiment_planned": "experiment_plan.json",
    }.items():
        artifact = tmp_path / "artifacts" / stage / filename
        assert artifact.is_file()
        assert state.stages[stage]["outputs"]["artifacts"][filename]


def test_experiment_summary_handlers(project_root, tmp_path):
    require_real_dataset(project_root)
    store = WorkflowStore(tmp_path)
    store.create("summary-task", config("autonomous"))
    state = run_stages(store, "summary-task", 15, project_root, tmp_path / "artifacts")
    assert state.completed_stage == "main_comparison_completed"
    for stage, filename in {
        "baseline_completed": "baseline.json",
        "improvements_completed": "improvements.json",
        "tuning_completed": "tuning.json",
        "ablations_completed": "ablations.json",
        "main_comparison_completed": "main_comparison.json",
    }.items():
        assert (tmp_path / "artifacts" / stage / filename).is_file()




def test_stage_cost_reservation_enforces_budget(tmp_path):
    store = WorkflowStore(tmp_path)
    store.create("cost-task", config("autonomous"))
    state = store.load("cost-task")
    state.stages["task_created"]["inputs"] = {"reserved_usd": 0.2, "cost_usd": 0.1}
    store.save(state)

    class BudgetExecutor:
        def __init__(self):
            self.events = []

        def reserve_cost(self, task_id, stage, amount_usd):
            self.events.append(("reserve", task_id, stage, amount_usd))
            return True

        def settle_cost(self, task_id, stage, actual_usd, usage):
            self.events.append(("settle", task_id, stage, actual_usd, usage))

        def release_cost(self, task_id, stage):
            self.events.append(("release", task_id, stage))

        def execute(self, stage, state, artifact_root):
            path = artifact_root / list(STAGE_CONTRACTS[stage])[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            return {path.name: "stub-sha256"}

    executor = BudgetExecutor()
    state = advance(store, "cost-task", executor, tmp_path / "artifacts")
    assert state.spent_usd == 0.1
    assert state.reserved_usd == 0.0
    assert state.stages["task_created"]["outputs"]["cost_usd"] == 0.1
    assert executor.events[0] == ("reserve", "cost-task", "task_created", 0.2)
    assert executor.events[1][0:3] == ("settle", "cost-task", "task_created")
    assert executor.events[1][3] == 0.1
    assert executor.events[1][4]["task_id"] == "cost-task"
    assert executor.events[1][4]["stage"] == "task_created"


def test_stage_cost_reservation_blocks_insufficient_budget(tmp_path):
    store = WorkflowStore(tmp_path)
    store.create("budget-task", config("autonomous"))
    state = store.load("budget-task")
    state.spent_usd = 50.0
    state.stages["task_created"]["inputs"] = {"reserved_usd": 0.2, "cost_usd": 0.1}
    store.save(state)

    class FailingExecutor:
        def reserve_cost(self, task_id, stage, amount_usd):
            raise AssertionError("stage should not reserve")

        def execute(self, stage, state, artifact_root):
            raise AssertionError("stage should not execute")

    with pytest.raises(WorkflowError, match="Budget limit reached"):
        advance(store, "budget-task", FailingExecutor(), tmp_path / "artifacts")


def test_task_creation_initializes_budget_ledger(tmp_path):
    from gate_a.budget import BudgetLedger
    from pathmnist.workflow import task_budget_path

    state_root = tmp_path / "state" / "workflow"
    store = WorkflowStore(state_root)
    store.create("ledger-task", config("autonomous"))
    snapshot = BudgetLedger(
        task_budget_path(state_root, "ledger-task"), 50.0
    ).snapshot()
    assert snapshot.hard_limit_usd == 50.0
    assert snapshot.spent_usd == 0.0
    assert snapshot.reserved_usd == 0.0
    assert snapshot.available_usd == 50.0


def test_task_executor_binds_persistent_budget(project_root, tmp_path):
    from pathmnist.workflow import task_budget_path, task_executor

    state_root = tmp_path / "state" / "workflow"
    store = WorkflowStore(state_root)
    store.create("bound-task", config("autonomous"))
    executor = task_executor(project_root, state_root, "bound-task")
    assert executor.cost_ledger.ledger.snapshot().hard_limit_usd == 50.0
    assert task_budget_path(state_root, "bound-task").is_file()


def test_gate_a_stage_cost_ledger_is_idempotent(tmp_path):
    from gate_a.budget import BudgetLedger
    from pathmnist.workflow import GateAStageCostLedger

    ledger = BudgetLedger(tmp_path / "budget.json", 2.0)
    adapter = GateAStageCostLedger(ledger)
    assert adapter.reserve("cost-task", "task_created", 0.2)
    adapter.settle("cost-task", "task_created", 0.1, {"execution_seconds": 1.0})
    snapshot = ledger.snapshot()
    assert snapshot.spent_usd == 0.1
    assert snapshot.reserved_usd == 0.0
    assert not adapter.reserve("cost-task", "task_created", 0.2)


def test_gate_a_stage_cost_ledger_releases_failed_stage(tmp_path):
    from gate_a.budget import BudgetLedger
    from pathmnist.workflow import GateAStageCostLedger

    ledger = BudgetLedger(tmp_path / "budget.json", 2.0)
    adapter = GateAStageCostLedger(ledger)
    assert adapter.reserve("cost-task", "task_created", 0.2)
    adapter.release("cost-task", "task_created")
    assert ledger.snapshot().available_usd == 2.0


def test_workflow_llm_client_records_usage(tmp_path):
    from pathmnist.workflow import WorkflowLLMClient

    class StubProvider:
        def call_text(self, role, request_id, system, prompt):
            return "result", {
                "request_id": request_id,
                "resolved_model": "stub-model",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "actual_cost_usd": 0.123,
            }

    usage_path = tmp_path / "llm_usage.jsonl"
    client = WorkflowLLMClient(StubProvider(), usage_path)
    value, metadata = client.call_text(
        "ideation", "request-1", "system", "prompt"
    )
    assert value == "result"
    assert metadata["actual_cost_usd"] == 0.123
    records = [json.loads(line) for line in usage_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["request_id"] == "request-1"
    assert records[0]["usage"]["prompt_tokens"] == 10
    assert WorkflowLLMClient.total_cost_usd(usage_path) == 0.123


class CompletedProcess:
    def wait(self):
        return 0


class FailedProcess:
    def wait(self):
        return 1


def test_llm_config_is_explicit_and_defaults_off(project_root, tmp_path):
    from pathmnist.workflow import task_executor

    assert WorkflowConfig.from_mapping(config()).llm_config_path == ""
    assert WorkflowConfig.from_mapping(
        {**config(), "llm_config_path": "configs/gate_a_llm.yaml"}
    ).llm_config_path == "configs/gate_a_llm.yaml"

    state_root = tmp_path / "state" / "workflow"
    store = WorkflowStore(state_root)
    store.create("offline-llm", config("autonomous"))
    assert task_executor(project_root, state_root, "offline-llm").llm_client is None

    store.create(
        "live-llm",
        {
            **config("autonomous"),
            "llm_config_path": str(project_root / "configs" / "gate_a_llm.yaml"),
        },
    )
    try:
        executor = task_executor(project_root, state_root, "live-llm")
        assert executor.llm_client is not None
        assert "responses" in str(executor.llm_client.provider.response_dir)
    except RuntimeError:
        pytest.skip("LLM API key is intentionally unavailable in offline tests")


def test_real_training_mode_is_explicit(project_root, tmp_path):
    from pathmnist.workflow import TrainingScheduler, task_executor

    assert WorkflowConfig.from_mapping(config()).enable_real_training is False
    assert WorkflowConfig.from_mapping(
        {**config(), "enable_real_training": True}
    ).enable_real_training is True

    state_root = tmp_path / "state" / "workflow"
    store = WorkflowStore(state_root)
    store.create("readonly-training", config("autonomous"))
    read_only_executor = task_executor(project_root, state_root, "readonly-training")
    assert read_only_executor.training_scheduler is None
    assert read_only_executor.resource_policy.require_gpu is False

    store.create(
        "real-training",
        {**config("autonomous"), "enable_real_training": True},
    )
    real_executor_instance = task_executor(project_root, state_root, "real-training")
    assert isinstance(real_executor_instance.training_scheduler, TrainingScheduler)
    assert real_executor_instance.resource_policy.require_gpu is True


def test_training_scheduler_starts_stage_command(project_root, tmp_path):
    from pathmnist.workflow import TrainingScheduler

    commands = []

    def runner(command, log_path):
        commands.append((command, log_path))
        return CompletedProcess()

    scheduler = TrainingScheduler(runner=runner)
    log_path = tmp_path / "training.log"
    assert scheduler.start("train-task", "tuning_completed", project_root, log_path)
    assert commands[0][0] == [
        "python",
        "-m",
        "pathmnist",
        "train",
        "--phase",
        "tune",
        "--project-root",
        str(project_root),
        "--output-root",
        str(project_root / "runs" / "pathmnist-m4"),
    ]
    assert TrainingScheduler.wait(scheduler.processes[("train-task", "tuning_completed")]) == 0


def test_training_scheduler_failure_fails_stage(project_root, tmp_path):
    require_real_dataset(project_root)
    from pathmnist.workflow import TrainingScheduler

    executor = real_executor(project_root)
    executor.training_scheduler = TrainingScheduler(runner=lambda command, log_path: FailedProcess())
    store = WorkflowStore(tmp_path)
    store.create("train-failure", config("autonomous"))
    state = store.load("train-failure")
    state.completed_stage = "improvements_completed"
    store.save(state)
    with pytest.raises(WorkflowError, match="training failed with exit code 1"):
        advance(store, "train-failure", executor, tmp_path / "artifacts")


def test_frozen_artifact_workflow_skips_training_resource_limits(project_root):
    executor = real_executor(project_root)
    executor.resource_policy = type(executor.resource_policy)(
        require_gpu=True,
        require_ac_power=True,
        minimum_free_gb=10_000,
        minimum_memory_gb=10_000,
    )
    executor.training_scheduler = None

    resources = executor.check_resources("baseline_completed")

    assert resources == {
        "mode": "reuse_frozen_artifacts",
        "fresh_training": False,
        "resource_check_required": False,
    }


def test_training_resource_check_uses_project_mount(project_root, monkeypatch):
    from pathmnist.workflow import TrainingScheduler

    executor = real_executor(project_root)
    executor.training_scheduler = TrainingScheduler(
        runner=lambda command, log_path: CompletedProcess()
    )
    checked = []

    class Usage:
        free = 100 * 1024**3

    def fake_disk_usage(path):
        checked.append(path)
        return Usage()

    monkeypatch.setattr("shutil.disk_usage", fake_disk_usage)
    resources = executor.check_resources("baseline_completed")

    assert checked == [project_root.resolve()]
    assert resources["disk_path"] == str(project_root.resolve())


def test_paper_workflow_handlers_use_llm_usage(project_root, tmp_path):
    require_real_dataset(project_root)
    from pathmnist.workflow import WorkflowLLMClient

    class StubProvider:
        def __init__(self):
            self.calls = []

        def call_text(self, role, request_id, system, prompt):
            self.calls.append((role, request_id))
            return f"generated-{role}", {
                "request_id": request_id,
                "resolved_model": "stub-model",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "actual_cost_usd": 0.01,
            }

    provider = StubProvider()
    usage_path = tmp_path / "llm_usage.jsonl"
    executor = real_executor(project_root)
    executor.llm_client = WorkflowLLMClient(provider, usage_path)
    store = WorkflowStore(tmp_path)
    store.create("paper-task", config("autonomous"))
    state = run_stages(
        store, "paper-task", 24, project_root, tmp_path / "artifacts", executor
    )
    assert state.completed_stage == "archived"
    for stage, filename in {
        "english_paper_completed": "paper.json",
        "review_completed": "review.json",
        "revision_completed": "revision.json",
        "chinese_translation_completed": "translation.json",
    }.items():
        payload = json.loads(
            (tmp_path / "artifacts" / stage / filename).read_text(encoding="utf-8")
        )
        assert payload["llm_output"].startswith("generated-")
    revision_root = tmp_path / "artifacts" / "revision_completed"
    assert (revision_root / "final_paper.md").is_file()
    assert (revision_root / "final_paper.tex").is_file()
    assert (revision_root / "final_paper.md").read_text(encoding="utf-8").startswith("#")
    assert len(provider.calls) == 4
    assert WorkflowLLMClient.total_cost_usd(usage_path) == 0.04


def test_worker_recovery_rolls_back_invalid_completed_artifact(project_root, tmp_path):
    require_real_dataset(project_root)
    store = WorkflowStore(tmp_path)
    store.create("recovery-task", config("autonomous"))
    state = run_stages(store, "recovery-task", 2, project_root, tmp_path / "artifacts")
    assert state.completed_stage == "dataset_validated"
    artifact = tmp_path / "artifacts" / "dataset_validated" / "dataset.json"
    artifact.write_text("{}\n", encoding="utf-8")
    state = repair_to_valid_stage(state, store, tmp_path / "artifacts")
    assert state.completed_stage == "task_created"
    assert state.stages["dataset_validated"]["status"] == "waiting"

