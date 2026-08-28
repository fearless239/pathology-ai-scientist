

def test_tasks_are_listed_newest_first(tmp_path):
    from pathmnist.workflow import WorkflowStore

    store = WorkflowStore(tmp_path / "state" / "workflow")
    config = {
        "mode": "autonomous",
        "budget_limit_usd": 2,
        "execution_limit_seconds": 3600,
    }
    first = store.create("first", config)
    second = store.create("second", config)
    first.created_at = "2026-08-19T00:00:00+00:00"
    second.created_at = "2026-08-20T00:00:00+00:00"
    store.save(first)
    store.save(second)

    assert [state.task_id for state in store.list_states()] == ["second", "first"]


def test_task_delete_removes_state_and_artifacts(tmp_path):
    from pathmnist.workflow import WorkflowStore

    store = WorkflowStore(tmp_path / "state" / "workflow")
    store.create(
        "old-task",
        {
            "mode": "autonomous",
            "budget_limit_usd": 2,
            "execution_limit_seconds": 3600,
        },
    )
    artifact = store.root / "old-task" / "artifacts" / "paper.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("paper", encoding="utf-8")

    store.delete("old-task")

    assert not store.path("old-task").exists()
    assert not (store.root / "old-task").exists()


def test_workflow_dashboard_creates_persistent_task(project_root, tmp_path):
    from app import workflow_root
    from pathmnist.workflow import WorkflowStore

    root = tmp_path / "state" / "workflow"
    store = WorkflowStore(root)
    store.create("ui-task", {"mode": "staged_approval", "budget_limit_usd": 2, "execution_limit_seconds": 3600})
    assert (root / "ui-task.json").is_file()
    assert (root / "ui-task" / "budget.json").is_file()
    assert store.load("ui-task").completed_stage == ""
    assert workflow_root(project_root).name == "workflow"


def test_task_executor_uses_task_budget(project_root, tmp_path):
    from app import workflow_root
    from pathmnist.workflow import WorkflowStore, task_budget_path, task_executor

    root = tmp_path / "state" / "workflow"
    WorkflowStore(root).create(
        "ui-budget-task",
        {"mode": "staged_approval", "budget_limit_usd": 2, "execution_limit_seconds": 3600},
    )
    executor = task_executor(project_root, root, "ui-budget-task")
    assert executor.cost_ledger.ledger.hard_limit_usd == 2.0
    assert task_budget_path(root, "ui-budget-task").is_file()
    assert workflow_root(project_root).name == "workflow"
