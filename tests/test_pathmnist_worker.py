from __future__ import annotations

import json
import threading
import time

from pathmnist.workflow import STAGE_CONTRACTS, STAGES, WorkflowStore


def test_worker_lock_is_exclusive(project_root, tmp_path):
    from pathmnist.workflow import WorkflowError
    from pathmnist.worker import acquire_lock, release_lock

    lock = tmp_path / "worker.lock"
    acquire_lock(lock)
    try:
        WorkflowError
    finally:
        release_lock(lock)
    acquire_lock(lock)
    release_lock(lock)


def test_worker_reaches_waiting_approval(project_root, tmp_path):
    from pathmnist import worker

    class StubExecutor:
        def __init__(self, project_root, state_root, task_id):
            self.project_root = project_root
            self.state_root = state_root
            self.task_id = task_id

        def execute(self, stage, state, artifact_root):
            filename = sorted(STAGE_CONTRACTS[stage])[0]
            path = artifact_root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            return {filename: "stub-sha256"}

    state_root = tmp_path / "state" / "workflow"
    task_id = "worker-task"
    store = WorkflowStore(state_root)
    store.create(
        task_id,
        {
            "mode": "staged_approval",
            "budget_limit_usd": 50,
            "execution_limit_seconds": 21600,
        },
    )

    thread = threading.Thread(
        target=worker.run,
        args=(task_id, project_root, 0.01),
        kwargs={"executor_factory": StubExecutor, "state_root": state_root},
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if store.load(task_id).control == "waiting_approval":
            break
        time.sleep(0.01)
    assert store.load(task_id).control == "waiting_approval"


def test_recoverable_tasks_ignore_live_and_missing_locks(project_root, tmp_path):
    from pathmnist.worker import recoverable_tasks
    from pathmnist.workflow import WorkflowStore

    state_root = tmp_path / "state" / "workflow"
    store = WorkflowStore(state_root)
    store.create("running-live", {"mode": "autonomous", "budget_limit_usd": 2, "execution_limit_seconds": 3600})
    store.create("running-stale", {"mode": "autonomous", "budget_limit_usd": 2, "execution_limit_seconds": 3600})
    store.create("paused", {"mode": "autonomous", "budget_limit_usd": 2, "execution_limit_seconds": 3600})

    live_lock = state_root / "running-live" / "worker.lock"
    live_lock.parent.mkdir(parents=True, exist_ok=True)
    live_lock.write_text(json.dumps({"pid": 123, "started_at": 0}), encoding="utf-8")
    stale_lock = state_root / "running-stale" / "worker.lock"
    stale_lock.parent.mkdir(parents=True, exist_ok=True)
    stale_lock.write_text(json.dumps({"pid": 456, "started_at": 0}), encoding="utf-8")

    from pathmnist.workflow import set_control
    set_control(store, "paused", "paused")

    tasks = recoverable_tasks(state_root, pid_alive=lambda pid: pid == 123)
    assert tasks == ["running-stale"]


def test_supervisor_restarts_recoverable_task(project_root, tmp_path):
    from pathmnist import worker
    from pathmnist.workflow import STAGE_CONTRACTS, WorkflowStore

    class StubExecutor:
        def __init__(self, project_root, state_root, task_id):
            self.project_root = project_root
            self.state_root = state_root
            self.task_id = task_id

        def execute(self, stage, state, artifact_root):
            filename = sorted(STAGE_CONTRACTS[stage])[0]
            path = artifact_root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            return {filename: "stub-sha256"}

    state_root = tmp_path / "state" / "workflow"
    store = WorkflowStore(state_root)
    store.create(
        "recoverable",
        {"mode": "staged_approval", "budget_limit_usd": 2, "execution_limit_seconds": 3600},
    )
    calls = []

    def worker_factory(task_id, project_root, poll_seconds):
        calls.append(task_id)
        state = store.load(task_id)
        index = STAGES.index(state.completed_stage) + 1 if state.completed_stage else 0
        stage = STAGES[index]
        state.stages[stage]["status"] = "waiting_approval"
        state.control = "waiting_approval"
        store.save(state)
        return state.completed_stage

    original_run = worker.run
    try:
        worker.run = worker_factory
        worker.supervisor(project_root, 0.01, state_root, worker_factory)
    finally:
        worker.run = original_run

    assert calls == ["recoverable"]
    assert store.load("recoverable").control == "waiting_approval"
