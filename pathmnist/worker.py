from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .workflow import (
    STAGES,
    WorkflowError,
    WorkflowStore,
    task_executor,
    advance,
    repair_to_valid_stage,
)


def acquire_lock(lock_path: Path) -> None:
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, json.dumps({"pid": os.getpid(), "started_at": time.time()}).encode())
        os.close(descriptor)
    except FileExistsError as exc:
        raise WorkflowError(f"Task worker is already active: {lock_path}") from exc


def lock_is_stale(lock_path: Path, pid_alive=None) -> bool:
    if not lock_path.is_file():
        return False
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return True
    if pid_alive is None:
        pid_alive = _host_pid_alive
    return not pid_alive(pid)


def _host_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def recoverable_tasks(state_root: Path, pid_alive=None) -> list[str]:
    store = WorkflowStore(state_root)
    task_ids: list[str] = []
    for path in store.root.glob("*.json"):
        task_id = path.stem
        state = store.load(task_id)
        if state.control != "running" or state.completed_stage == STAGES[-1]:
            continue
        lock_path = state_root / task_id / "worker.lock"
        if not lock_path.is_file() or lock_is_stale(lock_path, pid_alive):
            task_ids.append(task_id)
    return sorted(task_ids)


def supervisor(
    project_root: Path,
    poll_seconds: float = 1.0,
    state_root: Path | None = None,
    worker_factory=None,
) -> None:
    store = WorkflowStore(state_root or project_root / "state" / "workflow")
    state_root = store.root
    worker_factory = worker_factory or run
    while True:
        recoverable = recoverable_tasks(state_root)
        for task_id in recoverable:
            try:
                worker_factory(task_id, project_root, poll_seconds)
            except WorkflowError:
                pass
        if not recoverable:
            time.sleep(poll_seconds)
        else:
            return


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def run(
    task_id: str,
    project_root: Path,
    poll_seconds: float = 1.0,
    executor_factory=None,
    state_root: Path | None = None,
) -> str:
    store = WorkflowStore(state_root or project_root / "state" / "workflow")
    state_root = store.root
    artifact_root = state_root / task_id / "artifacts"
    lock_path = state_root / task_id / "worker.lock"
    acquire_lock(lock_path)
    try:
        executor_factory = executor_factory or task_executor
        executor = executor_factory(project_root, state_root, task_id)
        state = store.load(task_id)
        state = repair_to_valid_stage(state, store, artifact_root)
        while True:
            state = store.load(task_id)
            if state.control in {"paused", "cancelled", "waiting_approval"}:
                time.sleep(poll_seconds)
                continue
            if state.completed_stage == STAGES[-1]:
                return state.completed_stage
            try:
                state = advance(store, task_id, executor, artifact_root)
            except WorkflowError:
                state = store.load(task_id)
                exhausted = any(
                    item["status"] == "interrupted"
                    and int(item.get("retries", 0)) >= 3
                    for item in state.stages.values()
                )
                _log(state_root / task_id / "worker.log", state)
                if exhausted:
                    raise
                time.sleep(poll_seconds)
            else:
                _log(state_root / task_id / "worker.log", state)
    finally:
        release_lock(lock_path)


def _log(path: Path, state) -> None:
    payload = {
        "timestamp": time.time(),
        "completed_stage": state.completed_stage,
        "control": state.control,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run persistent PathMNIST workflow workers")
    parser.add_argument("task_id", nargs="?")
    parser.add_argument("--supervisor", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.supervisor:
        supervisor(args.project_root.resolve(), args.poll_seconds)
        return 0
    if not args.task_id:
        parser.error("task_id is required unless --supervisor is used")
    run(args.task_id, args.project_root.resolve(), args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
