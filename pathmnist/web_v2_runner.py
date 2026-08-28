from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .autonomous_orchestrator import ResearchOrchestrator


RUN_FILE = "web_run.json"
LOG_FILE = "web_run.log"
LOCK_FILE = "web_run.lock"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_status(task_root: Path) -> dict[str, Any]:
    path = task_root / RUN_FILE
    if not path.is_file():
        return {"state": "not_started", "message": "尚未从网页启动完整实验"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state": "unknown", "message": "后台状态文件暂时不可读"}
    pid = value.get("pid")
    if value.get("state") == "running" and isinstance(pid, int):
        try:
            os.kill(pid, 0)
        except OSError:
            value["state"] = "interrupted"
            value["message"] = "网页容器曾中断；点击继续即可从已验证阶段恢复"
    try:
        progress = json.loads((task_root / "agent_progress.json").read_text(encoding="utf-8"))
        if isinstance(progress, dict):
            value["progress"] = progress
    except (OSError, ValueError):
        pass
    return value


def log_tail(task_root: Path, lines: int = 120) -> str:
    path = task_root / LOG_FILE
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def start(project_root: Path, state_root: Path, task_id: str) -> int:
    task_root = state_root.resolve() / task_id
    task_root.mkdir(parents=True, exist_ok=True)
    current = run_status(task_root)
    if current.get("state") == "running":
        raise RuntimeError("该任务已经在后台运行")
    if current.get("state") in {"failed", "interrupted"}:
        (task_root / LOCK_FILE).unlink(missing_ok=True)
    log_path = task_root / LOG_FILE
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "path_ai_scientist.ui.web_runner",
            "execute",
            "--project-root",
            str(project_root.resolve()),
            "--state-root",
            str(state_root.resolve()),
            "--task-id",
            task_id,
        ],
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    _write_json(
        task_root / RUN_FILE,
        {
            "schema_version": 1,
            "state": "running",
            "pid": process.pid,
            "started_at": _now(),
            "updated_at": _now(),
            "message": "后台实验已启动",
        },
    )
    return process.pid


def execute(project_root: Path, state_root: Path, task_id: str) -> None:
    task_root = state_root.resolve() / task_id
    status_path = task_root / RUN_FILE
    lock_path = task_root / LOCK_FILE
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("后台任务锁已存在；请确认没有另一个实验进程") from exc
    os.close(lock_fd)
    started = _now()
    try:
        _write_json(status_path, {
            "schema_version": 1, "state": "running", "pid": os.getpid(),
            "started_at": started, "updated_at": _now(), "message": "正在执行完整研究流程",
        })
        orchestrator = ResearchOrchestrator(project_root, state_root, task_id)
        while True:
            stage = orchestrator.status()["completed_stage"]
            print(f"[{_now()}] 当前阶段: {stage}", flush=True)
            if stage == "archived":
                _write_json(status_path, {
                    "schema_version": 1, "state": "completed", "pid": os.getpid(),
                    "started_at": started, "updated_at": _now(), "message": "实验、绘图和论文已经全部完成",
                })
                return
            if stage == "research_contract_generated":
                _write_json(status_path, {
                    "schema_version": 1, "state": "interrupted", "pid": os.getpid(),
                    "started_at": started, "updated_at": _now(),
                    "message": "研究合同已生成，等待用户确认后继续付费实验",
                })
                return
            if stage == "candidate_frozen":
                print(f"[{_now()}] 执行已授权的一次性 sealed test 审批", flush=True)
                orchestrator.approve_test()
                continue
            result = orchestrator.run(allow_paid=True, allow_test=True, allow_pdf=True)
            transitions = result.get("transitions", [])
            print(f"[{_now()}] 已完成 {len(transitions)} 个转换", flush=True)
    except Exception as exc:
        _write_json(status_path, {
            "schema_version": 1, "state": "failed", "pid": os.getpid(),
            "started_at": started, "updated_at": _now(), "message": str(exc),
        })
        print(f"[{_now()}] 运行失败: {exc}", flush=True)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("execute",))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    execute(args.project_root, args.state_root, args.task_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
