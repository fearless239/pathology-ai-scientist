from __future__ import annotations

import os
import re
import subprocess
import time
import uuid
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .import_preflight import sandbox_launcher, validate_dataset_access
from .model_contract import requirements_for_dataset
from .config import RunnerConfig
from .policy import require_relative_artifact, validate_generated_code


class RunnerError(RuntimeError):
    """Raised when the isolated experiment runner cannot complete safely."""


class SandboxCleanupError(RunnerError):
    """Cleanup could not be confirmed; never turn this into a retryable node."""


SECRET_PATTERN = re.compile(r"sk-or-v1-[A-Za-z0-9_-]+")


def _redact(value: str | bytes | None) -> str:
    """Redact provider secrets from normal and timeout subprocess output."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return SECRET_PATTERN.sub("[REDACTED_OPENROUTER_KEY]", value)


@dataclass(frozen=True)
class ExecutionResult:
    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class DockerRunner:
    """Executes generated code with no network, no capabilities, and one writable mount."""

    def __init__(self, config: RunnerConfig, gpus: str | None = None, shm_size: str = "512m", stream_output: bool = False):
        self.config = config
        self.gpus = gpus
        self.shm_size = shm_size
        self.stream_output = stream_output

    def cancel_active(self, workspace_root: Path):
        for receipt in (workspace_root / ".active-sandboxes").glob("*.json"):
            if receipt.name.endswith(".progress.json"):
                continue
            name = json.loads(receipt.read_text())["name"]
            if not re.fullmatch(r"path-scientist-run-[0-9a-f]{32}", name):
                raise RunnerError("Invalid active sandbox identity")
            result = subprocess.run([*self.config.docker_command, "rm", "--force", name],
                                    capture_output=True, text=True, timeout=30)
            if result.returncode != 0 and "No such container" not in result.stderr:
                raise SandboxCleanupError("Cannot confirm sandbox cleanup; stop scheduling")
            receipt.unlink(missing_ok=True)

    def run_python(
        self, code: str, workspace: Path, file_name: str = "runfile.py",
        dataset_mount: Path | None = None,
        enforce_model_contract: bool = True,
        require_standard_smoothing: bool = False,
    ) -> ExecutionResult:
        validate_generated_code(code)
        validate_dataset_access(code, dataset_mount)
        try:
            model_requirements = requirements_for_dataset(dataset_mount) if enforce_model_contract else {}
            if enforce_model_contract:
                from pathmnist.method_spec import custom_smoothing_classes
                custom = custom_smoothing_classes(code)
                if custom:
                    model_requirements['custom_smoothing_classes'] = custom
                if require_standard_smoothing:
                    model_requirements['standard_smoothing_required'] = True
        except ValueError as error:
            raise RunnerError(f'Model contract invalid: {error}') from error
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        working = workspace / "working"
        working.mkdir(parents=True, exist_ok=True)
        # The orchestrator may run as root, but the experiment always runs as numeric non-root.
        os.chmod(workspace, 0o777)
        os.chmod(working, 0o777)
        script = require_relative_artifact(workspace / file_name, workspace)
        script.write_text(code, encoding="utf-8")
        os.chmod(script, 0o644)
        return self._run_container(
            workspace,
            ["python", "-c", sandbox_launcher(code, f"/workspace/{file_name}", model_requirements)],
            timeout=self.config.timeout_seconds,
            dataset_mount=dataset_mount,
        )

    def compile_latex(
        self, paper_dir: Path, tex_name: str = "paper.tex"
    ) -> ExecutionResult:
        paper_dir = paper_dir.resolve()
        require_relative_artifact(paper_dir / tex_name, paper_dir)
        os.chmod(paper_dir, 0o777)
        command = [
            "sh",
            "-lc",
            f"pdflatex -interaction=nonstopmode -halt-on-error {tex_name} && "
            f"pdflatex -interaction=nonstopmode -halt-on-error {tex_name}",
        ]
        return self._run_container(
            paper_dir, command, timeout=self.config.timeout_seconds
        )

    def inspect_image_id(self) -> str:
        command = [
            *self.config.docker_command,
            "image",
            "inspect",
            self.config.image,
            "--format",
            "{{.Id}}",
        ]
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=30, check=False
        )
        if result.returncode != 0:
            raise RunnerError(
                f"Runner image is unavailable: {_redact(result.stderr.strip())}"
            )
        return result.stdout.strip()

    def _run_container(
        self, workspace: Path, inner_command: Sequence[str], timeout: int,
        dataset_mount: Path | None = None,
    ) -> ExecutionResult:
        mounts = ["--mount", f"type=bind,src={workspace},dst=/workspace"]
        if dataset_mount is not None:
            dataset_mount = dataset_mount.resolve()
            if not dataset_mount.exists():
                raise RunnerError(f"Dataset mount is absent: {dataset_mount}")
            mounts += ["--mount", f"type=bind,src={dataset_mount},dst=/dataset,readonly"]
        accelerator = ["--gpus", self.gpus] if self.gpus else []
        container_name = f"path-scientist-run-{uuid.uuid4().hex}"
        command = [
            *self.config.docker_command,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--user",
            "65532:65532",
            "--cpus",
            str(self.config.cpus),
            "--memory",
            self.config.memory,
            "--shm-size",
            self.shm_size,
            "--pids-limit",
            str(self.config.pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=128m",
            "--env",
            "HOME=/tmp",
            "--env",
            "MPLCONFIGDIR=/tmp/matplotlib",
            "--env",
            "PYTHONUNBUFFERED=1",
            *accelerator,
            *mounts,
            "--workdir",
            "/workspace",
            self.config.image,
            *inner_command,
        ]
        started = time.monotonic()
        registry = workspace.parent / ".active-sandboxes"
        registry.mkdir(parents=True, exist_ok=True)
        receipt = registry / f"{container_name}.json"
        receipt.write_text(json.dumps({"name": container_name}), encoding="utf-8")
        log_path = registry / f"{container_name}.log"
        log_lock = threading.Lock()
        def on_line(channel, line):
            line = _redact(line)
            with log_lock:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"[{time.monotonic() - started:.1f}s {channel}] {line}")
                progress = registry / f"{container_name}.progress.json"
                temporary = progress.with_suffix(".tmp")
                temporary.write_text(json.dumps({"elapsed_seconds": time.monotonic() - started,
                                                  "last_output_at": time.time(), "last_line": line}), encoding="utf-8")
                temporary.replace(progress)
                print(line, end="", flush=True)
        try:
            if self.stream_output:
                from .streaming import run_streaming
                result = run_streaming(command, timeout=timeout,
                                       env={"PATH": os.environ.get("PATH", "")}, on_line=on_line)
            else:
                result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
                env={"PATH": os.environ.get("PATH", "")},
                )
            receipt.unlink(missing_ok=True)
            return ExecutionResult(
                command=tuple(command),
                exit_code=result.returncode,
                stdout=_redact(result.stdout),
                stderr=_redact(result.stderr),
                elapsed_seconds=time.monotonic() - started,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            # ``docker run`` being killed does not reliably stop the container it
            # created. Remove the precisely named sandbox so timed-out GPU jobs do
            # not continue in the background or contend with a repair attempt.
            cleanup = subprocess.run(
                [*self.config.docker_command, "rm", "--force", container_name],
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
                env={"PATH": os.environ.get("PATH", "")},
            )
            if cleanup.returncode != 0 and "No such container" not in cleanup.stderr:
                raise SandboxCleanupError("Timed-out sandbox cleanup failed; execution must stop")
            receipt.unlink(missing_ok=True)
            return ExecutionResult(
                command=tuple(command),
                exit_code=124,
                stdout=_redact(exc.stdout or ""),
                stderr=_redact(exc.stderr or ""),
                elapsed_seconds=time.monotonic() - started,
                timed_out=True,
            )
