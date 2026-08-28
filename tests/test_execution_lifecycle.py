import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from gate_a.streaming import run_streaming
from pathmnist.autonomous import AIScientistExperimentRunner
from pathmnist.execution_control import task_lock


def test_duplicate_start_refused_and_lock_released(tmp_path):
    with task_lock(tmp_path):
        with pytest.raises(RuntimeError, match="duplicate start"):
            with task_lock(tmp_path):
                pytest.fail("second owner admitted")
    with task_lock(tmp_path):
        pass


def test_stream_timeout_preserves_partial_epoch_output():
    lines = []
    with pytest.raises(subprocess.TimeoutExpired) as failure:
        run_streaming([sys.executable, "-u", "-c", "import time; print('epoch 1', flush=True); time.sleep(10)"],
                      timeout=1, env=os.environ.copy(), on_line=lambda channel, line: lines.append(line))
    assert "epoch 1" in "".join(lines)
    assert "epoch 1" in failure.value.output


def test_actual_agent_step_stops_on_worker_timeout_before_gpu_release(project_root, tmp_path):
    events = []

    class Runner:
        def cancel_active(self, root):
            events.append("sandbox_stopped")

    _, parallel, _, _ = AIScientistExperimentRunner(project_root, object(), Runner())._runtime_classes(tmp_path)
    agent = object.__new__(parallel)
    agent.cfg = OmegaConf.create({"agent": {"contract_metric": "accuracy"}, "workspace_dir": str(tmp_path)})
    agent.stage_name = "3_creative_research"
    agent.task_desc = "fixture"
    agent.timeout = 0.01
    agent.evaluation_metrics = agent._define_global_metrics()
    assert json.loads(agent.evaluation_metrics)[0]["name"] == "accuracy"
    agent._select_parallel_nodes = lambda: [None]
    agent.journal = SimpleNamespace(generate_summary=lambda **kwargs: "", nodes=[])
    agent.best_stage1_node = agent.best_stage2_node = agent.best_stage3_node = None

    class Future:
        def result(self, timeout):
            raise TimeoutError("still running")

        def done(self):
            return False

    class Process:
        alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            events.append("worker_stopped")
            self.alive = False

        def join(self, timeout):
            pass

    process = Process()
    agent.executor = SimpleNamespace(
        submit=lambda *args: Future(), _processes={1: process},
        shutdown=lambda **kwargs: events.append("shutdown"),
    )
    agent.gpu_manager = SimpleNamespace(
        acquire_gpu=lambda worker: 0, gpu_assignments={"worker_0": 0},
        release_gpu=lambda worker: events.append("gpu_released"),
    )
    agent._is_shutdown = False
    with pytest.raises(RuntimeError, match="Worker deadline exceeded"):
        agent.step(None)
    assert "gpu_released" not in events
    agent.cleanup()
    assert events.index("worker_stopped") < events.index("sandbox_stopped") < events.index("gpu_released")
