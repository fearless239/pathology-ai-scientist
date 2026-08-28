import subprocess

import pytest

from gate_a.config import RunnerConfig
from gate_a.policy import CodePolicyError, validate_generated_code
from gate_a.runner import DockerRunner, _redact


def test_policy_accepts_offline_numeric_code():
    validate_generated_code("import os\nimport numpy as np\nprint(np.arange(3))\n")


def test_redact_decodes_timeout_bytes_and_hides_key():
    assert _redact(b"failure sk-or-v1-secret") == "failure [REDACTED_OPENROUTER_KEY]"


@pytest.mark.parametrize(
    "code",
    [
        "import requests\n",
        "import subprocess\n",
        "from socket import socket\n",
        "eval('1 + 1')\n",
        "import os\nos.system('whoami')\n",
        "print('OPENROUTER_API_KEY')\n",
    ],
)
def test_policy_rejects_network_process_and_secret_code(code):
    with pytest.raises(CodePolicyError):
        validate_generated_code(code)


def test_runner_command_contains_isolation_guards(tmp_path, monkeypatch):
    captured = {}

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Result()

    monkeypatch.setattr("gate_a.runner.subprocess.run", fake_run)
    config = RunnerConfig(
        image="runner:test",
        timeout_seconds=30,
        cpus=1.0,
        memory="1g",
        pids_limit=32,
        docker_command=("docker",),
    )
    result = DockerRunner(config).run_python("print('ok')", tmp_path / "node")
    assert result.succeeded
    command = captured["command"]
    assert ["--network", "none"] == command[
        command.index("--network") : command.index("--network") + 2
    ]
    assert "--read-only" in command
    assert "65532:65532" in command
    assert "ALL" in command
    assert "no-new-privileges:true" in command
    assert "--shm-size" in command
    env_pairs = list(zip(command, command[1:]))
    assert ("--env", "PYTHONUNBUFFERED=1") in env_pairs
    assert captured["kwargs"]["env"] == {"PATH": captured["kwargs"]["env"]["PATH"]}


def test_research_dataset_is_mounted_read_only_without_test_split(tmp_path, monkeypatch):
    captured = {}

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    # The staged mount deliberately contains only train and validation. The sealed
    # test path is never passed to Docker, so generated code cannot discover it.
    staged = tmp_path / "research_dataset"
    (staged / "train").mkdir(parents=True)
    (staged / "validation").mkdir()
    config = RunnerConfig("runner:test", 30, 1.0, "1g", 32, ("docker",))

    # Use an explicit fake instead of the compact lambda above so subprocess.run's
    # return contract remains visible to the runner.
    def fake_run(command, **kwargs):
        captured["command"] = command
        return Result()

    monkeypatch.setattr("gate_a.runner.subprocess.run", fake_run)
    DockerRunner(config).run_python("print('ok')", tmp_path / "node", dataset_mount=staged)
    command = captured["command"]
    dataset_arg = next(arg for arg in command if "dst=/dataset" in arg)
    assert dataset_arg.endswith(",readonly")
    assert {path.name for path in staged.iterdir()} == {"train", "validation"}
    assert sum("dst=/dataset" in arg for arg in command) == 1


def test_gpu_flag_is_opt_in(tmp_path, monkeypatch):
    captured = {}

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        return Result()

    monkeypatch.setattr("gate_a.runner.subprocess.run", fake_run)
    config = RunnerConfig("runner:test", 30, 1.0, "1g", 32, ("docker",))
    DockerRunner(config, gpus="all", shm_size="2g").run_python("print('ok')", tmp_path / "node")
    command = captured["command"]
    assert command[command.index("--gpus") + 1] == "all"
    assert command[command.index("--shm-size") + 1] == "2g"


def test_timeout_force_removes_named_container(tmp_path, monkeypatch):
    calls = []

    class CleanupResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, 1, output="partial", stderr="timeout")
        return CleanupResult()

    monkeypatch.setattr("gate_a.runner.subprocess.run", fake_run)
    config = RunnerConfig("runner:test", 1, 1.0, "1g", 32, ("docker",))
    result = DockerRunner(config).run_python("print('ok')", tmp_path / "node")
    container_name = calls[0][calls[0].index("--name") + 1]

    assert result.timed_out
    assert calls[1] == ["docker", "rm", "--force", container_name]



def _run_launcher(tmp_path, code):
    import sys
    from gate_a.import_preflight import sandbox_launcher

    script = tmp_path / "experiment.py"
    script.write_text(code, encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-c", sandbox_launcher(code, str(script))],
        cwd=tmp_path, text=True, capture_output=True, timeout=30,
    )


def test_import_preflight_rejects_typo_before_experiment_side_effects(tmp_path):
    (tmp_path / "example_models.py").write_text("Correct_Model_Weights = object()\n")
    result = _run_launcher(tmp_path, """from pathlib import Path
Path('training_started').touch()
from example_models import CorrectModel_Weights
""")
    assert result.returncode == 86
    assert "IMPORT_PREFLIGHT_FAILED" in result.stderr
    assert "Correct_Model_Weights" in result.stderr
    assert not (tmp_path / "training_started").exists()


def test_import_preflight_preserves_future_and_optional_imports(tmp_path):
    result = _run_launcher(tmp_path, """from __future__ import annotations
from pathlib import Path
try:
    import nonexistent_optional_package
except ImportError:
    pass
if False:
    from math import nonexistent_symbol
def unused():
    import nonexistent_lazy_package
def identity(x: NotDefined):
    return x
if __name__ == '__main__':
    Path('executed').write_text(str(identity(3)))
""")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "executed").read_text() == "3"


def test_import_preflight_accepts_submodule_and_alias_imports(tmp_path):
    result = _run_launcher(tmp_path, """from xml import etree
import json as js
from math import sqrt as root
print(js.dumps({'root': root(9)}))
""")
    assert result.returncode == 0, result.stderr
    assert '"root": 3.0' in result.stdout


def test_import_preflight_reports_missing_package(tmp_path):
    result = _run_launcher(tmp_path, "import nonexistent_required_package_123\n")
    assert result.returncode == 86
    assert "nonexistent_required_package_123" in result.stderr


def test_docker_runner_invokes_preflight_without_changing_generated_artifact(tmp_path, monkeypatch):
    from gate_a.runner import ExecutionResult

    captured = {}
    def fake_container(self, workspace, inner_command, timeout, dataset_mount=None):
        captured['command'] = inner_command
        return ExecutionResult(tuple(inner_command), 0, '', '', 0.0, False)
    monkeypatch.setattr(DockerRunner, '_run_container', fake_container)
    code = 'from math import sqrt\nprint(sqrt(9))\n'
    config = RunnerConfig("runner:test", 30, 1.0, "1g", 32, ("docker",))
    DockerRunner(config).run_python(code, tmp_path)
    assert captured['command'][:2] == ['python', '-c']
    assert 'IMPORT_PREFLIGHT_FAILED' in captured['command'][2]
    assert (tmp_path / 'runfile.py').read_text() == code



@pytest.mark.parametrize("loader", [
    "import numpy as np\ndata = np.load('/dataset/dataset.npz')",
    "from numpy import load\ndata = load('/dataset/dataset.npz')",
])
def test_dataset_access_guard_rejects_source_alias_before_container(tmp_path, monkeypatch, loader):
    import numpy as np
    np.savez(tmp_path / "dataset.npz", validation_images=np.zeros((2, 4, 4, 3)))
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid access must be rejected before launching Docker")
    monkeypatch.setattr(DockerRunner, '_run_container', forbidden)
    config = RunnerConfig("runner:test", 30, 1.0, "1g", 32, ("docker",))
    with pytest.raises(CodePolicyError, match="DATASET_INTERFACE_FAILED.*val_images"):
        DockerRunner(config).run_python(loader + '\nx = data["val_images"]', tmp_path / 'node', dataset_mount=tmp_path)


def test_dataset_access_guard_preserves_inference_branches_and_other_mappings(tmp_path):
    import numpy as np
    from gate_a.import_preflight import validate_dataset_access
    np.savez(tmp_path / "dataset.npz", validation_images=np.zeros((2, 4, 4, 3)))
    validate_dataset_access("""import numpy as np
data = np.load('/dataset/dataset.npz')
if 'train_images' in data.files:
    train = data['train_images']
validation = data['validation_images']
optional = data['train_images'] if 'train_images' in data.files else None
lazy = lambda: data['train_images']
short_circuit = False and data['train_images']
data = {'custom': 1}
value = data['custom']
""", tmp_path)


def test_dataset_access_guard_checks_context_manager(tmp_path):
    import numpy as np
    from gate_a.import_preflight import validate_dataset_access
    np.savez(tmp_path / "dataset.npz", validation_images=np.zeros((2, 4, 4, 3)))
    with pytest.raises(CodePolicyError, match="DATASET_INTERFACE_FAILED"):
        validate_dataset_access("""import numpy as np
with np.load('/dataset/dataset.npz') as data:
    images = data['val_images']
""", tmp_path)


@pytest.mark.parametrize("selection", [
    'key = "val_images" if "val_images" in data.files else None',
    'key = "val_images"',
])
def test_dataset_guard_resolves_indirect_keys_and_path_variables(tmp_path, selection):
    import numpy as np
    from gate_a.import_preflight import validate_dataset_access
    np.savez(tmp_path / 'dataset.npz', validation_images=np.zeros((2, 4, 4, 3)))
    code = '''import numpy as np
PATH = '/dataset/dataset.npz'
data = np.load(PATH)
''' + selection + '''
if key is None:
    print('ERROR: Validation data not found')
    raise SystemExit(1)
images = data[key]
'''
    with pytest.raises(CodePolicyError, match='DATASET_INTERFACE_FAILED'):
        validate_dataset_access(code, tmp_path)


def test_dataset_guard_accepts_correct_fallback_and_unknown_branch(tmp_path):
    import numpy as np
    from gate_a.import_preflight import validate_dataset_access
    np.savez(tmp_path / 'dataset.npz', validation_images=np.zeros((2, 4, 4, 3)))
    validate_dataset_access('''import numpy as np
data = np.load('/dataset/dataset.npz')
key = 'val_images' if 'val_images' in data.files else 'validation_images'
images = data[key]
key = 'val_images'
if some_runtime_condition():
    key = 'validation_images'
images = data[key]
''', tmp_path)


def test_model_requirements_follow_approved_task_for_training_and_inference(tmp_path):
    import json
    import numpy as np
    from gate_a.model_contract import requirements_for_dataset
    from pathmnist.research_contract import generate_contract, write_contract, approve_contract
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'task.json').write_text(json.dumps({'schema_version': 2, 'completed_stage': 'research_contract_generated', 'stages': {}}))
    contract = generate_contract('使用28×28图像训练CNN', {'classes': ['0', '1']})
    write_contract(task, contract)
    approve_contract(task)
    view = task / 'dataset/research_view'
    view.mkdir(parents=True)
    np.savez(view / 'dataset.npz', train_images=np.zeros((1,64,64,3)), validation_images=np.zeros((1,64,64,3)))
    assert requirements_for_dataset(view) == {'input_sizes': [[28,28]], 'inference_only': False}
    sealed = task / 'final_evaluation/validation_only'
    sealed.mkdir(parents=True)
    np.savez(sealed / 'dataset.npz', validation_images=np.zeros((1,64,64,3)))
    assert requirements_for_dataset(sealed) == {'input_sizes': [[28,28]], 'inference_only': True}


def test_environment_preflight_can_skip_model_execution_checks(tmp_path, monkeypatch):
    from gate_a.runner import ExecutionResult
    def unexpected(*args):
        pytest.fail('Environment-only preflight must not request a model forward')
    monkeypatch.setattr('gate_a.runner.requirements_for_dataset', unexpected)
    monkeypatch.setattr(DockerRunner, '_run_container', lambda *a, **kw: ExecutionResult((),0,'','',0,False))
    config = RunnerConfig('runner:test',30,1.0,'1g',32,('docker',))
    assert DockerRunner(config).run_python('print(1)',tmp_path,enforce_model_contract=False).succeeded


@pytest.mark.parametrize('required', [True, False])
def test_dynamic_smoothing_runtime_requirement_reaches_launcher(tmp_path, monkeypatch, required):
    from gate_a.runner import ExecutionResult
    captured = {}
    def launcher(code, path, requirements):
        captured.update(requirements)
        return 'offline-test'
    monkeypatch.setattr('gate_a.runner.sandbox_launcher', launcher)
    monkeypatch.setattr(DockerRunner, '_run_container', lambda *a, **kw: ExecutionResult((),0,'','',0,False))
    config = RunnerConfig('runner:test',30,1.0,'1g',32,('docker',))
    code = 'def train(amount):\n    return nn.CrossEntropyLoss(label_smoothing=amount)\ntrain(0.1)\n'
    assert DockerRunner(config).run_python(code,tmp_path,require_standard_smoothing=required).succeeded
    assert captured.get('standard_smoothing_required', False) is required
