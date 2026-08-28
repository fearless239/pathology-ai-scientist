import json

import pytest

from gate_a.runner import ExecutionResult

from pathmnist.autonomous_preflight import _load_spec
from pathmnist.autonomous import (
    AIScientistExperimentRunner,
    EXECUTION_POLICY_REVISION,
    _grant_policy_repair_window,
    has_valid_generated_node,
    _allowed_contract_roles,
    _validate_experiment_execution_budget,
    _preserve_host_injected_seed,
    _validate_stage_semantics,
    _validate_tuning_execution_budget,
)
from pathmnist.scientific_integrity import IntegrityError


def test_load_profile_ignores_derived_normalized_path(tmp_path):
    profile = {
        "schema_version": 2, "name": "x", "source_type": "npz", "source_path": "/x.npz",
        "normalized_path": "/x.npz", "content_sha256": "a" * 64, "image_shape": [4, 4, 3],
        "channels": 3, "classes": ["a", "b"], "label_mapping": {"a": 0, "b": 1},
        "split_counts": {"train": 1, "validation": 1, "test": 1}, "class_counts": {},
        "samples": [{"id": "1", "path": "/x.npz", "label": "a", "split": "train", "group_id": None, "array_key": "train_images", "index": 0}],
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    assert _load_spec(path).name == "x"


def test_runtime_class_loading_prepares_vendor_import_path(project_root, tmp_path):
    from gate_a.config import RunnerConfig
    from gate_a.runner import DockerRunner

    runner = AIScientistExperimentRunner(
        project_root,
        object(),
        DockerRunner(RunnerConfig("runner:test", 10, 1.0, "1g", 16, ("docker",))),
    )
    manager, parallel, minimal, interpreter = runner._runtime_classes(tmp_path)
    assert manager.__name__ == "PathologyAgentManager"
    assert parallel.__name__ == "PathologyParallelAgent"
    assert minimal.__name__ == "PathologyMinimalAgent"
    assert interpreter.__name__ == "DockerInterpreter"

    node = type("Node", (), {})()
    minimal._analyze_plots_with_vlm(object.__new__(minimal), node)
    assert node.is_buggy_plots is False
    assert node.plot_analyses == []
    assert "skipped" in node.vlm_feedback_summary
    tuning_prompt = minimal._prompt_hyperparam_tuning_resp_fmt.__get__(
        object.__new__(minimal), minimal
    )
    recovery_contract = tuning_prompt["Tuning execution and recovery contract"]
    assert "exactly two candidate" in recovery_contract
    assert "tuning_progress.json" in recovery_contract


def test_invalid_generated_manifest_is_a_repairable_node_error(project_root, tmp_path):
    class FakeDockerRunner:
        def run_python(self, code, workspace, file_name, dataset_mount=None):
            working = workspace / "working"
            working.mkdir(parents=True, exist_ok=True)
            (working / "experiment_result.json").write_text(
                json.dumps({"metrics": {"validation_accuracy": 0.5}}), encoding="utf-8"
            )
            (working / "experiment_manifest.json").write_text(
                json.dumps({"schema_version": 1, "dataset": "x"}), encoding="utf-8"
            )
            return ExecutionResult(("python",), 0, "", "", 0.1, False)

    runner = AIScientistExperimentRunner(project_root, object(), FakeDockerRunner())
    *_, interpreter = runner._runtime_classes(tmp_path / "dataset")
    result = interpreter(tmp_path / "process").run("print('experiment')")

    assert result.exc_type == "ExperimentContractError"
    assert "missing required fields" in "".join(result.term_out)


def test_stage_semantics_rejects_intervention_in_baseline():
    code = """
import torch
from torchvision.transforms import ColorJitter, RandomRotation, RandomHorizontalFlip
augment = [ColorJitter(), RandomRotation(20), RandomHorizontalFlip()]
loss = torch.tensor(1.0, requires_grad=True)
loss.backward()
"""
    with pytest.raises(IntegrityError, match="Baseline stage implements"):
        _validate_stage_semantics(
            code,
            "1_initial_implementation_1_preliminary",
            ["color_jitter", "rotation", "flip"],
        )


def test_stage_semantics_allows_generic_baseline_preprocessing():
    code = """
import torch
from torchvision import transforms
normalize = transforms.Normalize((0.5,), (0.5,))
loss = torch.tensor(1.0, requires_grad=True)
loss.backward()
"""
    _validate_stage_semantics(
        code,
        "1_initial_implementation_1_preliminary",
        ["transform", "color_jitter", "rotation", "flip"],
    )


def test_stage_semantics_requires_all_interventions_in_proposed_method():
    code = """
import torch
from torchvision.transforms import ColorJitter
loss = torch.tensor(1.0, requires_grad=True)
loss.backward()
"""
    with pytest.raises(IntegrityError, match="every approved intervention"):
        _validate_stage_semantics(
            code,
            "3_creative_research_1_preliminary",
            ["color_jitter", "rotation", "flip"],
        )


def test_seed_normalizer_keeps_host_seed_and_removes_generated_override():
    code = """# Set random seed
seed = 2
import random
seed = 42
random.seed(seed)
"""
    normalized = _preserve_host_injected_seed(code)
    assert "seed = 2" in normalized
    assert "seed = 42" not in normalized


@pytest.mark.parametrize('body', [
    'if "seed" not in globals():\n    seed = 42\n',
    'if False:\n    seed = 42\nelse:\n    seed = 7\n',
    'try:\n    seed = 42\nfinally:\n    seed = 7\n',
    'if True: seed = 42\n',
])
def test_seed_normalizer_preserves_nonempty_suites(body):
    source = '# Set random seed\nseed = 2\n' + body + '\nresult = seed\n'
    normalized = _preserve_host_injected_seed(source)
    namespace = {}
    exec(compile(normalized, '<test>', 'exec'), namespace)
    assert namespace['result'] == 2
    assert _preserve_host_injected_seed(normalized) == normalized


def test_seed_normalizer_does_not_rewrite_local_scopes_or_comments():
    source = '''# Set random seed
seed = 2
# PATH_AI_METHOD_SPEC: preserved
def helper():
    seed = 99
    return seed
class Config:
    seed = 7
seed = 42
'''
    normalized = _preserve_host_injected_seed(source)
    namespace = {}
    exec(normalized, namespace)
    assert namespace['seed'] == 2
    assert namespace['helper']() == 99
    assert namespace['Config'].seed == 7
    assert '# PATH_AI_METHOD_SPEC: preserved' in normalized


def test_malformed_generated_code_is_a_node_error_not_worker_crash(project_root, tmp_path):
    class NoExecution:
        def run_python(self, *args, **kwargs):
            pytest.fail('Malformed code must never reach Docker')
    runner = AIScientistExperimentRunner(project_root, object(), NoExecution())
    *_, interpreter = runner._runtime_classes(tmp_path / 'dataset')
    result = interpreter(tmp_path / 'process').run('if True:\nprint(1)')
    assert result.exc_type == 'CodePolicyError'


def test_tuning_stage_accepts_explicit_role_alias():
    assert _allowed_contract_roles("2_baseline_tuning_1_first_attempt") == {
        "baseline",
        "hyperparameter_tuning",
    }


def test_tuning_budget_rejects_too_many_candidates():
    code = "learning_rates = [1e-4, 5e-4, 1e-3]\nnum_epochs = 15\n"
    with pytest.raises(IntegrityError, match="condition limit"):
        _validate_tuning_execution_budget(code)


def test_tuning_budget_rejects_too_many_total_epochs():
    code = "learning_rates = [5e-4, 1e-3]\nnum_epochs = 16\n"
    with pytest.raises(IntegrityError, match="epoch budget"):
        _validate_tuning_execution_budget(code)


def test_tuning_budget_accepts_two_baseline_length_candidates():
    code = "learning_rates = [5e-4, 1e-3]\nnum_epochs = 15\n"
    _validate_tuning_execution_budget(code)


def test_ablation_budget_rejects_four_nonliteral_conditions():
    code = """
num_epochs = 15
ablation_configs = [
    ("none", baseline_transform),
    ("color", color_transform),
    ("geometry", geometry_transform),
    ("full", full_transform),
]
"""
    with pytest.raises(IntegrityError, match="condition limit"):
        _validate_experiment_execution_budget(
            code, "4_ablation_studies_1_first_attempt"
        )


def test_proposed_stage_rejects_retraining_baseline_condition():
    code = """
def train_condition(label, transform):
    loss.backward()
    optimizer.step()

train_condition("baseline", baseline_transform)
train_condition("proposed_method", augmented_transform)
"""
    with pytest.raises(IntegrityError, match="trusted baseline artifact"):
        _validate_experiment_execution_budget(
            code, "3_creative_research_1_first_attempt"
        )


def test_proposed_stage_accepts_one_augmented_condition():
    code = """
num_epochs = 15
def train_condition(label, transform):
    loss.backward()
    optimizer.step()

train_condition("proposed_method", augmented_transform)
"""
    _validate_experiment_execution_budget(
        code, "3_creative_research_1_first_attempt"
    )


def test_unlabelled_training_helpers_defer_to_runtime_contract():
    code = """
def train(transform):
    loss.backward()
    optimizer.step()

train(baseline_transform)
train(augmented_transform)
"""
    _validate_experiment_execution_budget(code, "3_creative_research_1_first_attempt")


def test_proposed_stage_rejects_internal_multi_seed_training_loop():
    code = """
num_epochs = 15
seeds = [0, 1, 2]
def train_condition(seed):
    loss.backward()
    optimizer.step()

for seed in seeds:
    train_condition(seed)
"""
    with pytest.raises(IntegrityError, match="one host-injected seed"):
        _validate_experiment_execution_budget(
            code, "3_creative_research_1_first_attempt"
        )


def test_proposed_stage_accepts_single_host_seed_variable():
    code = """
num_epochs = 15
seed = int(os.environ["SEED"])
def train_condition(seed):
    loss.backward()
    optimizer.step()

train_condition(seed)
"""
    _validate_experiment_execution_budget(
        code, "3_creative_research_1_first_attempt"
    )


def test_single_dataset_ablation_rejects_multi_dataset_design():
    code = """
# Multi-Dataset Generalization
from torchvision.transforms import ColorJitter, RandomRotation, RandomHorizontalFlip
"""
    with pytest.raises(IntegrityError, match="Single-dataset contract"):
        _validate_stage_semantics(
            code,
            "4_ablation_studies_1_first_attempt",
            ["color_jitter", "rotation", "flip"],
        )


def test_stage_evidence_excludes_inherited_and_seed_aggregation_nodes():
    class Node:
        def __init__(self, *, parent, seed=False, aggregate=False, buggy=False):
            self.parent = parent
            self.is_seed_node = seed
            self.is_seed_agg_node = aggregate
            self.is_buggy = buggy

    journal = type(
        "Journal",
        (),
        {
            "nodes": [
                Node(parent=None),
                Node(parent=object(), seed=True, aggregate=True),
                Node(parent=object(), buggy=True),
            ]
        },
    )()
    assert not has_valid_generated_node(journal)
    journal.nodes.append(Node(parent=object()))
    assert has_valid_generated_node(journal)


def test_exhausted_stage_gets_one_versioned_policy_repair_window():
    class Node:
        is_buggy = True
        parent = object()
        is_seed_node = False
        is_seed_agg_node = False

    stage = type(
        "Stage",
        (),
        {"name": "3_creative_research_1_first_attempt", "max_iterations": 2},
    )()
    journal = type("Journal", (), {"nodes": [Node(), Node()]})()
    manager = type(
        "Manager",
        (),
        {"current_stage": stage, "journals": {stage.name: journal}},
    )()

    assert _grant_policy_repair_window(manager)
    assert stage.max_iterations == 5
    assert manager._path_ai_policy_repairs[stage.name] == EXECUTION_POLICY_REVISION
    assert not _grant_policy_repair_window(manager)
    assert stage.max_iterations == 5



def test_import_preflight_failure_returns_repairable_node(project_root, tmp_path):
    class FakeDockerRunner:
        def run_python(self, *args, **kwargs):
            return ExecutionResult(("python",), 86, "", "IMPORT_PREFLIGHT_FAILED: bad symbol", 0.1, False)
    runner = AIScientistExperimentRunner(project_root, object(), FakeDockerRunner())
    *_, interpreter = runner._runtime_classes(tmp_path / "dataset")
    result = interpreter(tmp_path / "process").run("print('experiment')")
    assert result.exc_type == "ImportPreflightError"
    assert "IMPORT_PREFLIGHT_FAILED" in "".join(result.term_out)


def test_worker_task_description_retains_exported_interface(project_root, tmp_path):
    runner = AIScientistExperimentRunner(project_root, object(), object())
    manager, _, _, _ = runner._runtime_classes(tmp_path)
    instance = object.__new__(manager)
    instance.task_desc = {
        'Title': 'Study', 'Abstract': 'Study', 'Short Hypothesis': 'Study',
        'Research View Interface': {'array_keys': ['validation_images', 'validation_labels']},
        'Approved Research Execution Contract': {'baseline': {'name': 'Required CNN'}},
    }
    description = instance._get_task_desc_str()
    assert 'validation_images' in description
    assert 'validation_labels' in description
    assert 'Required CNN' in description


def test_code_response_retries_prose_with_actionable_feedback():
    from pathmnist.code_response import query_program
    requests = []
    responses = iter([
        'METHOD_SPEC: {}\nI will fix validation keys. The rest remains unchanged.',
        'METHOD_SPEC: {}\n```python\nprint("complete program")\n```',
    ])
    def query(**kwargs):
        import copy
        requests.append(copy.deepcopy(kwargs))
        return next(responses)
    original = {'Task': 'Repair'}
    plan, code = query_program(query, original, model='fake', temperature=0, retries=2)
    assert 'METHOD_SPEC' in plan
    assert 'print(' in code
    assert 'FULL corrected runnable' in requests[1]['user_message']['Parsing Feedback']
    assert 'I will fix' in requests[1]['user_message']['Previous invalid response (diagnostic only)']
    assert isinstance(requests[0]['system_message'], str)
    assert requests[0]['user_message']['Task'] == 'Repair'
    assert original == {'Task': 'Repair'}


def test_code_response_exhaustion_never_returns_prose_as_code():
    from pathmnist.code_response import CodeGenerationError, query_program
    calls = []
    def query(**kwargs):
        calls.append(kwargs)
        return 'METHOD_SPEC: {}\nI will fix the keys.'
    with pytest.raises(CodeGenerationError, match='after 2 attempts'):
        query_program(query, {}, model='fake', temperature=0, retries=2)
    assert len(calls) == 2


@pytest.mark.parametrize('response', [
    'METHOD_SPEC: {}\nThe remaining code is unchanged.',
    '```python\nif :\n```',
    '```python\n# omitted\n...\n```',
    '```python\nprint(1)\n```\n```python\nprint(',
])
def test_code_response_rejects_missing_invalid_placeholder_or_partial_program(response):
    from pathmnist.code_response import CodeGenerationError, parse_program
    with pytest.raises(CodeGenerationError):
        parse_program(response)


def test_code_only_response_does_not_trigger_wasteful_retry():
    from pathmnist.code_response import parse_program
    plan, code = parse_program('```python\nprint(1)\n```')
    assert plan
    assert code == 'print(1)'


def test_both_runtime_agents_use_strict_code_response_parser(project_root, tmp_path, monkeypatch):
    from types import SimpleNamespace
    runner = AIScientistExperimentRunner(project_root, object(), object())
    _, parallel, minimal, _ = runner._runtime_classes(tmp_path)
    from ai_scientist.treesearch import parallel_agent
    for cls in (parallel, minimal):
        instance = object.__new__(cls)
        instance.cfg = SimpleNamespace(agent=SimpleNamespace(code=SimpleNamespace(model='fake', temp=0)))
        calls = []
        def query(**kwargs):
            calls.append(kwargs)
            return '```python\nprint(1)\n```'
        monkeypatch.setattr(parallel_agent, 'query', query)
        plan, code = instance.plan_and_code_query({'Task': 'baseline'})
        assert len(calls) == 1
        assert 'Required complete response' in calls[0]['user_message']
        assert plan
        assert 'print(1)' in code


def test_repeated_preflight_failures_stop_before_more_paid_attempts():
    from types import SimpleNamespace as N
    from pathmnist.autonomous import _check_repeated_preflight_failures, AutonomousExperimentError
    def node(details):
        return N(is_buggy=True, _term_out=['Generated experiment rejected before execution: Unknown components: '+details])
    with pytest.raises(AutonomousExperimentError, match='REPEATED_PREFLIGHT_BLOCKED'):
        _check_repeated_preflight_failures(N(nodes=[node('loss'),node('model'),node('optimizer')]))
    _check_repeated_preflight_failures(N(nodes=[node('loss'),node('model')]))
    _check_repeated_preflight_failures(N(nodes=[node('loss'),N(is_buggy=False,_term_out=[]),node('model')]))


def test_baseline_structural_contract_signals_are_allowed():
    code = 'model = Net(num_classes=9)\nx=self.conv1(x)\ncriterion=nn.CrossEntropyLoss()\nloss.backward()'
    _validate_stage_semantics(code, '1_initial_implementation', ['conv1','num_classes','cross_entropy','label_smoothing'])
    with pytest.raises(IntegrityError, match='Baseline stage implements'):
        _validate_stage_semantics(code.replace('CrossEntropyLoss()', 'CrossEntropyLoss(label_smoothing=0.1)'), '1_initial_implementation', ['label_smoothing'])
