"""One paid, constrained augmentation patch; not a full tree-search acceptance."""
import argparse
import ast
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from gate_a.budget import BudgetLedger
from gate_a.config import RunnerConfig, load_config
from gate_a.pipeline import select_live_models
from gate_a.provider import ZhipuProvider
from gate_a.runner import DockerRunner
from pathmnist.autonomous import AIScientistExperimentRunner
from pathmnist.autonomous_evidence import verified_metrics
from pathmnist.execution_control import task_lock


def validate_patch(text):
    """Only torchvision augmentation constructors with literal bounded arguments."""
    tree = ast.parse(text, mode='eval')
    if not isinstance(tree.body, ast.List) or not 1 <= len(tree.body.elts) <= 4:
        raise ValueError('Expected 1-4 transforms in a Python list')
    allowed = {'ColorJitter', 'RandomRotation', 'RandomHorizontalFlip', 'RandomVerticalFlip'}
    for call in tree.body.elts:
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == 'transforms'
                and call.func.attr in allowed):
            raise ValueError('Only allowlisted torchvision transforms are permitted')
        for value in [*call.args, *(kw.value for kw in call.keywords)]:
            if not isinstance(value, ast.Constant) or type(value.value) not in (int, float):
                raise ValueError('Arguments must be numeric literals')
            limit = 45 if call.func.attr == 'RandomRotation' else 0.5
            if not 0 <= value.value <= limit:
                raise ValueError('Augmentation argument outside smoke bounds')
        if any(kw.arg is None for kw in call.keywords):
            raise ValueError('Keyword expansion is forbidden')
    return ', '.join(ast.unparse(call) for call in tree.body.elts) + ','


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--baseline-root', required=True, type=Path)
    parser.add_argument('--task-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Shared task lock prevents concurrent modifications of the formal budget.
    with task_lock(args.task_root), task_lock(root):
        config = load_config(args.project_root / 'configs/gate_a_llm.yaml')
        roles = dict(config.roles)
        roles['experiment_code'] = replace(roles['experiment_code'], max_input_tokens=2000, max_output_tokens=1000)
        config = replace(config, roles=roles, provider=replace(config.provider, max_retries=0, timeout_seconds=120))
        selected = select_live_models(config)
        maximum = selected['experiment_code'].maximum_cost(roles['experiment_code'], config.budget.reserve_margin)
        if maximum > 0.25:
            raise RuntimeError('Single request reservation exceeds $0.25 smoke cap')
        ledger = BudgetLedger(args.task_root / 'budget.json', 8.0)
        before = ledger.snapshot()
        provider = ZhipuProvider(config, selected, ledger, root / 'responses')
        request_id = 'engineering-llm-gpu-smoke-20260827-single-patch'
        print(json.dumps({'phase': 'generation', 'budget_before': asdict(before), 'maximum_request_usd': maximum}), flush=True)
        # One low-level text request: no empty-response, JSON, or HTTP retries.
        # Stable ID and response cache also prevent paid retries after interruption.
        response = provider._call('experiment_code', request_id,
            'Return only a Python list expression. No markdown, explanation or executable statements.',
            'Design a mild pathology color/orientation augmentation for a ResNet18 PathMNIST engineering smoke. '
            'Return a list of 2-4 transforms using only transforms.ColorJitter, transforms.RandomRotation, '
            'transforms.RandomHorizontalFlip, transforms.RandomVerticalFlip. '
            'All arguments must be numeric constants; rotation degrees 0-45, other values 0-0.5. '
            'Include color jitter, rotation and at least one flip. Host keeps seed=0, Adam lr=0.001, '
            'batch_size=32, native64 input, 2 epochs and all data/evaluation/manifest code fixed.', None, None)
        patch = validate_patch(response.value.strip())
        template = (args.project_root / 'scripts/fixtures/gpu_smoke_experiment.py').read_text()
        code = template.replace('# AUGMENTATIONS', patch).replace("'ROLE'", "'proposed_method'")
        (root / 'generated_candidate.py').write_text(code)
        digest = hashlib.sha256(code.encode()).hexdigest()
        view = args.baseline_root / 'dataset'
        baseline_report = json.loads((args.baseline_root / 'smoke_report.json').read_text())
        baseline_record = next(row for row in baseline_report['records'] if row['role'] == 'baseline')
        baseline_dir = args.baseline_root / 'experiment_logs/evidence' / baseline_record['code_sha256']
        baseline = verified_metrics(baseline_dir, view / 'dataset_profile.json', baseline_record['code_sha256'])
        runner = DockerRunner(RunnerConfig('path-scientist-pathmnist-runner:0.1', 180, 2.0, '4g', 128, ('docker',)),
                              gpus='all', stream_output=True)
        adapter = AIScientistExperimentRunner(args.project_root, provider, runner)
        _, _, minimal, interpreter = adapter._runtime_classes(view, validation_count=72,
            intervention_signals=('color_jitter', 'rotation', 'flip'))
        directory = root / 'experiment_workspace/proposed_method'
        engine = interpreter(directory)
        engine.stage_name = '3_creative_research'
        print('phase=gpu_execution (single augmented model only)', flush=True)
        result = engine.run(code)
        if result.exc_type:
            raise RuntimeError(f'{result.exc_type}: {result.term_out}')
        metrics = verified_metrics(directory / 'working', view / 'dataset_profile.json', digest)
        from ai_scientist.treesearch.journal import Node
        node = Node(code=code, plan='LLM-generated constrained augmentation patch')
        minimal.parse_exec_result(object.__new__(minimal), node, result, directory)
        assert node.is_buggy is False
        controls = ('dataset', 'model', 'optimizer', 'learning_rate', 'epochs', 'batch_size', 'seed', 'input_resolutions', 'selection_metric')
        baseline_manifest = json.loads((baseline_dir / 'experiment_manifest.json').read_text())
        candidate_manifest = json.loads((directory / 'working/experiment_manifest.json').read_text())
        assert all(baseline_manifest[key] == candidate_manifest[key] for key in controls)
        report = {'passed': True, 'kind': 'llm-controlled-patch-smoke-not-full-tree-search',
                  'model': selected['experiment_code'].model_id, 'request_id': request_id,
                  'code_sha256': digest, 'augmentation_patch': patch,
                  'baseline_accuracy': baseline['accuracy'], 'candidate_accuracy': metrics['accuracy'],
                  'paired_seed': 0, 'epochs': 2, 'baseline_retrained': False,
                  'fixed_controls_match': True, 'sealed_test_accessed': False,
                  'budget_after': asdict(ledger.snapshot()),
                  'request_cost_usd': json.loads(ledger.path.read_text())['requests'][request_id]['actual_usd']}
        (root / 'smoke_report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
