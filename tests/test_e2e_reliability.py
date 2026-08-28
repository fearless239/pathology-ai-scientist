"""Cross-boundary regressions; external compute/services are replaced, not stage outcomes."""
import hashlib
import json
import re
from concurrent.futures import Future
from types import SimpleNamespace as N

import numpy as np
import pytest
from omegaconf import OmegaConf

from pathmnist.autonomous import AIScientistExperimentRunner, AutonomousTaskWorkspace
from pathmnist.autonomous_freeze import _paired_experiments
from pathmnist.dataset_adapter import DatasetAdapter
from pathmnist.research_contract import generate_contract, write_contract


def test_freeze_uses_fulfillment_identity_not_highest_draft():
    rows = [{'experiment_id': name, 'seed': 0,
             'contract_role': 'baseline' if name == 'baseline' else 'proposed_method'}
            for name in ('baseline', 'best_draft', 'repeat')]
    fulfillment = {'paired_sources': [{'seed': 0, 'baseline_experiment_id': 'baseline',
                                       'proposed_experiment_id': 'repeat'}]}
    selected = _paired_experiments(rows, fulfillment, {'repeat_plan': {'seeds': [0]}})
    assert [row['experiment_id'] for row in selected] == ['baseline', 'repeat']
    with pytest.raises(RuntimeError, match='identities'):
        _paired_experiments(rows[:2], fulfillment, {'repeat_plan': {'seeds': [0]}})


@pytest.mark.parametrize('repeats', [1, 2])
@pytest.mark.parametrize('data_mode,full_work', [('npz', False), ('imagefolder', False), ('patient_manifest', False), ('npz', True)])
@pytest.mark.parametrize('primary_metric', ['accuracy', 'macro_f1', 'class_f1', 'confusion_pair_mean_f1'])
def test_real_upstream_generation_execution_and_resume(project_root, tmp_path, monkeypatch, repeats, data_mode, full_work, primary_metric):
    source = tmp_path / 'data.npz'
    np.savez(source, **{f'{split}_{kind}': np.zeros((2, 8, 8, 3), dtype=np.uint8)
                        if kind == 'images' else np.array([0, 1])
                        for split in ('train', 'val', 'test') for kind in ('images', 'labels')})
    if data_mode != 'npz':
        from PIL import Image
        source = tmp_path / 'images'
        rows = []
        for split in ('train', 'validation', 'test'):
            for label in (0, 1):
                path = source / split / str(label) / 'image.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (8, 8), (label, label, label)).save(path)
                rows.append({'path': path.relative_to(source).as_posix(), 'label': str(label),
                             'split': split, 'patient_id': f'{split}-{label}'})
        if data_mode == 'patient_manifest':
            import csv
            with (source / 'labels.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
    spec = DatasetAdapter().discover(source)
    workspace = AutonomousTaskWorkspace.create(tmp_path / 'state', 'audit')
    spec.write_profile(workspace.dataset / 'dataset_profile.json')
    workspace.prepare_research_dataset(spec)
    contract = generate_contract('Single-run comparison of label smoothing accuracy', spec.to_dict())
    contract['interventions'][0]['implementation_signals'] = ['label_smoothing']
    contract['metrics']['primary']['name'] = primary_metric
    if primary_metric == 'class_f1':
        contract['metrics']['primary'].update(class_id=0, scope='specified_class')
    if primary_metric == 'confusion_pair_mean_f1':
        contract['metrics']['primary']['scope'] = 'baseline_locked_confusion_pair'
        contract['subgroup_policy'] = {'kind': 'lock_most_confused_pair_from_baseline',
                                      'selection_split': 'validation', 'locked_before_intervention_comparison': True}
    metric_score = 0.5 if primary_metric == 'accuracy' else 2/3 if primary_metric == 'class_f1' else 1/3
    contract['experiment_policy'] = {'tuning': full_work, 'ablation': full_work}
    if full_work:
        contract['required_ablations'] = [{'id': 'component_ablation', 'description': 'Disable label smoothing'}]
    contract['repeat_plan'].update(count=repeats, seeds=list(range(repeats)))
    from pathmnist.research_contract import contract_sha256
    contract.pop('contract_sha256', None)
    contract['contract_sha256'] = contract_sha256(contract)
    write_contract(workspace.root, contract)
    (workspace.research / 'research_contract_approval.json').write_text(json.dumps({
        'approved': True, 'contract_sha256': contract['contract_sha256']}))
    calls = []
    successful = []
    failed = []

    class Provider:
        def call_text(self, *args):
            prompt = str(args)
            if 'propose ONE new hyperparameter tuning idea' in prompt:
                return ('HYPERPARAM NAME: learning rate\nDESCRIPTION: Compare two learning rates.', {})
            if 'propose ONE new ablation study' in prompt:
                return ('ABLATION NAME: disable smoothing\nABLATION DESCRIPTION: Remove label smoothing.', {})
            calls.append('generate')
            completed = {kind for kind, seed in successful}
            sequence = ['baseline', 'tuned', 'method', 'ablation'] if full_work else ['baseline', 'method']
            kind = next((kind for kind in sequence if kind not in completed), sequence[-1])
            smoothing = 0.1 if kind == 'method' else 0.0
            return ('METHOD_SPEC: {"hypothesis":"fixture","components":[],"changes":[],"preserved":[]}\n'
                        f'```python\n# fixture_kind={kind}\nimport torch\nimport torch.nn as nn\n'
                        "FINAL_TRAINING_PLAN = {'max_epochs': 1, 'early_stopping': {'enabled': False}, 'search_epochs': []}\n"
                    'class Net(nn.Module):\n    def forward(self, x):\n        return x\n'
                    'model = Net()\nHAS_TRAIN_SPLIT = True\n'
                    f'loss = nn.CrossEntropyLoss(label_smoothing={smoothing})\n'
                    'loss.backward()\noptimizer.step()\n'
                    'if not HAS_TRAIN_SPLIT:\n    model.load_state_dict(torch.load("/workspace/model_checkpoint.pt"))\n```', {})

        def call_json(self, *args):
            raise AssertionError('Unexpected auxiliary LLM call: ' + str(args[:2]))

    class Runtime:
        def __init__(self, *args, **kwargs):
            pass

        def cancel_active(self, root):
            pass

        def run_python(self, code, directory, *args, **kwargs):
            calls.append('execute')
            working = directory / 'working'
            working.mkdir(parents=True, exist_ok=True)
            role = 'proposed_method' if 'label_smoothing=0.1' in code else 'baseline'
            kind = re.search(r'fixture_kind=(\w+)', code).group(1)
            if kind == 'ablation':
                role = 'component_ablation'
            match = re.search(r'seed = (\d+)', code)
            seed = int(match.group(1)) if match else 0
            if repeats == 2 and role == 'baseline' and seed == 1 and not failed:
                failed.append(seed)
                return N(succeeded=False, elapsed_seconds=0.01, stdout='', stderr='injected interruption', exit_code=1, timed_out=False)
            successful.append((kind if full_work else role, seed))
            view = kwargs.get('dataset_mount') or workspace.dataset / 'research_view'
            profile = json.loads((view / 'dataset_profile.json').read_text())
            samples = [row for row in profile['samples'] if row['split'] == 'validation']
            raw = {'seed': seed, 'metrics': {'accuracy': 0.5}, 'sample_ids': [row['id'] for row in samples],
                   'targets': [0, 1], 'predictions': [0, 0]}
            (working / 'experiment_result.json').write_text(json.dumps(raw))
            (working / 'experiment_manifest.json').write_text(json.dumps({
                'schema_version': 1, 'dataset': spec.name, 'model': 'Net', 'optimizer': 'Adam',
                'learning_rate': 0.001, 'epochs': 1, 'max_epochs': 1, 'early_stopping': {'enabled': False},
                'training_runs': [{'max_epochs': 1, 'epochs': 1}] * (2 if kind == 'tuned' and 'PATH_AI_REPEAT' not in code else 1), 'seed': seed, 'batch_size': 2,
                'input_resolutions': [[8, 8]], 'selection_metric': primary_metric, 'hardware': 'fixture',
                'primary_metric': primary_metric,
                'checkpoint_selection': {'metric': 'validation_loss', 'mode': 'min'}}))
            (working / 'contract_execution.json').write_text(json.dumps({'contract_role': role, 'training_seed': seed}))
            (working / 'model_checkpoint.pt').write_bytes(b'opaque test checkpoint')
            if kind == 'tuned' and 'PATH_AI_REPEAT' not in code:
                (working / 'tuning_evidence.json').write_text(json.dumps({
                    'schema_version': 1, 'complete': True, 'seed': seed, 'selection_metric': primary_metric,
                    'selected_learning_rate': 0.001,
                    'candidates': [{'learning_rate': lr, 'validation_metric': metric_score, 'selected_epoch': 1,
                                    'history': [{'epoch': 1, 'train_loss': 1.0, 'validation_loss': 1.0, 'validation_metric': metric_score}]}
                                   for lr in [0.001, 0.002]],
                }))
            return N(succeeded=True, elapsed_seconds=0.01, stdout='', stderr='', exit_code=0, timed_out=False)

    class InlineExecutor:
        def __init__(self, **kwargs):
            self._processes = {}

        def submit(self, function, *args):
            future = Future()
            try:
                future.set_result(function(*args))
            except BaseException as error:
                future.set_exception(error)
            return future

        def shutdown(self, **kwargs):
            pass

    runner = AIScientistExperimentRunner(project_root, Provider(), Runtime())
    runner._runtime_classes(workspace.dataset / 'research_view')
    monkeypatch.setattr('ai_scientist.treesearch.parallel_agent.ProcessPoolExecutor', InlineExecutor)
    cfg = OmegaConf.load(project_root / 'vendor/AI-Scientist-v2/bfts_config.yaml')
    cfg.agent.multi_seed_eval.num_seeds = repeats
    cfg.agent.stages.stage1_max_iters = 2
    cfg.agent.stages.stage3_max_iters = 2
    cfg.agent.stages.stage2_max_iters = 2
    cfg.agent.stages.stage4_max_iters = 2
    if repeats == 2:
        with pytest.raises(RuntimeError, match='Repeat seed 1 failed'):
            runner.run('audit', 'Single-run label smoothing', spec, workspace, cfg)
        assert successful == [('baseline', 0)]
    manager = runner.run('audit', 'Single-run label smoothing', spec, workspace, cfg)
    assert manager.current_stage is None
    assert len([name for name in manager.journals if name.startswith(('1_', '3_'))]) == 2
    assert any(name.startswith('2_') for name in manager.journals) == full_work
    assert any(name.startswith('4_') for name in manager.journals) == full_work
    before = list(calls)
    resumed = runner.run('audit', 'Single-run label smoothing', spec, workspace, cfg)
    assert resumed.current_stage is None
    assert calls == before
    roles = ('baseline', 'tuned', 'method', 'ablation') if full_work else ('baseline', 'proposed_method')
    assert sorted(successful) == sorted((role, seed) for role in roles for seed in range(repeats))
    for journal in resumed.journals.values():
        assert journal.good_nodes
        for node in journal.good_nodes:
            digest = hashlib.sha256(node.code.encode()).hexdigest()
            assert (workspace.experiment_logs / 'evidence' / digest / 'artifact_hashes.json').is_file()
    from pathmnist.autonomous_export import export_journals
    from pathmnist.research_contract import evaluate_fulfillment
    from pathmnist.autonomous_freeze import freeze_best
    from pathmnist.candidates import require_inference_candidate
    exported = export_journals(project_root, tmp_path / 'state', 'audit')
    assert len(exported['experiments']) == len(roles) * repeats
    report = evaluate_fulfillment(workspace.root)
    assert report['passed'], report['errors']
    assert report['statistics']['n'] == repeats
    (workspace.research / 'semantic_review.json').write_text(json.dumps({'passed': True}))
    (workspace.root / 'task.json').write_text(json.dumps({'schema_version': 2, 'stages': {}}))
    freeze_best(project_root, tmp_path / 'state', 'audit')
    require_inference_candidate(workspace.root)
    bundle = json.loads((workspace.root / 'candidate_frozen/comparison_bundle.json').read_text())
    assert bundle['selected_candidate_experiment_id'] in {arm['experiment_id'] for arm in bundle['experiments']}
    from pathmnist import autonomous_test
    monkeypatch.setattr('gate_a.runner.DockerRunner', Runtime)
    autonomous_test.approve(project_root, tmp_path / 'state', 'audit')
    commit = autonomous_test._commit_test
    def interrupted_commit(*args):
        raise RuntimeError('injected stage commit interruption')
    monkeypatch.setattr(autonomous_test, '_commit_test', interrupted_commit)
    with pytest.raises(RuntimeError, match='injected stage commit'):
        autonomous_test.evaluate(project_root, tmp_path / 'state', 'audit')
    assert (workspace.root / 'final_evaluation/completed.json').is_file()
    before = list(calls)
    monkeypatch.setattr(autonomous_test, '_commit_test', commit)
    result = autonomous_test.evaluate(project_root, tmp_path / 'state', 'audit')
    assert result['split'] == 'test'
    assert calls == before  # Neither validation preflight nor sealed inference executes twice.
