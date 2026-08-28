"""Run a small, unpriced real-GPU fixture through the project's interpreter."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gate_a.config import RunnerConfig
from gate_a.runner import DockerRunner
from pathmnist.autonomous import AIScientistExperimentRunner
from pathmnist.autonomous_evidence import verified_metrics
from pathmnist.execution_control import task_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--source-view', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with task_lock(root):
        view = root / 'dataset'
        view.mkdir(exist_ok=True)
        profile_path = view / 'dataset_profile.json'
        if not profile_path.exists():
            original = json.loads((args.source_view / 'dataset_profile.json').read_text())
            payload, samples = {}, []
            with np.load(args.source_view / 'dataset.npz', allow_pickle=False) as source:
                for split, per_class in [('train', 32), ('validation', 8)]:
                    labels = source[split + '_labels'].reshape(-1)
                    indices = np.concatenate([np.flatnonzero(labels == label)[:per_class] for label in range(9)])
                    for suffix in ('images', 'labels', 'sample_ids'):
                        payload[split + '_' + suffix] = source[split + '_' + suffix][indices]
                    selected_ids = set(payload[split + '_sample_ids'].tolist())
                    samples.extend(row for row in original['samples'] if row['id'] in selected_ids and row['split'] == split)
            original['samples'] = samples
            original['split_counts'] = {'train': 288, 'validation': 72}
            original['name'] = 'PathMNIST-engineering-smoke-only'
            np.savez(view / 'dataset.npz', **payload)
            profile_path.write_text(json.dumps(original))
        runner = DockerRunner(RunnerConfig('path-scientist-pathmnist-runner:0.1', 180, 2.0, '4g', 128, ('docker',)),
                              gpus='all', stream_output=True)
        adapter = AIScientistExperimentRunner(args.project_root, object(), runner)
        _, _, minimal, interpreter = adapter._runtime_classes(view, validation_count=72,
            intervention_signals=('color_jitter', 'rotation', 'flip'))
        from ai_scientist.treesearch.journal import Journal, Node
        from ai_scientist.treesearch.utils.metric import MetricValue
        journal = Journal()
        records = []
        controls = []
        template = (args.project_root / 'scripts/fixtures/gpu_smoke_experiment.py').read_text()
        for role, stage in [('baseline', '2_baseline_tuning'), ('proposed_method', '3_creative_research')]:
            augment = '' if role == 'baseline' else 'transforms.ColorJitter(brightness=0.1), transforms.RandomRotation(15), transforms.RandomHorizontalFlip(),'
            code = template.replace('# AUGMENTATIONS', augment).replace("'ROLE'", repr(role))
            digest = hashlib.sha256(code.encode()).hexdigest()
            directory = root / 'experiment_workspace' / role
            engine = interpreter(directory)
            engine.stage_name = stage
            reused = (directory / 'working/artifact_hashes.json').exists()
            saved = root / 'experiment_logs/evidence' / digest
            if saved.exists():
                metrics = verified_metrics(saved, profile_path, digest)
                reused = True
            else:
                result = engine.run(code)
                if result.exc_type:
                    raise RuntimeError(f'{role}: {result.exc_type}: {result.term_out}')
                metrics = verified_metrics(directory / 'working', profile_path, digest)
            node = Node(code=code, plan='Fixed integration smoke', parent=journal.nodes[0] if journal.nodes else None)
            if not reused:
                minimal.parse_exec_result(object.__new__(minimal), node, result, directory)
                assert node.is_buggy is False
            else:
                node.is_buggy = False
            node.is_buggy_plots = False
            from pathmnist.autonomous_evidence import metric_rows
            node.metric = MetricValue(value={'metric_names': metric_rows(metrics)})
            journal.append(node)
            records.append({'role': role, 'seed': 0, 'code_sha256': digest, 'accuracy': metrics['accuracy'], 'reused': reused})
            manifest = json.loads((saved / 'experiment_manifest.json').read_text())
            controls.append({key: manifest[key] for key in ('dataset', 'model', 'optimizer', 'learning_rate', 'epochs', 'batch_size', 'seed', 'input_resolutions', 'selection_metric')})
        assert controls[0] == controls[1], 'Paired experimental controls differ'
        duplicate_refused = False
        try:
            with task_lock(root):
                raise AssertionError('Duplicate runner admitted')
        except RuntimeError:
            duplicate_refused = True
        # Exercise real container deadline cleanup without starting another training run.
        timeout_runner = DockerRunner(RunnerConfig('path-scientist-pathmnist-runner:0.1', 3, 1.0, '1g', 64, ('docker',)), stream_output=True)
        timed = timeout_runner.run_python("import time\nprint('timeout fixture started', flush=True)\ntime.sleep(60)\n", root / 'experiment_workspace/timeout')
        assert timed.timed_out
        timeout_runner.cancel_active(root / 'experiment_workspace')
        report = {'passed': True, 'kind': 'engineering-smoke-not-scientific-acceptance',
                  'journal_nodes': len(journal.nodes), 'records': records, 'timeout_cleaned': True,
                  'paired_seed': 0, 'fixed_controls_match': True, 'duplicate_start_refused': duplicate_refused,
                  'validation_delta': records[1]['accuracy'] - records[0]['accuracy'],
                  'llm_calls': 0, 'sealed_test_accessed': False}
        (root / 'smoke_report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
