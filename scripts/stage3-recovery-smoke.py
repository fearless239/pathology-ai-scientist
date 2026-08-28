"""Real stage-3 worker/recovery smoke with cached code, no live provider."""
import argparse
import copy
import hashlib
import json
import shutil
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from gate_a.config import RunnerConfig
from gate_a.runner import DockerRunner
from pathmnist.autonomous import AIScientistExperimentRunner, _patched_upstream, has_valid_generated_node
from pathmnist.autonomous_evidence import verified_metrics, metric_rows
from pathmnist.execution_control import task_lock


class InjectedInterruption(RuntimeError):
    pass


class StageBoundaryReached(RuntimeError):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root', 'baseline-root', 'candidate-root', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with task_lock(root):
        checkpoint = root / 'experiment_logs/manager.pkl'
        if checkpoint.exists():
            raise RuntimeError('Use a fresh output directory; this smoke creates its own checkpoint')
        view = root / 'dataset/research_view'
        shutil.copytree(args.baseline_root / 'dataset', view)
        workspace = SimpleNamespace(root=root, dataset=root / 'dataset',
            experiment_workspace=root / 'experiment_workspace', experiment_logs=root / 'experiment_logs')
        workspace.experiment_workspace.mkdir()
        workspace.experiment_logs.mkdir()
        report = json.loads((args.baseline_root / 'smoke_report.json').read_text())
        record = next(row for row in report['records'] if row['role'] == 'baseline')
        evidence = args.baseline_root / 'experiment_logs/evidence' / record['code_sha256']
        metrics = verified_metrics(evidence, view / 'dataset_profile.json', record['code_sha256'])
        baseline_code = (args.project_root / 'scripts/fixtures/gpu_smoke_experiment.py').read_text().replace('# AUGMENTATIONS', '').replace("'ROLE'", "'baseline'")
        assert hashlib.sha256(baseline_code.encode()).hexdigest() == record['code_sha256']
        candidate = (args.candidate_root / 'generated_candidate.py').read_text()
        candidate_report = json.loads((args.candidate_root / 'smoke_report.json').read_text())
        assert hashlib.sha256(candidate.encode()).hexdigest() == candidate_report['code_sha256']
        runner = DockerRunner(RunnerConfig('path-scientist-pathmnist-runner:0.1', 180, 2.0, '4g', 128, ('docker',)), gpus='all', stream_output=True)
        adapter = AIScientistExperimentRunner(args.project_root, object(), runner)
        manager_cls, parallel, minimal, interpreter = adapter._runtime_classes(view, validation_count=72,
            intervention_signals=('color_jitter', 'rotation', 'flip'))
        from ai_scientist.treesearch.agent_manager import Stage, StageTransition
        from ai_scientist.treesearch.journal import Journal, Node
        from ai_scientist.treesearch.utils.metric import MetricValue
        cfg = OmegaConf.load(args.project_root / 'vendor/AI-Scientist-v2/bfts_config.yaml')
        cfg.workspace_dir = str(workspace.experiment_workspace)
        cfg.log_dir = str(workspace.experiment_logs)
        cfg.data_dir = str(view)
        cfg.copy_data = False
        cfg.agent.num_workers = 1
        cfg.agent.contract_metric = 'accuracy'
        cfg.agent.multi_seed_eval.num_seeds = 1
        cfg.agent.summary = None
        cfg.exec.timeout = 240
        cfg.generate_report = False
        cfg.experiment.num_syn_datasets = 1
        desc = json.dumps({'Title': 'Independent stage-3 engineering smoke',
            'Abstract': 'Cached augmentation code on 288/72 PathMNIST subset.',
            'Short Hypothesis': 'Test worker and resume boundaries, not scientific improvement.',
            'Experiments': 'One seed, two epochs, reuse baseline; never read test data.',
            'Risk Factors and Limitations': 'No live code generation or scientific acceptance.'})

        def replay_query(*args, **kwargs):
            # Only generation may consume the cached response; unexpected calls fail.
            prompt = kwargs.get('system_message', {})
            if isinstance(prompt, dict) and 'summarizing experimental progress' in prompt.get('Introduction', ''):
                return 'Smoke fixture summary: reuse the verified baseline; evaluate one augmentation child.'
            if kwargs.get('func_spec') is not None:
                raise RuntimeError('Unexpected structured query in replay smoke')
            receipt = root / 'replay_query.json'
            if receipt.exists():
                raise RuntimeError('Unexpected second query; automatic generation retry forbidden')
            receipt.write_text(json.dumps({'cached_code_sha256': candidate_report['code_sha256']}))
            return 'Replay the previously generated augmentation with fixed training controls.\n```python\n' + candidate + '\n```'

        with _patched_upstream(replay_query, parallel, minimal, interpreter, sandbox_gpu_count=1):
            manager = manager_cls(desc, cfg, workspace.experiment_workspace)
            stage2 = Stage('2_baseline_tuning_1_first_attempt', 'verified baseline', [], 1, 0, 2)
            stage3 = Stage('3_creative_research_1_first_attempt', 'bounded smoke', [], 2, 0, 3)
            parent = Node(code=baseline_code, plan='Previously verified baseline')
            parent.is_buggy = parent.is_buggy_plots = False
            parent.metric = MetricValue(value={'metric_names': metric_rows(metrics)})
            previous, active = Journal(), Journal()
            previous.append(parent)
            active.append(copy.deepcopy(parent))
            manager.stages = [stage2, stage3]
            manager.journals = {stage2.name: previous, stage3.name: active}
            manager.current_stage = stage3
            manager.stage_history = [StageTransition(stage2.name, stage3.name, 'Smoke baseline reuse', {})]

            def interrupt_after_save(stage, journal):
                if not has_valid_generated_node(journal):
                    adapter._checkpoint(checkpoint, manager)
                    raise RuntimeError('Generated child failed; inspect smoke checkpoint and logs')
                adapter._checkpoint(checkpoint, manager)
                raise InjectedInterruption('Deliberate stop after durable stage-3 child')

            try:
                manager.run(interpreter, step_callback=interrupt_after_save)
            except InjectedInterruption:
                pass
            else:
                raise AssertionError('Interruption was not exercised')
            restored = adapter._load_or_create_manager(checkpoint, manager_cls, desc, cfg, workspace)
            assert restored._check_stage_completion(restored.current_stage)[0]
            before = len(restored.journals[stage3.name].nodes)
            saved_factory = manager_cls._create_agent_for_stage

            def stop_at_ablation(self, stage):
                if stage.name.startswith('4_'):
                    raise StageBoundaryReached('Stop before ablation execution')
                return saved_factory(self, stage)

            manager_cls._create_agent_for_stage = stop_at_ablation
            try:
                with patch.object(runner, 'run_python', side_effect=AssertionError('Resume attempted another sandbox execution')):
                    restored.run(interpreter)
            except StageBoundaryReached:
                pass
            else:
                raise AssertionError('Resume did not reach ablation boundary')
            finally:
                manager_cls._create_agent_for_stage = saved_factory
            assert len(restored.journals[stage3.name].nodes) == before
            assert len(list((root / 'experiment_logs/evidence').iterdir())) == 1
            result = {'passed': True, 'live_llm_calls': 0, 'cached_llm_code': True,
                'real_process_pool': True, 'stage3_generated_child_valid': True,
                'checkpoint_reload': True, 'resume_without_retraining': True,
                'stopped_at': restored.current_stage.name, 'sealed_test_accessed': False,
                'scope': 'single child, cached generation, one seed, two epochs; not full research'}
            (root / 'smoke_report.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
