"""Offline, explicit stage-2 recovery; never launches agents or training."""
import argparse
import copy
import hashlib
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))
sys.path.insert(0, str(project / 'vendor/AI-Scientist-v2'))
from pathmnist.autonomous import AIScientistExperimentRunner, _write_agent_progress  # noqa: E402 -- local project and vendor paths must be initialized first
from pathmnist.autonomous_evidence import verified_metrics  # noqa: E402 -- local project and vendor paths must be initialized first
from pathmnist.tuning_evidence import model_signature  # noqa: E402 -- local project and vendor paths must be initialized first
from pathmnist.execution_control import task_lock  # noqa: E402 -- local project and vendor paths must be initialized first

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task-root', type=Path, required=True)
parser.add_argument('--apply', action='store_true')
args = parser.parse_args()
root = args.task_root.resolve()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

with task_lock(root):
    assert read(root / 'web_run.json')['state'] != 'running', 'Stop the task first'
    task = read(root / 'task.json')
    assert task['control'] in ('paused', 'interrupted'), 'Task must be stopped'
    checkpoint = root / 'experiment_logs/manager.pkl'
    original_hash = digest(checkpoint)
    state = pickle.loads(checkpoint.read_bytes())  # Trusted local application checkpoint only.
    stage = state['current_stage']
    assert stage.name.startswith('2_'), 'Only interrupted stage 2 is supported'
    assert all(s.stage_number <= 2 for s in state['stages']), 'Later-stage recovery needs separate review'
    journal = state['journals'][stage.name]
    assert len(journal.nodes) > 1, 'No legacy tuning attempts to replace'
    baseline = journal.nodes[0]
    assert baseline.is_buggy is False and not baseline.is_seed_node
    code_hash = hashlib.sha256(baseline.code.encode()).hexdigest()
    evidence = root / 'experiment_logs/evidence' / code_hash
    metrics = verified_metrics(evidence, root/'dataset/research_view/dataset_profile.json', code_hash)
    model_signature(baseline.code)
    assert any(n.code == baseline.code and n.is_buggy is False
               for name,j in state['journals'].items() if name.startswith('1_') for n in j.nodes)
    stage1 = {name: pickle.dumps(j) for name,j in state['journals'].items() if name.startswith('1_')}
    old_ids = [n.id for n in journal.nodes[1:]]
    baseline = copy.deepcopy(baseline)
    baseline.parent = None
    baseline.children = set()
    baseline.is_buggy_plots = False
    journal.nodes = [baseline]
    stage.max_iterations = 4  # Inherited baseline plus at most three generated attempts.
    stage.num_drafts = 0
    stage.goals = ('Tune exactly two learning rates including the baseline value; preserve all other controls. '
                   'Save complete tuning_evidence.json with both epoch histories and selected checkpoint. '
                   'Use the sole local dataset. Valid no-improvement results complete tuning.')
    runner = AIScientistExperimentRunner(project, object(), object())
    manager_class, *_ = runner._runtime_classes(root / 'dataset/research_view')
    manager = object.__new__(manager_class)
    manager.__dict__.update(state)
    assert manager._check_stage_completion(stage)[0] is False
    assert manager._check_substage_completion(stage, journal)[0] is False
    assert all(pickle.dumps(state['journals'][name]) == raw for name,raw in stage1.items())
    new_bytes = pickle.dumps(state)
    checked = pickle.loads(new_bytes)
    assert len(checked['journals'][stage.name].nodes) == 1
    assert checked['current_stage'].name == stage.name
    report = {'prepared_at': datetime.now(timezone.utc).isoformat(), 'task_id': task['task_id'],
              'original_checkpoint_sha256': original_hash, 'stage': stage.name,
              'baseline_code_sha256': code_hash, 'baseline_validation_accuracy': metrics['accuracy'],
              'archived_attempt_ids': old_ids, 'remaining_active_nodes': 1,
              'max_generated_attempts': 3, 'training_started': False, 'paid_calls': 0,
              'checks': ['baseline artifact hashes and predictions', 'stage1 unchanged',
                         'checkpoint roundtrip', 'stage2 pending without model calls'],
              'applied': args.apply}
    if args.apply:
        backup = root / 'experiment_logs/recovery_backups' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup.mkdir(parents=True, exist_ok=False)
        files = [checkpoint, root/'task.json', root/'web_run.json', root/'web_run.log', root/'agent_progress.json']
        hashes = {}
        for path in files:
            if path.is_file():
                shutil.copy2(path, backup/path.name)
                assert digest(path) == digest(backup/path.name)
                hashes[path.name] = digest(path)
        report['backup'] = str(backup.relative_to(root))
        report['backup_sha256'] = hashes
        assert digest(checkpoint) == original_hash
        temporary = checkpoint.with_suffix('.recovery.tmp')
        temporary.write_bytes(new_bytes)
        temporary.replace(checkpoint)
        assert digest(checkpoint) == hashlib.sha256(new_bytes).hexdigest()
        _write_agent_progress(root, stage, journal, state['stage_history'])
        status = read(root/'web_run.json')
        status['state'] = 'interrupted'
        status['message'] = '恢复准备完成，仍未运行；旧记录已备份，下一次从已验证基线重新调参。'
        status['updated_at'] = report['prepared_at']
        (root/'web_run.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
        assert read(root/'task.json')['control'] == task['control']
        report['prepared_checkpoint_sha256'] = digest(checkpoint)
        (backup/'recovery_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (root/'tuning_recovery.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report, indent=2))

