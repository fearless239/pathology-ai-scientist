"""Explicit offline recovery of a completed but unfair stage-3 comparison."""
import argparse
import hashlib
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

project = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(project), str(project / 'vendor/AI-Scientist-v2')]
from pathmnist.execution_control import task_lock  # noqa: E402
from pathmnist.autonomous_evidence import verified_metrics  # noqa: E402
from pathmnist.experiment_manifest import training_policy  # noqa: E402
from pathmnist.comparison_policy import bind_policy  # noqa: E402
from pathmnist.stage_policy import POLICIES  # noqa: E402
from pathmnist.autonomous import AIScientistExperimentRunner, _write_agent_progress  # noqa: E402


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task-root', type=Path, required=True)
parser.add_argument('--apply', action='store_true')
args = parser.parse_args()
root = args.task_root.resolve()
with task_lock(root):
    task = read(root/'task.json')
    status = read(root/'web_run.json')
    assert status['state'] != 'running' and task['control'] == 'interrupted'
    assert task['completed_stage'] == 'ablations_completed'
    assert all(task['stages'].get(k) == 'waiting' for k in ('candidate_frozen','test_evaluation_approved','test_evaluated'))
    fulfillment = read(root/'research/contract_fulfillment.json')
    assert not fulfillment['passed'] and fulfillment['errors']
    assert all("fixed controls differ: ['max_epochs/early_stopping']" in e for e in fulfillment['errors'])
    checkpoint = root/'experiment_logs/manager.pkl'
    state = pickle.loads(checkpoint.read_bytes())  # Own trusted checkpoint only.
    assert state['current_stage'] is None
    assert all(s.stage_number <= 3 for s in state['stages'])
    stage = next(s for s in state['stages'] if s.name.startswith('3_'))
    journal = state['journals'][stage.name]
    preserved = {n:pickle.dumps(j) for n,j in state['journals'].items() if n.startswith(('1_','2_'))}
    def policy(node):
        digest = hashlib.sha256(node.code.encode()).hexdigest()
        directory = root/'experiment_logs/evidence'/digest
        verified_metrics(directory,root/'dataset/research_view/dataset_profile.json',digest)
        return training_policy(read(directory/'experiment_manifest.json'))
    baseline = journal.nodes[0]
    reference = policy(baseline)
    assert reference is not None
    invalid = [n for n in journal.nodes[1:] if n.is_buggy is False]
    assert invalid and all(policy(n) != reference for n in invalid)
    for node in invalid:
        node.is_buggy = True
        node.exc_type = 'ComparisonPolicyError'
        node.analysis = f'Completed training retained but excluded from fair comparison: final policy must equal {reference}. Reserve final cap first; shorten search instead.'
        node._term_out = [node.analysis]
    state['current_stage'] = stage
    state['current_stage_number'] = 3
    stage.max_iterations = len(journal.nodes) + 3
    stage.goals = POLICIES[3].prompt()
    assert all(pickle.dumps(state['journals'][n]) == raw for n,raw in preserved.items())
    report = {'applied':args.apply,'excluded_nodes':[n.id for n in invalid],
              'locked_policy':reference,'new_attempt_limit':3,'training_started':False}
    if args.apply:
        backup = root/'experiment_logs/recovery_backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ-comparison')
        backup.mkdir(parents=True)
        for path in [checkpoint,root/'task.json',root/'web_run.json',root/'agent_progress.json',root/'research/contract_fulfillment.json']:
            shutil.copy2(path,backup/path.name)
            assert sha(path) == sha(backup/path.name)
        bind_policy(root,baseline.code)
        runner = AIScientistExperimentRunner(project,object(),object())
        cls,*_ = runner._runtime_classes(root/'dataset/research_view')
        manager = object.__new__(cls)
        manager.__dict__.update(state)
        assert manager._check_stage_completion(stage)[0] is False
        temporary = checkpoint.with_suffix('.tmp')
        temporary.write_bytes(pickle.dumps(state))
        temporary.replace(checkpoint)
        task['completed_stage'] = 'baseline_tuning_completed'
        task['stages']['creative_research_completed'] = 'waiting'
        task['stages']['ablations_completed'] = 'waiting'
        (root/'task.json').write_text(json.dumps(task,ensure_ascii=False,indent=2),encoding='utf-8')
        status.update(state='interrupted',message='Comparison recovery prepared; baseline retained, stage 3 pending manual restart.',updated_at=datetime.now(timezone.utc).isoformat())
        (root/'web_run.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
        _write_agent_progress(root,stage,journal,state['stage_history'])
        report['backup'] = str(backup.relative_to(root))
        (backup/'recovery_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
