import json

import pytest

from pathmnist.autonomous_orchestrator import AutonomousOrchestrator, OrchestrationError
from pathmnist.autonomous_stages import V2_STAGES


def _task(root, *, schema_version=2):
    stages = {stage: "waiting" for stage in V2_STAGES}
    for stage in V2_STAGES[: V2_STAGES.index("dataset_validated") + 1]:
        stages[stage] = "completed"
    path = root / "demo/task.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": schema_version, "task_id": "demo", "completed_stage": "dataset_validated", "control": "paused", "stages": stages}))
    profile = root / "demo/dataset/dataset_profile.json"
    profile.parent.mkdir(parents=True)
    profile.write_text("{}")
    view = root / "demo/dataset/research_view/dataset.npz"
    view.parent.mkdir(parents=True)
    view.write_bytes(b"fixture")
    (view.parent / 'dataset_profile.json').write_text('{}')


def test_status_identifies_the_next_formal_stage(tmp_path):
    _task(tmp_path)
    status = AutonomousOrchestrator(tmp_path, tmp_path, "demo").status()
    assert status["valid"]
    assert status["next_stage"] == "research_understood"


def test_legacy_task_is_rejected_by_public_control_plane(tmp_path):
    _task(tmp_path, schema_version=1)
    with pytest.raises(OrchestrationError, match="Legacy"):
        AutonomousOrchestrator(tmp_path, tmp_path, "demo").status()


def test_failure_preserves_diagnosis_without_committing_bad_stage(tmp_path, monkeypatch):
    from pathmnist.autonomous_orchestrator import StageHandler
    _task(tmp_path)
    path = tmp_path / 'demo/task.json'
    orchestrator = AutonomousOrchestrator(tmp_path, tmp_path, 'demo')
    def failed():
        task = json.loads(path.read_text())
        task.update(publication_mode='failure_diagnosis', completed_stage='test_evaluated')
        path.write_text(json.dumps(task))
        raise RuntimeError('injected failure')
    monkeypatch.setattr(orchestrator, '_handler', lambda **kw: StageHandler('test', lambda: None, failed, lambda: None))
    with pytest.raises(RuntimeError, match='injected'):
        orchestrator.run()
    task = json.loads(path.read_text())
    assert task['completed_stage'] == 'dataset_validated'
    assert task['publication_mode'] == 'failure_diagnosis'
    assert task['control'] == 'interrupted'
