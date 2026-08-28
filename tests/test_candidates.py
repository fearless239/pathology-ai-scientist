
import pytest

from pathmnist.candidates import CandidateError, OneTimeTestEvaluator, approve_test_evaluation, evidence_value, freeze_candidate, select_validation_candidate
from pathmnist.experiment_contract import ExperimentResult, code_sha256


def _result(path, code, name, value, metric="custom_score"):
    return ExperimentResult(name, code_sha256(code), 7, {metric: value}, {}, {"code": "run.py"}).write(path)


def test_selects_any_named_validation_metric_and_freezes_exact_code(tmp_path):
    code = "print('agent generated')\n"
    code_path = tmp_path / "run.py"
    code_path.write_text(code)
    first = _result(tmp_path / "a.json", code, "a", 2.0)
    second = _result(tmp_path / "b.json", code, "b", 3.0)
    selected_id, selected_path, result = select_validation_candidate([("a", first), ("b", second)], "custom_score")
    assert selected_id == "b"
    frozen = freeze_candidate(selected_id, selected_path, code_path, tmp_path / "frozen", "custom_score")
    assert frozen.method_name == "b"
    assert frozen.snapshot_sha256 == frozen.code_sha256


def test_one_time_test_attempt_is_irreversible_and_evidence_is_traceable(tmp_path):
    code = "print('frozen')\n"
    code_path = tmp_path / "run.py"
    code_path.write_text(code)
    validation = _result(tmp_path / "validation.json", code, "method", 0.7)
    frozen = freeze_candidate("node-1", validation, code_path, tmp_path / "candidate", "custom_score")
    final = tmp_path / "final"
    approve_test_evaluation(final, frozen)
    sealed = tmp_path / "sealed_test"
    sealed.mkdir()
    evaluator = OneTimeTestEvaluator(final)

    def execute(snapshot, test_view):
        return ExperimentResult("method", code_sha256(snapshot.read_text()), 7, {"custom_score": 0.6}, {}, {}, split="test", test_data_accessed=True)

    evaluator.evaluate(tmp_path / "candidate", execute, sealed)
    with pytest.raises(CandidateError, match="already attempted"):
        evaluator.evaluate(tmp_path / "candidate", execute, sealed)
    evidence = evidence_value(final / "experiment_result.json", "custom_score", allow_test=True)
    assert evidence["json_pointer"] == "/metrics/custom_score"
    assert len(evidence["artifact_sha256"]) == 64


def test_modified_snapshot_is_rejected_before_test_execution(tmp_path):
    code = "print(1)\n"
    source = tmp_path / "run.py"
    source.write_text(code)
    validation = _result(tmp_path / "validation.json", code, "method", 1.0)
    frozen = freeze_candidate("node", validation, source, tmp_path / "candidate", "custom_score")
    approve_test_evaluation(tmp_path / "final", frozen)
    (tmp_path / "candidate" / "run.py").chmod(0o644)
    (tmp_path / "candidate" / "run.py").write_text("print(2)\n")
    with pytest.raises(CandidateError, match="modified"):
        OneTimeTestEvaluator(tmp_path / "final").evaluate(tmp_path / "candidate", lambda *_: None, tmp_path)


def test_completed_test_recovers_without_reexecution_and_detects_tampering(tmp_path):
    code = 'print(1)\n'
    source = tmp_path / 'run.py'
    source.write_text(code)
    validation = _result(tmp_path / 'validation.json', code, 'method', 0.7)
    frozen = freeze_candidate('node', validation, source, tmp_path / 'candidate', 'custom_score')
    approve_test_evaluation(tmp_path / 'final', frozen)
    evaluator = OneTimeTestEvaluator(tmp_path / 'final')
    calls = []
    def execute(*args):
        calls.append(1)
        return ExperimentResult('method', code_sha256(code), 7, {'custom_score': 0.6}, {}, {}, split='test', test_data_accessed=True)
    evaluator.evaluate(tmp_path / 'candidate', execute, tmp_path)
    assert evaluator.recover(tmp_path / 'candidate').metrics['custom_score'] == 0.6
    assert calls == [1]
    (tmp_path / 'final/experiment_result.json').write_text('{}')
    with pytest.raises(CandidateError, match='hash mismatch'):
        evaluator.recover(tmp_path / 'candidate')
