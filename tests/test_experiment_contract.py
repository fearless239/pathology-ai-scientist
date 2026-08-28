import json

import pytest

from pathmnist.experiment_contract import (
    ExperimentContractError,
    ExperimentResult,
    canonicalize_metrics,
    code_sha256,
)


def test_arbitrary_metric_dictionary_is_preserved(tmp_path):
    code = "print('generated')\n"
    result = ExperimentResult("agent-created method", code_sha256(code), 7, {"macro_f1": 0.8, "average_flops": 123.0, "upgrade_ratio": 0.2}, {"elapsed_seconds": 1.2}, {"code": "run.py"})
    path = result.write(tmp_path / "experiment_result.json")
    assert json.loads(path.read_text())["metrics"]["upgrade_ratio"] == 0.2


def test_research_result_cannot_claim_test_access(tmp_path):
    result = ExperimentResult("bad", "0" * 64, 7, {"accuracy": 1}, {}, {}, split="test", test_data_accessed=True)
    with pytest.raises(ExperimentContractError, match="test"):
        result.write(tmp_path / "result.json")


def test_test_metric_aliases_are_migrated_without_ambiguity():
    assert canonicalize_metrics(
        {"validation_accuracy": 0.8, "validation_inference_seconds": 1.2}, "test"
    ) == {"test_accuracy": 0.8, "dynamic_inference_seconds": 1.2}

    assert canonicalize_metrics(
        {"best_val_macro_f1_no_aug": 0.81, "final_validation_loss": 0.2}, "test"
    ) == {"best_test_macro_f1_no_aug": 0.81, "final_test_loss": 0.2}


def test_schema_v2_rejects_validation_names_on_test_result(tmp_path):
    result = ExperimentResult(
        "method", "0" * 64, 7, {"validation_accuracy": 0.8}, {}, {},
        split="test", test_data_accessed=True,
    )
    with pytest.raises(ExperimentContractError, match="legacy"):
        result.write(tmp_path / "result.json", allow_test=True)


def test_optional_sample_level_evidence_must_align():
    result = ExperimentResult(
        "method", "0" * 64, 7, {"accuracy": 0.8}, {}, {},
        predictions=[0, 1], targets=[0], probabilities=[[0.8, 0.2], [0.1, 0.9]],
        class_names=["a", "b"],
    )
    with pytest.raises(ExperimentContractError, match="align"):
        result.validate()
