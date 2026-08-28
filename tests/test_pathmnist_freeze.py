import json
from pathlib import Path

import pytest

from pathmnist.freeze import load_frozen_candidate
from pathmnist.reporting import ReportError, generate_m4_report
from pathmnist.freeze import (
    FreezeError,
    REQUIRED_APPROVAL,
    require_ready_checkpoints,
    write_once,
)


def test_approved_candidate_is_frozen(project_root):
    candidate = load_frozen_candidate(
        project_root / "configs/pathmnist_final_candidate.json"
    )
    assert candidate.variant == "optimization"
    assert candidate.seeds == (7, 17, 27)
    assert candidate.learning_rate == pytest.approx(0.001)
    assert candidate.weight_decay == pytest.approx(1e-05)


def test_checkpoint_readiness_requires_all_seeds(tmp_path):
    candidate = load_frozen_candidate(Path("configs/pathmnist_final_candidate.json"))
    with pytest.raises(FreezeError, match="incomplete"):
        require_ready_checkpoints(candidate, candidate.dataset_sha256, tmp_path)


def test_evaluation_output_is_written_once(tmp_path):
    output = tmp_path / "evaluation.json"
    write_once(output, {"value": 1})
    assert json.loads(output.read_text()) == {"value": 1}
    with pytest.raises(FreezeError, match="already exists"):
        write_once(output, {"value": 2})


def test_required_approval_is_explicit():
    assert REQUIRED_APPROVAL == "I APPROVE ONE-TIME TEST EVALUATION"


def test_report_requires_exactly_one_test_evaluation(tmp_path):
    (tmp_path / "test_evaluation.json").write_text(
        json.dumps({"evaluation_count": 2, "split": "test"}), encoding="utf-8"
    )
    with pytest.raises(ReportError, match="exactly one"):
        generate_m4_report(tmp_path, Path("configs/pathmnist_final_candidate.json"), tmp_path / "report.md")
