import json

from pathmnist.autonomous_stages import V2_STAGES
from pathmnist.autonomous_acceptance import STAGE_ARTIFACTS, validate_task
from pathmnist.autonomous_postprocess import _analysis_claims


def _write(path, value=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def test_archive_rejects_skipped_research_stages(tmp_path):
    stages = {stage: "completed" for stage in V2_STAGES}
    for stage in ("research_understood", "literature_collected", "idea_proposed"):
        stages[stage] = "waiting"
    (tmp_path / "task.json").write_text(
        json.dumps({"task_id": "demo", "completed_stage": "archived", "stages": stages})
    )
    report = validate_task(tmp_path, "archived", require_pdf=True)
    assert not report.passed
    assert any("literature_collected" in error for error in report.errors)


def test_formal_paper_requires_verified_identifiable_reference(tmp_path):
    stages = {stage: "waiting" for stage in V2_STAGES}
    end = V2_STAGES.index("literature_collected")
    stages.update({stage: "completed" for stage in V2_STAGES[: end + 1]})
    (tmp_path / "task.json").write_text(
        json.dumps({"task_id": "demo", "completed_stage": "literature_collected", "stages": stages})
    )
    _write(tmp_path / "dataset/dataset_profile.json", b"{}")
    _write(tmp_path / "dataset/research_view/dataset.npz")
    _write(tmp_path / "research/research_understanding.json", b"{}")
    _write(
        tmp_path / "research/literature.json",
        json.dumps({"status": "verified", "references": [{"title": "A paper"}]}).encode(),
    )
    report = validate_task(tmp_path, "literature_collected")
    assert not report.passed
    assert any("stable identifier" in error for error in report.errors)


def test_analysis_claims_do_not_assert_unmeasured_superiority():
    finding, limitations = _analysis_claims(
        {"primary_metric": "accuracy"},
        {
            "test_accuracy": 0.85,
            "dynamic_inference_seconds": 0.60,
            "fixed_high_inference_seconds": 0.57,
        },
    )
    assert "not evidence of superiority" in finding
    assert any("not lower than" in limitation for limitation in limitations)


def test_analysis_claims_map_validation_primary_metric_to_trusted_test_metric():
    finding, _ = _analysis_claims(
        {"primary_metric": "validation_macro_f1"},
        {"test_accuracy": 0.95, "test_macro_f1": 0.81},
    )
    assert "test_macro_f1=0.810000" in finding


def test_formal_analysis_rejects_missing_integrity_evidence(tmp_path):
    stages = {stage: "waiting" for stage in V2_STAGES}
    end = V2_STAGES.index("analysis_completed")
    stages.update({stage: "completed" for stage in V2_STAGES[: end + 1]})
    (tmp_path / "task.json").write_text(
        json.dumps({"schema_version": 2, "task_id": "unsafe", "completed_stage": "analysis_completed", "stages": stages})
    )
    for stage in V2_STAGES[: end + 1]:
        for relative in STAGE_ARTIFACTS.get(stage, ()):
            _write(tmp_path / relative, b"{}")
    report = validate_task(tmp_path, "analysis_completed")
    assert not report.passed
    assert report.publication_mode == "failure_diagnosis"
    assert report.data_integrity == "failed"
    assert report.sealed_test_integrity == "failed"
    assert report.metric_integrity == "failed"
