import json

from pathmnist.autonomous_stages import V2_STAGES
from pathmnist.autonomous_acceptance import validate_task
from pathmnist.research_contract import generate_contract, write_contract
from pathmnist.figures import _score_comparison, generate_template_figures
from pathmnist.paper_export import markdown_to_latex


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_template_figure_manifest_is_evidence_backed(tmp_path, monkeypatch):
    stages = {stage: "waiting" for stage in V2_STAGES}
    end = V2_STAGES.index("analysis_completed")
    stages.update({stage: "completed" for stage in V2_STAGES[: end + 1]})
    _json(tmp_path / "task.json", {"schema_version": 2, "task_id": "demo", "completed_stage": "analysis_completed", "stages": stages})
    contract = generate_contract("比较增强方法的宏平均F1", {"classes": ["a", "b"]})
    write_contract(tmp_path, contract)
    _json(tmp_path / "research/research_contract_approval.json", {"schema_version": 1, "approved": True, "contract_sha256": contract["contract_sha256"]})
    fulfillment = {"schema_version": 1, "passed": True, "checks": {"comparison_integrity": "passed", "repeat_integrity": "passed", "statistical_integrity": "passed"}, "errors": []}
    _json(tmp_path / "research/contract_fulfillment.json", fulfillment)
    _json(tmp_path / "research/semantic_review.json", {"schema_version": 1, "contract_sha256": contract["contract_sha256"], "passed": True})
    _json(tmp_path / "candidate_frozen/comparison_bundle.json", {"schema_version": 1, "experiments": []})
    _json(tmp_path / "candidate_frozen/contract_fulfillment.json", fulfillment)
    arms = [
        {"role_id": role, "seed": seed, "trusted_metrics": {"macro_f1": 0.7 + (0.1 if role == "proposed_method" else 0.0)}}
        for role in ("baseline", "proposed_method") for seed in (0, 1, 2)
    ]
    _json(tmp_path / "final_evaluation/comparison_results.json", {"schema_version": 1, "attempt_count": 1, "split": "test", "primary_metric": "macro_f1", "arms": arms, "statistics": {"n": 3, "mean_baseline": 0.7, "mean_intervention": 0.8, "confidence_interval_95": [0.05, 0.15]}})
    _json(tmp_path / "dataset/dataset_profile.json", {"split_counts": {"train": 8, "validation": 2}, "class_counts": {"train": {"a": 4, "b": 4}}, "classes": ["a", "b"]})
    (tmp_path / "dataset/research_view").mkdir(parents=True)
    (tmp_path / "dataset/research_view/dataset.npz").write_bytes(b"fixture")
    _json(tmp_path / 'dataset/research_view/dataset_profile.json', {})
    for relative in (
        "research/research_understanding.json", "research/idea.json",
        "research/preflight.json", "candidates/selection.json", "candidate_frozen/candidate.json",
        "candidate_frozen/run.py", "candidate_frozen/experiment_manifest.json",
        "final_evaluation/approval.json", "paper/analysis_completed/analysis.json",
    ):
        _json(tmp_path / relative, {})
    _json(tmp_path / "research/experiment_spec.json", {"research_contract_sha256": contract["contract_sha256"], "primary_metric": "macro_f1"})
    receipt = {"schema_version": 1, "recorded_by": "trusted-runner", "samples_consumed": 1, "sample_ids_consumed": ["x"], "dataset_profile_sha256": "a" * 64, "code_sha256": "b" * 64}
    _json(tmp_path / "candidate_frozen/dataset_execution_receipt.json", {**receipt, "split": "validation"})
    _json(tmp_path / "final_evaluation/integrity/dataset_execution_receipt.json", {**receipt, "split": "test"})
    _json(tmp_path / "final_evaluation/integrity/trusted_metrics.json", {"evaluator_version": "p0-classification-v1", "metrics": {"accuracy": 0.8}})
    _json(tmp_path / "final_evaluation/integrity/metric_provenance.json", {"evaluator_version": "p0-classification-v1"})
    _json(tmp_path / "research/literature.json", {"status": "verified", "references": [{"title": "Paper", "doi": "10.1/example"}]})
    _json(tmp_path / "candidate_frozen/validation_result.json", {"metrics": {"macro_f1": 0.7}})
    _json(tmp_path / "final_evaluation/experiment_result.json", {"metrics": {"test_accuracy": 0.8, "confusion_matrix": [[4, 1], [0, 5]]}})

    def fake_bar(path, *args, **kwargs):
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (10).to_bytes(4, "big") + (10).to_bytes(4, "big"))

    monkeypatch.setattr("pathmnist.figures._save_bar", fake_bar)
    monkeypatch.setattr("pathmnist.figures._save_confusion", fake_bar)
    manifest = generate_template_figures(tmp_path)
    assert manifest["figures"]
    assert all(item["source_artifacts"] for item in manifest["figures"])
    assert validate_task(tmp_path, "figures_generated").passed


def test_markdown_images_are_rendered_safely():
    latex = markdown_to_latex("# Study\n\n![Confusion matrix](figures/confusion.png)\n")
    assert "\\includegraphics" in latex
    assert "figures/confusion.png" in latex


def test_score_comparison_supports_explicit_condition_suffixes():
    metrics = {
        "final_val_macro_f1_no_aug": 0.81,
        "final_val_macro_f1_with_aug": 0.84,
        "improvement_best_val_macro_f1": 0.03,
        "validation_loss": 0.2,
    }
    assert _score_comparison(metrics) == {
        "final_val_macro_f1_no_aug": 0.81,
        "final_val_macro_f1_with_aug": 0.84,
    }
