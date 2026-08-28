from pathmnist.publication import (
    normalize_publication_language,
    publication_candidate,
    publication_dataset_profile,
    publication_integrity_summary,
)


def test_publication_profile_excludes_audit_fields() -> None:
    result = publication_dataset_profile({"name": "pathmnist_64", "content_sha256": "abc", "source_type": "npz", "classes": 9, "split_counts": {"train": 1}})
    assert 'Resizing the supplied images is allowed' in result.pop('preprocessing_reporting_guidance')
    assert result == {"dataset_name": "pathmnist", "classes": 9, "split_counts": {"train": 1}}
    assert "content_sha256" not in result


def test_integrity_summary_contains_no_receipt_details() -> None:
    result = publication_integrity_summary({"code_sha256": "secret"}, {"runner_identity": "runner"}, {"evaluator_version": "v1"})
    rendered = str(result)
    assert "secret" not in rendered
    assert "runner_identity" not in rendered
    assert result["metrics_independently_recomputed"] is True


def test_internal_workflow_terms_are_normalized() -> None:
    result = normalize_publication_language("A frozen candidate was evaluated on the sealed test set by the trusted evaluator.")
    assert "frozen candidate" not in result.lower()
    assert "sealed test" not in result.lower()
    assert "trusted evaluator" not in result.lower()


def test_publication_candidate_excludes_internal_identity() -> None:
    result = publication_candidate({"experiment_id": "abc", "code_sha256": "secret", "frozen_at": "now", "primary_metric": "macro_f1", "validation_value": 0.9})
    assert result == {"primary_metric": "macro_f1", "validation_value": 0.9}
