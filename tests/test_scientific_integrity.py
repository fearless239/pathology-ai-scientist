import json

import pytest

from pathmnist.scientific_integrity import (
    IntegrityError,
    TrustedMetricEvaluator,
    record_trusted_evaluation,
    validate_no_synthetic_dataset,
)


def _profile(path):
    value = {
        "schema_version": 1,
        "classes": ["a", "b"],
        "label_mapping": {"a": 0, "b": 1},
        "samples": [
            {"id": "validation:0", "path": "data.npz", "label": "a", "split": "validation", "array_key": "validation_images", "index": 0},
            {"id": "validation:1", "path": "data.npz", "label": "b", "split": "validation", "array_key": "validation_images", "index": 1},
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_trusted_metrics_are_recomputed_from_per_sample_outputs():
    metrics = TrustedMetricEvaluator().evaluate([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 0], [1, 2]]
    assert metrics["per_class"][0]["precision"] == 0.5
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_receipt_checks_targets_and_records_reported_metric_mismatch(tmp_path):
    profile = _profile(tmp_path / "profile.json")
    receipt, metrics, provenance = record_trusted_evaluation(
        profile_path=profile,
        split="validation",
        sample_ids=["validation:0", "validation:1"],
        targets=[0, 1],
        predictions=[0, 0],
        probabilities=None,
        code_sha256="a" * 64,
        output_dir=tmp_path / "integrity",
        reported_metrics={"accuracy": 1.0},
    )
    assert receipt.recorded_by == "trusted-runner"
    assert metrics["accuracy"] == 0.5
    assert provenance["metric_mismatches"]["accuracy"]["trusted"] == 0.5
    assert (tmp_path / "integrity/trusted_metrics.json").is_file()


@pytest.mark.parametrize(
    "sample_ids,targets,error",
    [
        (["validation:0", "validation:0"], [0, 0], "Duplicate"),
        (["validation:0", "unknown"], [0, 1], "Unknown"),
        (["validation:0", "validation:1"], [1, 1], "Target does not match"),
    ],
)
def test_receipt_rejects_untrusted_sample_evidence(tmp_path, sample_ids, targets, error):
    with pytest.raises(IntegrityError, match=error):
        record_trusted_evaluation(
            profile_path=_profile(tmp_path / "profile.json"), split="validation",
            sample_ids=sample_ids, targets=targets, predictions=[0, 1], probabilities=None,
            code_sha256="b" * 64, output_dir=tmp_path / "integrity",
        )


def test_probability_rows_must_match_classes_and_be_normalized(tmp_path):
    with pytest.raises(IntegrityError, match="not normalized"):
        record_trusted_evaluation(
            profile_path=_profile(tmp_path / "profile.json"), split="validation",
            sample_ids=["validation:0", "validation:1"], targets=[0, 1], predictions=[0, 1],
            probabilities=[[0.9, 0.9], [0.1, 0.9]], code_sha256="c" * 64,
            output_dir=tmp_path / "integrity",
        )


def test_synthetic_policy_blocks_dataset_replacement_but_allows_augmentation():
    with pytest.raises(IntegrityError, match="Randomly generated data"):
        validate_no_synthetic_dataset("import numpy as np\ntrain_data = np.random.rand(10, 4)\n")
    with pytest.raises(IntegrityError, match="FakeData"):
        validate_no_synthetic_dataset("from torchvision.datasets import FakeData\ndataset = FakeData()\n")
    with pytest.raises(IntegrityError, match="download"):
        validate_no_synthetic_dataset("dataset = ImageFolder('/tmp', download=True)\n")
    with pytest.raises(IntegrityError, match="Custom Dataset"):
        validate_no_synthetic_dataset(
            "class Lazy(Dataset):\n    def __getitem__(self, i):\n        return torch.rand(3, 8, 8), 0\n"
        )
    validate_no_synthetic_dataset(
        "import numpy as np\nimages = load_dataset('/dataset')\nnoise = np.random.normal(0, .1, images.shape)\naugmented = images + noise\n"
    )
