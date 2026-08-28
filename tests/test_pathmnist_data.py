import pytest

from pathmnist.config import load_config
from pathmnist.data import DataValidationError, validate_dataset


def test_config_and_dataset_validation(project_root):
    if not (project_root / "pathmnist_64.npz").is_file():
        pytest.skip("manual real-PathMNIST integration test; dataset is intentionally absent from Git")
    original = load_config(project_root / "configs/pathmnist_m4.yaml")
    dataset = type(original.dataset)(
        path=project_root / original.dataset.path,
        sha256=original.dataset.sha256,
        expected_splits=original.dataset.expected_splits,
        classes=original.dataset.classes,
    )
    summary = validate_dataset(dataset)
    assert summary.sha256 == original.dataset.sha256
    assert summary.splits == {"train": 89996, "val": 10004, "test": 7180}
    assert all(len(counts) == 9 for counts in summary.classes.values())


def test_dataset_hash_mismatch_rejected(project_root):
    if not (project_root / "pathmnist_64.npz").is_file():
        pytest.skip("manual real-PathMNIST integration test; dataset is intentionally absent from Git")
    original = load_config(project_root / "configs/pathmnist_m4.yaml")
    dataset = type(original.dataset)(
        path=project_root / original.dataset.path,
        sha256="0" * 64,
        expected_splits=original.dataset.expected_splits,
        classes=original.dataset.classes,
    )
    try:
        validate_dataset(dataset)
    except DataValidationError:
        return
    raise AssertionError("Hash mismatch was not rejected")
