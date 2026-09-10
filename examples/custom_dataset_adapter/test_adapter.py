"""Replace DATASET_PATH before running this opt-in conformance check."""

import os
from pathlib import Path

import pytest

from path_ai_scientist.adapters import load_dataset_adapter, validate_dataset_adapter


def test_custom_dataset_adapter(tmp_path):
    source = os.environ.get("PATH_AI_CUSTOM_DATASET")
    if not source:
        pytest.skip("set PATH_AI_CUSTOM_DATASET to run this trusted-adapter check")
    adapter = load_dataset_adapter(
        "examples.custom_dataset_adapter.adapter:CustomDatasetAdapter", seed=7
    )
    report = validate_dataset_adapter(adapter, Path(source), tmp_path / "adapter-check")
    assert report["passed"] is True
    assert len(report["classes"]) == 2
