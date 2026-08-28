from pathlib import Path

import pytest

from pathmnist.dataset_adapter import DatasetAdapter as ConcreteDatasetAdapter
from pathmnist.framework import DatasetAdapter, ResearchTaskConfig, RunPermissions


def test_reference_dataset_adapter_satisfies_public_protocol():
    assert isinstance(ConcreteDatasetAdapter(), DatasetAdapter)


def test_research_task_config_is_provider_neutral(tmp_path: Path):
    config = ResearchTaskConfig(
        task_id="task-1",
        direction="Compare a baseline and intervention",
        dataset_adapter="example.Adapter",
        dataset_path=tmp_path / "dataset",
        output_root=tmp_path / "output",
        permissions=RunPermissions(allow_sealed_test=True),
    )
    value = config.as_dict()
    assert value["permissions"]["allow_sealed_test"] is True
    assert value["permissions"]["allow_paid_llm"] is False
    assert value["dataset_adapter"] == "example.Adapter"


def test_research_task_config_rejects_unsafe_values(tmp_path: Path):
    with pytest.raises(ValueError, match="budget_usd"):
        ResearchTaskConfig("task", "direction", "adapter", tmp_path, tmp_path, budget_usd=-1)
