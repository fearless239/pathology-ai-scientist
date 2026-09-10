import json

import numpy as np
import pytest

from pathmnist.cli import main


def test_autonomous_init_creates_schema_v2_task_without_test_arrays(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "generic.npz"
    rng = np.random.default_rng(4)
    np.savez(dataset, train_images=rng.integers(0, 255, (8, 4, 4, 3), dtype=np.uint8), train_labels=np.tile([0, 1], 4), val_images=rng.integers(0, 255, (4, 4, 4, 3), dtype=np.uint8), val_labels=np.tile([0, 1], 2), test_images=rng.integers(0, 255, (4, 4, 4, 3), dtype=np.uint8), test_labels=np.tile([0, 1], 2))
    monkeypatch.setattr("sys.argv", ["pathmnist", "autonomous-init", "--state-root", str(tmp_path / "state"), "--task-id", "v2-task", "--dataset-path", str(dataset), "--direction", "Study adaptive computation"])
    assert main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == 2
    task = json.loads((tmp_path / "state/v2-task/task.json").read_text())
    assert task["task_type"] == "autonomous_experiment"
    assert task["budget_limit_usd"] == 10.0
    assert task["dataset_adapter"] == "generic"
    # Resuming a historical task must not silently increase its authorization.
    task["budget_limit_usd"] = 8.0
    task.pop("dataset_adapter")
    (tmp_path / "state/v2-task/task.json").write_text(json.dumps(task))
    with np.load(tmp_path / "state/v2-task/dataset/research_view/dataset.npz") as view:
        assert "test_images" not in view.files
    monkeypatch.setattr("sys.argv", ["pathmnist", "autonomous-init", "--resume", "--state-root", str(tmp_path / "state"), "--task-id", "v2-task", "--dataset-path", str(dataset), "--direction", "Study adaptive computation"])
    assert main() == 0
    resumed = json.loads((tmp_path / "state/v2-task/task.json").read_text())
    assert resumed["budget_limit_usd"] == 8.0
    assert resumed["dataset_adapter"] == "generic"


def test_resume_rejects_dataset_adapter_change(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "generic.npz"
    images = np.zeros((4, 4, 4), dtype=np.uint8)
    labels = np.array([0, 1, 0, 1])
    np.savez(
        dataset,
        train_images=images,
        train_labels=labels,
        val_images=images,
        val_labels=labels,
        test_images=images,
        test_labels=labels,
    )
    command = [
        "pathmnist",
        "autonomous-init",
        "--state-root",
        str(tmp_path / "state"),
        "--task-id",
        "adapter-task",
        "--dataset-path",
        str(dataset),
        "--direction",
        "Binary classification",
    ]
    monkeypatch.setattr("sys.argv", command)
    assert main() == 0
    capsys.readouterr()
    monkeypatch.setattr(
        "sys.argv",
        [*command, "--resume", "--dataset-adapter", "custom.module:Adapter"],
    )
    with pytest.raises(RuntimeError, match="Existing task dataset adapter"):
        main()
