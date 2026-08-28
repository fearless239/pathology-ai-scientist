import json

import numpy as np

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
    with np.load(tmp_path / "state/v2-task/dataset/research_view/dataset.npz") as view:
        assert "test_images" not in view.files
    monkeypatch.setattr("sys.argv", ["pathmnist", "autonomous-init", "--resume", "--state-root", str(tmp_path / "state"), "--task-id", "v2-task", "--dataset-path", str(dataset), "--direction", "Study adaptive computation"])
    assert main() == 0
