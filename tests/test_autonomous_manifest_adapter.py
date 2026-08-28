import json

from pathmnist.autonomous import _normalize_tuning_manifest
from pathmnist.experiment_manifest import load_manifest


def test_tuning_manifest_maps_only_explicit_aliases(tmp_path):
    path = tmp_path / "experiment_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "PathMNIST",
                "model": "SimpleCNN",
                "optimizer": "Adam",
                "best_learning_rate": 0.001,
                "epochs_per_candidate": 3,
                "batch_size": 128,
                "seed": 7,
                "input_resolutions": [64],
                "selection_metric": "macro_f1",
                "hardware": "cuda",
            }
        ),
        encoding="utf-8",
    )

    assert _normalize_tuning_manifest(path)
    manifest = load_manifest(path)
    assert manifest["learning_rate"] == 0.001
    assert manifest["epochs"] == 3


def test_tuning_manifest_does_not_invent_missing_values(tmp_path):
    path = tmp_path / "experiment_manifest.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    assert not _normalize_tuning_manifest(path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"schema_version": 1}


def test_ablation_manifest_maps_explicit_per_condition_epochs(tmp_path):
    path = tmp_path / "experiment_manifest.json"
    path.write_text(
        json.dumps({"schema_version": 1, "epochs_per_condition": 2, "total_epochs": 6}),
        encoding="utf-8",
    )

    assert _normalize_tuning_manifest(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["epochs"] == 2


def test_manifest_alias_requires_positive_integer(tmp_path):
    path = tmp_path / "experiment_manifest.json"
    path.write_text(
        json.dumps({"schema_version": 1, "epochs_per_condition": 0}), encoding="utf-8"
    )

    assert not _normalize_tuning_manifest(path)


def test_explicit_policy_never_infers_completed_epochs_from_budget(tmp_path):
    path = tmp_path / "experiment_manifest.json"
    value = {"max_epochs": 15, "epochs_per_candidate": 15, "early_stopping": {"enabled": False}}
    path.write_text(json.dumps(value))
    assert not _normalize_tuning_manifest(path)
    assert "epochs" not in json.loads(path.read_text())
