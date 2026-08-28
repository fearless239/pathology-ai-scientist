from types import SimpleNamespace

import json

from pathmnist.autonomous_export import (
    _augment_compute_metrics,
    _metric_dict,
    _select_complete_result,
)


def test_flattens_arbitrary_upstream_metric_bundle():
    node = SimpleNamespace(metric=SimpleNamespace(value={"metric_names": [{"metric_name": "Validation Macro-F1", "data": [{"dataset_name": "generic", "final_value": 0.8}]}]}))
    assert _metric_dict(node) == {"validation_macro_f1": 0.8}


def test_derives_upgrade_ratio_from_agent_recorded_relative_flops(tmp_path):
    import numpy as np

    path = tmp_path / "experiment_data.npy"
    np.save(path, {"generic": {"routing_ratio": [1.75]}})
    metrics = {}
    _augment_compute_metrics(metrics, path)
    assert metrics["upgrade_ratio"] == 0.25


def test_complete_working_result_is_preferred_over_scalar_summary(tmp_path):
    source = tmp_path / "result"
    working = source / "working"
    working.mkdir(parents=True)
    (source / "experiment_result.json").write_text(
        json.dumps({"metrics": {"accuracy": 0.9}}), encoding="utf-8"
    )
    complete = {
        "code_sha256": "a" * 64,
        "metrics": {"accuracy": 0.8},
        "predictions": [0, 1],
        "targets": [0, 1],
        "sample_ids": ["a", "b"],
    }
    expected = working / "experiment_result.json"
    expected.write_text(json.dumps(complete), encoding="utf-8")
    selected, path = _select_complete_result(source, "a" * 64)
    assert selected == complete
    assert path == expected


def test_complete_result_with_different_code_hash_is_rejected(tmp_path):
    source = tmp_path / "result/working"
    source.mkdir(parents=True)
    (source / "experiment_result.json").write_text(json.dumps({
        "code_sha256": "b" * 64,
        "metrics": {"accuracy": 0.8},
        "predictions": [0], "targets": [0], "sample_ids": ["a"],
    }), encoding="utf-8")
    selected, path = _select_complete_result(source.parent, "a" * 64)
    assert selected is None
    assert path is None


def test_complete_result_can_be_recovered_recursively_by_code_hash(tmp_path):
    root = tmp_path / "experiment_workspace"
    path = root / "process-3/working/experiment_result.json"
    path.parent.mkdir(parents=True)
    complete = {
        "code_sha256": "c" * 64,
        "metrics": {"accuracy": 0.7},
        "predictions": [1], "targets": [1], "sample_ids": ["sample"],
    }
    path.write_text(json.dumps(complete), encoding="utf-8")
    selected, selected_path = _select_complete_result(root, "c" * 64)
    assert selected == complete
    assert selected_path == path
