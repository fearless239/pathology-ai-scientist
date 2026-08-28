import json
from types import SimpleNamespace

from pathmnist.autonomous import AutonomousTaskWorkspace, GatewayQueryAdapter, V2_STAGES, pathology_task_description
from pathmnist.dataset_adapter import DatasetSpec, SampleRecord
from pathmnist.research_contract import generate_contract


def _spec(tmp_path):
    samples = [SampleRecord(f"{split}-{i}", str(tmp_path / "data.npz"), str(i % 2), split, array_key=f"{split}_images", index=i) for split in ("train", "validation", "test") for i in range(2)]
    return DatasetSpec(2, "generic", "npz", str(tmp_path / "data.npz"), "a" * 64, [8, 8, 3], 3, ["0", "1"], {"0": 0, "1": 1}, {"train": 2, "validation": 2, "test": 2}, {}, samples)


def test_v2_workspace_layout_and_stage_contract(tmp_path):
    workspace = AutonomousTaskWorkspace.create(tmp_path, "task-1")
    assert workspace.experiment_workspace.is_dir()
    assert workspace.final_evaluation.is_dir()
    assert V2_STAGES.index("candidate_frozen") < V2_STAGES.index("test_evaluated")


def test_task_description_contains_data_facts_but_not_fixed_model_or_variant(tmp_path):
    payload = pathology_task_description("Investigate adaptive resolution", _spec(tmp_path))
    parsed = json.loads(payload)
    assert parsed["Short Hypothesis"] == "Investigate adaptive resolution"
    assert parsed["Dataset Profile"]["split_counts"] == {"train": 2, "validation": 2}
    assert "SmallResNet" not in payload
    assert "optimization" not in payload.casefold()
    assert '"test": 2' not in payload
    assert parsed["Research View Interface"]["image_array_layout"] == "NHWC"


def test_contracted_task_description_has_distinct_scientific_fields(tmp_path):
    direction = (
        "基于PathMNIST的数据增强方法优化。以ResNet-18为基线，引入颜色扰动、"
        "旋转和翻转，目标使九分类准确率提高至少2个百分点。"
    )
    spec = _spec(tmp_path)
    contract = generate_contract(direction, spec.to_dict(), split_seed=7)
    parsed = json.loads(pathology_task_description(direction, spec, contract))

    assert parsed["Title"].startswith("PathMNIST classification")
    assert parsed["Abstract"] != direction
    assert parsed["Short Hypothesis"] != direction
    assert parsed["Abstract"] != parsed["Short Hypothesis"]
    assert "2.0 percentage points" in parsed["Short Hypothesis"]
    assert "paired seeds" in parsed["Short Hypothesis"]
    assert direction not in "\n".join(parsed["Experiments"])
    assert "one host-injected seed" in parsed["Experiments"][1]


def test_gateway_adapter_routes_structured_calls_through_provider_ledger():
    calls = []

    class Provider:
        def call_json(self, *args):
            calls.append(args)
            return {"ok": True}, {}

    function = SimpleNamespace(name="result", json_schema={"type": "object"})
    adapter = GatewayQueryAdapter(Provider(), "task")
    assert adapter(user_message={"x": 1}, func_spec=function) == {"ok": True}
    assert calls[0][0] == "ideation"


def test_gateway_adapter_continues_call_sequence_from_ledger(tmp_path):
    ledger_path = tmp_path / "budget.json"
    ledger_path.write_text(json.dumps({
        "requests": {
            "task-agent-v2-old-out12000-call9": {"state": "settled"},
            "other-agent-v2-old-out12000-call99": {"state": "settled"},
        }
    }), encoding="utf-8")
    calls = []

    class Provider:
        ledger = SimpleNamespace(path=ledger_path)

        def call_text(self, role, request_id, system, prompt):
            calls.append(request_id)
            return "ok", {}

    assert GatewayQueryAdapter(Provider(), "task")() == "ok"
    assert len(calls) == 1
    assert calls[0].startswith("task-agent-v2-")
    assert calls[0].endswith("-outdefault-call10")



def test_task_description_matches_materialized_source_aliases(tmp_path):
    from dataclasses import replace
    import numpy as np
    from pathmnist.dataset_adapter import materialize_split_view

    for alias in ("val", "valid", "validation"):
        spec = _spec(tmp_path)
        spec.samples = [
            replace(sample, array_key=f"{alias}_images")
            if sample.split == "validation" else sample for sample in spec.samples
        ]
        np.savez(spec.source_path, **{
            "train_images": np.zeros((2, 8, 8, 3), dtype=np.uint8),
            f"{alias}_images": np.ones((2, 8, 8, 3), dtype=np.uint8),
        })
        view = materialize_split_view(spec, tmp_path / alias, {"train", "validation"})
        payload = json.loads(pathology_task_description("Study augmentation", spec, research_view=view))
        with np.load(view / "dataset.npz", allow_pickle=False) as data:
            assert payload["Research View Interface"]["array_keys"] == sorted(data.files)
            assert data["validation_images"].sum() == 2 * 8 * 8 * 3
        assert "validation_sample_ids" in payload["Research View Interface"]["array_keys"]
        assert not any(key.startswith("test") for key in payload["Research View Interface"]["array_keys"])


def test_task_description_rejects_mounted_interface_drift(tmp_path):
    import numpy as np
    import pytest
    from pathmnist.dataset_adapter import DatasetDiscoveryError

    np.savez(tmp_path / "dataset.npz", val_images=np.zeros((2, 8, 8, 3)))
    with pytest.raises(DatasetDiscoveryError, match="interface mismatch"):
        pathology_task_description("Study augmentation", _spec(tmp_path), research_view=tmp_path)


def test_manifest_task_does_not_advertise_npz_arrays(tmp_path):
    spec = _spec(tmp_path)
    spec.source_type = "imagefolder"
    interface = json.loads(pathology_task_description("Study augmentation", spec))["Research View Interface"]
    assert interface["manifest_path"] == "/dataset/manifest.json"
    assert "npz_path" not in interface
    assert "array_keys" not in interface
