import csv
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pathmnist.dataset_adapter import (
    DatasetAdapter,
    DatasetAdapterLoadError,
    DatasetDiscoveryError,
    load_dataset_adapter,
    load_dataset_spec,
    materialize_split_view,
    normalize_inference_view_layout,
    validate_dataset_adapter,
)


def test_discovers_pathmnist_style_npz_without_name_or_fixed_keys(tmp_path):
    path = tmp_path / "anything.npz"
    rng = np.random.default_rng(3)
    np.savez(path, train_images=rng.integers(0, 255, (20, 12, 12, 3), dtype=np.uint8), train_labels=np.tile([2, 9], 10), val_images=rng.integers(0, 255, (6, 12, 12, 3), dtype=np.uint8), val_labels=np.tile([2, 9], 3), test_images=rng.integers(0, 255, (6, 12, 12, 3), dtype=np.uint8), test_labels=np.tile([2, 9], 3))
    profile = tmp_path / "dataset_profile.json"
    spec = DatasetAdapter().discover(path, profile)
    assert spec.classes == ["2", "9"]
    assert spec.image_shape == [12, 12, 3]
    assert spec.split_counts == {"train": 20, "validation": 6, "test": 6}
    assert json.loads(profile.read_text())["content_sha256"] == spec.content_sha256
    assert spec.content_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    research_view = materialize_split_view(spec, tmp_path / "research", {"train", "validation"})
    with np.load(research_view / "dataset.npz") as mounted:
        assert "test_images" not in mounted.files
        assert set(mounted.files) == {
            "train_images",
            "train_labels",
            "train_sample_ids",
            "validation_images",
            "validation_labels",
            "validation_sample_ids",
        }
    restored = load_dataset_spec(profile)
    assert restored.to_dict() == spec.to_dict()


def _image(path: Path, value: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 7, 3), value, dtype=np.uint8)).save(path)


def test_discovers_unsplit_imagefolder_and_generates_all_splits(tmp_path):
    for label in ("benign", "tumor"):
        for index in range(20):
            _image(tmp_path / label / f"{index}.png", index)
    spec = DatasetAdapter(seed=11).discover(tmp_path)
    assert set(spec.split_counts) == {"train", "validation", "test"}
    assert spec.label_mapping == {"benign": 0, "tumor": 1}


def test_manifest_group_split_keeps_patients_together(tmp_path):
    rows = []
    for patient in range(12):
        for patch in range(2):
            name = f"images/p{patient}_{patch}.png"
            _image(tmp_path / name, patient)
            rows.append({"path": name, "label": str(patient % 2), "patient_id": f"p{patient}"})
    with (tmp_path / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    spec = DatasetAdapter().discover(tmp_path)
    patient_splits = {}
    for sample in spec.samples:
        patient_splits.setdefault(sample.group_id, set()).add(sample.split)
    assert spec.has_group_ids
    assert all(len(splits) == 1 for splits in patient_splits.values())


def test_predefined_patient_leakage_is_rejected(tmp_path):
    rows = []
    for split in ("train", "validation", "test"):
        for label in ("a", "b"):
            name = f"{split}-{label}.png"
            _image(tmp_path / name)
            rows.append({"path": name, "label": label, "split": split, "patient_id": "leaked" if label == "a" else f"{split}-b"})
    with (tmp_path / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DatasetDiscoveryError, match="group occurs"):
        DatasetAdapter().discover(tmp_path)


def test_binary_grayscale_npz_and_conformance(tmp_path):
    dataset = tmp_path / "pneumonia-like.npz"
    rng = np.random.default_rng(9)
    np.savez(
        dataset,
        train_images=rng.integers(0, 255, (12, 28, 28), dtype=np.uint8),
        train_labels=np.tile([0, 1], 6),
        val_images=rng.integers(0, 255, (6, 28, 28), dtype=np.uint8),
        val_labels=np.tile([0, 1], 3),
        test_images=rng.integers(0, 255, (6, 28, 28), dtype=np.uint8),
        test_labels=np.tile([0, 1], 3),
    )
    adapter = load_dataset_adapter("generic", seed=7)
    report = validate_dataset_adapter(adapter, dataset, tmp_path / "conformance")
    assert report["passed"] is True
    assert report["classes"] == ["0", "1"]
    assert report["channels"] == 1
    assert report["image_shape"] == [28, 28]
    assert report["split_counts"] == {"train": 12, "validation": 6, "test": 6}
    with np.load(tmp_path / "conformance/research_view/dataset.npz") as view:
        assert not any(key.startswith("test_") for key in view.files)


def test_inference_view_adds_grayscale_channel_without_changing_color(tmp_path):
    grayscale = tmp_path / "grayscale.npz"
    np.savez(
        grayscale,
        train_images=np.zeros((4, 8, 8), dtype=np.uint8),
        train_labels=np.array([0, 1, 0, 1]),
        val_images=np.zeros((2, 8, 8), dtype=np.uint8),
        val_labels=np.array([0, 1]),
        test_images=np.zeros((2, 8, 8), dtype=np.uint8),
        test_labels=np.array([0, 1]),
    )
    spec = DatasetAdapter().discover(grayscale)
    view = materialize_split_view(spec, tmp_path / "inference", {"validation"})
    normalize_inference_view_layout(spec, view)
    with np.load(view / "dataset.npz") as mounted:
        assert mounted["validation_images"].shape == (2, 1, 8, 8)

    color_spec = DatasetAdapter().discover(
        _write_color_npz(tmp_path / "color.npz")
    )
    color_view = materialize_split_view(
        color_spec, tmp_path / "color-inference", {"validation"}
    )
    normalize_inference_view_layout(color_spec, color_view)
    with np.load(color_view / "dataset.npz") as mounted:
        assert mounted["validation_images"].shape == (2, 8, 8, 3)


def _write_color_npz(path):
    np.savez(
        path,
        train_images=np.zeros((4, 8, 8, 3), dtype=np.uint8),
        train_labels=np.array([0, 1, 0, 1]),
        val_images=np.zeros((2, 8, 8, 3), dtype=np.uint8),
        val_labels=np.array([0, 1]),
        test_images=np.zeros((2, 8, 8, 3), dtype=np.uint8),
        test_labels=np.array([0, 1]),
    )
    return path


def test_conformance_rejects_identical_content_across_splits(tmp_path):
    dataset = tmp_path / "leaked.npz"
    train = np.arange(32, dtype=np.uint8).reshape(2, 4, 4)
    validation = np.arange(32, 64, dtype=np.uint8).reshape(2, 4, 4)
    np.savez(
        dataset,
        train_images=train,
        train_labels=np.array([0, 1]),
        val_images=validation,
        val_labels=np.array([0, 1]),
        test_images=np.stack([train[0], validation[1]]),
        test_labels=np.array([0, 1]),
    )
    with pytest.raises(DatasetDiscoveryError, match="Identical sample content"):
        validate_dataset_adapter(
            DatasetAdapter(seed=7), dataset, tmp_path / "conformance"
        )


def test_loads_trusted_custom_adapter(monkeypatch):
    module = types.ModuleType("example_dataset_adapter")

    class CustomAdapter(DatasetAdapter):
        pass

    module.CustomAdapter = CustomAdapter
    monkeypatch.setitem(sys.modules, module.__name__, module)
    adapter = load_dataset_adapter("example_dataset_adapter:CustomAdapter", seed=19)
    assert isinstance(adapter, CustomAdapter)
    assert adapter.seed == 19


@pytest.mark.parametrize(
    "identifier, message",
    [
        ("missing-colon", "package.module:AdapterClass"),
        ("missing_adapter_module:Adapter", "Cannot import"),
    ],
)
def test_invalid_adapter_reference_is_rejected(identifier, message):
    with pytest.raises(DatasetAdapterLoadError, match=message):
        load_dataset_adapter(identifier)


def test_adapter_without_discover_is_rejected(monkeypatch):
    module = types.ModuleType("invalid_dataset_adapter")
    module.InvalidAdapter = type("InvalidAdapter", (), {"__init__": lambda self, seed: None})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(DatasetAdapterLoadError, match="does not implement discover"):
        load_dataset_adapter("invalid_dataset_adapter:InvalidAdapter")
