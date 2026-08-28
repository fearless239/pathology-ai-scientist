import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pathmnist.dataset_adapter import DatasetAdapter, DatasetDiscoveryError, materialize_split_view


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
