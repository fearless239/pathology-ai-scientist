from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class DatasetDiscoveryError(RuntimeError):
    """Raised when labels, samples, or split isolation cannot be established safely."""


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SampleRecord:
    id: str
    path: str
    label: str
    split: str
    group_id: str | None = None
    array_key: str | None = None
    index: int | None = None


@dataclass
class DatasetSpec:
    schema_version: int
    name: str
    source_type: str
    source_path: str
    content_sha256: str
    image_shape: list[int]
    channels: int
    classes: list[str]
    label_mapping: dict[str, int]
    split_counts: dict[str, int]
    class_counts: dict[str, dict[str, int]]
    samples: list[SampleRecord]
    has_group_ids: bool = False
    inference: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence: float = 1.0
    recommended_metrics: list[str] = field(default_factory=lambda: ["macro_f1", "accuracy"])

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["normalized_path"] = self.source_path
        return result

    def write_profile(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


class DatasetAdapter:
    """Discover a supervised image-classification dataset without dataset-name assumptions."""

    def __init__(self, seed: int = 7):
        self.seed = seed

    def discover(self, source: Path, profile_path: Path | None = None) -> DatasetSpec:
        source = source.expanduser().resolve()
        if not source.exists():
            raise DatasetDiscoveryError(f"Dataset path does not exist: {source}")
        if source.suffix.casefold() == ".zip":
            raise DatasetDiscoveryError(
                "Archive datasets must be extracted into the task dataset staging directory before discovery"
            )
        if source.is_file() and source.suffix.casefold() == ".npz":
            spec = self._discover_npz(source)
        elif source.is_dir():
            manifests = sorted([*source.glob("*.csv"), *source.glob("*.json")])
            spec = self._discover_manifest(source, manifests[0]) if manifests else self._discover_folders(source)
        else:
            raise DatasetDiscoveryError("Supported inputs are NPZ files and image dataset directories")
        self._validate(spec)
        if profile_path:
            spec.write_profile(profile_path)
        return spec

    def _discover_npz(self, path: Path) -> DatasetSpec:
        with np.load(path, mmap_mode="r", allow_pickle=False) as data:
            keys = set(data.files)
            aliases = {"validation": ("val", "valid", "validation")}
            found: dict[str, tuple[str, str]] = {}
            for split in SPLITS:
                names = aliases.get(split, (split,))
                matches = [(f"{name}_images", f"{name}_labels") for name in names if {f"{name}_images", f"{name}_labels"} <= keys]
                if len(matches) > 1:
                    raise DatasetDiscoveryError(f"Ambiguous NPZ keys for {split}: {matches}")
                if matches:
                    found[split] = matches[0]
            if not found:
                image_keys = [key for key in keys if key in {"images", "x", "X"}]
                label_keys = [key for key in keys if key in {"labels", "y", "Y"}]
                if len(image_keys) != 1 or len(label_keys) != 1:
                    raise DatasetDiscoveryError("NPZ labels/images cannot be uniquely identified")
                found["unsplit"] = (image_keys[0], label_keys[0])
            rows: list[dict[str, Any]] = []
            shape: list[int] = []
            for split, (image_key, label_key) in found.items():
                images, labels = data[image_key], data[label_key].reshape(-1)
                if len(images) != len(labels):
                    raise DatasetDiscoveryError(f"Sample/label length mismatch for {split}")
                if not shape:
                    shape = list(images.shape[1:])
                for index, label in enumerate(labels.tolist()):
                    rows.append({"id": f"{split}:{index}", "path": str(path), "label": str(label), "split": split, "array_key": image_key, "index": index})
        return self._build_spec(path, "npz", rows, shape, [f"NPZ key pairs: {found}"])

    def _discover_folders(self, root: Path) -> DatasetSpec:
        split_dirs = {p.name.casefold(): p for p in root.iterdir() if p.is_dir() and p.name.casefold() in {"train", "val", "valid", "validation", "test"}}
        rows: list[dict[str, Any]] = []
        if split_dirs:
            normalized = {("validation" if key in {"val", "valid"} else key): value for key, value in split_dirs.items()}
            for split, split_dir in normalized.items():
                for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                    for image in self._images(class_dir):
                        rows.append(self._file_row(image, class_dir.name, split, root))
            inference = ["Split and label inferred from split/class directory hierarchy"]
        else:
            class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
            if len(class_dirs) < 2:
                raise DatasetDiscoveryError("A folder dataset requires at least two unambiguous class directories")
            for class_dir in class_dirs:
                for image in self._images(class_dir):
                    rows.append(self._file_row(image, class_dir.name, "unsplit", root))
            inference = ["Labels inferred from class directory names; deterministic splits generated"]
        if not rows:
            raise DatasetDiscoveryError("No supported images were found")
        return self._build_spec(root, "image_folder", rows, self._image_shape(Path(rows[0]["path"])), inference)

    def _discover_manifest(self, root: Path, manifest: Path) -> DatasetSpec:
        if manifest.suffix.casefold() == ".json":
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            rows = raw if isinstance(raw, list) else raw.get("samples", [])
        else:
            with manifest.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        if not rows or not {"path", "label"} <= set(rows[0]):
            raise DatasetDiscoveryError("Manifest must contain path and label fields")
        normalized = []
        for index, row in enumerate(rows):
            image = (root / str(row["path"])).resolve()
            if root not in image.parents or not image.is_file():
                raise DatasetDiscoveryError(f"Manifest image is missing or escapes dataset root: {row['path']}")
            split = str(row.get("split") or "unsplit").casefold()
            split = "validation" if split in {"val", "valid"} else split
            if split not in {*SPLITS, "unsplit"}:
                raise DatasetDiscoveryError(f"Unknown split {split!r}")
            normalized.append({"id": str(row.get("id") or index), "path": str(image), "label": str(row["label"]), "split": split, "group_id": row.get("group_id") or row.get("patient_id")})
        return self._build_spec(root, "manifest", normalized, self._image_shape(Path(normalized[0]["path"])), [f"Labels loaded from {manifest.name}"])

    def _build_spec(self, source: Path, source_type: str, rows: list[dict[str, Any]], shape: list[int], inference: list[str]) -> DatasetSpec:
        rows = self._assign_splits(rows)
        classes = sorted({str(row["label"]) for row in rows})
        samples = [SampleRecord(**row) for row in rows]
        split_counts = Counter(row["split"] for row in rows)
        class_counts = {split: dict(Counter(row["label"] for row in rows if row["split"] == split)) for split in SPLITS}
        channels = int(shape[-1]) if len(shape) == 3 and shape[-1] <= 4 else 1
        warnings = []
        totals = Counter(row["label"] for row in rows)
        if totals and max(totals.values()) / min(totals.values()) >= 3:
            warnings.append("Class imbalance ratio is at least 3:1")
        return DatasetSpec(2, source.stem, source_type, str(source), self._hash_source(source), shape, channels, classes, {name: i for i, name in enumerate(classes)}, dict(split_counts), class_counts, samples, any(row.get("group_id") for row in rows), inference, warnings)

    def _assign_splits(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        present = {row["split"] for row in rows}
        if "unsplit" in present and len(present) > 1:
            raise DatasetDiscoveryError("Mixed split and unsplit samples are ambiguous")
        if present == {"unsplit"}:
            return self._stratified(rows, {"train": 0.70, "validation": 0.15, "test": 0.15})
        if present == {"train", "test"}:
            train = [row for row in rows if row["split"] == "train"]
            test = [row for row in rows if row["split"] == "test"]
            return self._stratified(train, {"train": 0.85, "validation": 0.15}) + test
        if not {"train", "validation", "test"} <= present:
            raise DatasetDiscoveryError(f"Incomplete predefined splits: {sorted(present)}")
        return rows

    def _stratified(self, rows: list[dict[str, Any]], ratios: dict[str, float]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        has_groups = any(row.get("group_id") for row in rows)
        if has_groups and not all(row.get("group_id") for row in rows):
            raise DatasetDiscoveryError("Group IDs are partially missing; group-aware splitting is unsafe")
        for row in rows:
            grouped[str(row.get("group_id") if has_groups else row["label"])].append(row)
        rng = random.Random(self.seed)
        result = []
        if has_groups:
            buckets = list(grouped.values())
            rng.shuffle(buckets)
            targets = {key: len(rows) * value for key, value in ratios.items()}
            counts = Counter()
            for bucket in buckets:
                split = min(ratios, key=lambda key: counts[key] / max(targets[key], 1))
                for row in bucket:
                    result.append({**row, "split": split})
                counts[split] += len(bucket)
        else:
            for bucket in grouped.values():
                rng.shuffle(bucket)
                splits = list(ratios)
                cumulative = 0
                for pos, row in enumerate(bucket):
                    fraction = (pos + 0.5) / len(bucket)
                    while cumulative < len(splits) - 1 and fraction > sum(ratios[s] for s in splits[: cumulative + 1]):
                        cumulative += 1
                    result.append({**row, "split": splits[cumulative]})
        return result

    def _validate(self, spec: DatasetSpec) -> None:
        if len(spec.classes) < 2:
            raise DatasetDiscoveryError("Classification requires at least two classes")
        identities: dict[str, str] = {}
        groups: dict[str, str] = {}
        for sample in spec.samples:
            identity = f"{sample.path}:{sample.array_key}:{sample.index}"
            if identity in identities and identities[identity] != sample.split:
                raise DatasetDiscoveryError("The same sample occurs in multiple splits")
            identities[identity] = sample.split
            if sample.group_id:
                if sample.group_id in groups and groups[sample.group_id] != sample.split:
                    raise DatasetDiscoveryError("A patient/group occurs in multiple splits")
                groups[sample.group_id] = sample.split
        missing = [split for split in SPLITS if not spec.split_counts.get(split)]
        if missing:
            raise DatasetDiscoveryError(f"Empty required splits: {missing}; provide more samples")

    @staticmethod
    def _images(root: Path) -> Iterable[Path]:
        return (path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES)

    @staticmethod
    def _file_row(path: Path, label: str, split: str, root: Path) -> dict[str, Any]:
        return {"id": path.relative_to(root).as_posix(), "path": str(path.resolve()), "label": label, "split": split}

    @staticmethod
    def _image_shape(path: Path) -> list[int]:
        try:
            from PIL import Image
            with Image.open(path) as image:
                return [image.height, image.width, len(image.getbands())]
        except Exception as exc:
            raise DatasetDiscoveryError(f"Cannot inspect image {path}: {exc}") from exc

    @staticmethod
    def _hash_source(source: Path) -> str:
        digest = hashlib.sha256()
        paths = [source] if source.is_file() else [p for p in sorted(source.rglob("*")) if p.is_file()]
        for path in paths:
            # A single-file dataset uses the conventional byte-for-byte file hash so
            # its profile agrees with pinned provenance records. Directory datasets
            # include relative paths to make their tree identity unambiguous.
            if source.is_dir():
                digest.update(path.relative_to(source).as_posix().encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()


def safely_extract_zip(archive: Path, destination: Path) -> Path:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise DatasetDiscoveryError(f"Archive member escapes staging directory: {member.filename}")
        bundle.extractall(destination)
    return destination


def split_array_keys(split: str) -> tuple[str, str, str]:
    """Canonical keys of the exported view, independent of source aliases."""
    if split not in SPLITS:
        raise DatasetDiscoveryError(f"Unknown split: {split}")
    return tuple(f"{split}_{suffix}" for suffix in ("images", "labels", "sample_ids"))


def research_view_interface(spec: DatasetSpec, view: Path | None = None) -> dict[str, Any]:
    """Describe the exported research input, and verify it when already materialized."""
    splits = sorted({sample.split for sample in spec.samples} & {"train", "validation"})
    if spec.source_type != "npz":
        if view is not None and not (view / "manifest.json").is_file():
            raise DatasetDiscoveryError("Research view manifest is missing")
        return {"manifest_path": "/dataset/manifest.json", "splits": splits,
                "note": "Read paths, labels and sample IDs from the mounted manifest."}
    keys = sorted(key for split in splits for key in split_array_keys(split))
    if view is not None:
        with np.load(view / "dataset.npz", allow_pickle=False) as data:
            actual = sorted(data.files)
        if actual != keys:
            raise DatasetDiscoveryError(
                f"Research view interface mismatch: expected {keys}, mounted {actual}"
            )
    return {
        "npz_path": "/dataset/dataset.npz",
        "array_keys": keys,
        "image_array_layout": (
            "NHWC" if len(spec.image_shape) == 3 and spec.image_shape[-1] == spec.channels
            else "NCHW_or_grayscale_as_reported_by_image_shape"
        ),
        "note": "These are canonical exported-view keys, not original source keys. "
                "Use validation_images/validation_labels, not val_images/val_labels. "
                "Read sample IDs from validation_sample_ids.",
    }


def materialize_split_view(spec: DatasetSpec, destination: Path, allowed_splits: set[str]) -> Path:
    """Create a physical dataset view containing only explicitly allowed splits.

    This is the path mounted into an experiment container. In particular, an NPZ
    containing all original splits is rewritten instead of mounted directly.
    """
    unknown = allowed_splits - set(SPLITS)
    if unknown:
        raise DatasetDiscoveryError(f"Unknown allowed splits: {sorted(unknown)}")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    selected = [sample for sample in spec.samples if sample.split in allowed_splits]
    if not selected:
        raise DatasetDiscoveryError("The requested split view is empty")
    if spec.source_type == "npz":
        arrays: dict[str, np.ndarray] = {}
        by_split: dict[str, list[SampleRecord]] = defaultdict(list)
        for sample in selected:
            by_split[sample.split].append(sample)
        with np.load(spec.source_path, allow_pickle=False) as source:
            for split, samples in by_split.items():
                keys = {sample.array_key for sample in samples}
                if len(keys) != 1 or None in keys:
                    raise DatasetDiscoveryError(f"Inconsistent NPZ array keys for {split}")
                image_key = next(iter(keys))
                indices = np.asarray([sample.index for sample in samples], dtype=np.int64)
                images_key, labels_key, ids_key = split_array_keys(split)
                arrays[images_key] = source[image_key][indices]
                arrays[labels_key] = np.asarray(
                    [spec.label_mapping[sample.label] for sample in samples], dtype=np.int64
                )
                arrays[ids_key] = np.asarray(
                    [sample.id for sample in samples], dtype=np.str_
                )
        np.savez_compressed(destination / "dataset.npz", **arrays)
    else:
        manifest = []
        for sample in selected:
            split_dir = destination / sample.split / sample.label
            split_dir.mkdir(parents=True, exist_ok=True)
            target = split_dir / f"{sample.id.replace('/', '_').replace(':', '_')}{Path(sample.path).suffix}"
            shutil.copy2(sample.path, target)
            manifest.append(
                {
                    "id": sample.id,
                    "path": target.relative_to(destination).as_posix(),
                    "label": sample.label,
                    "split": sample.split,
                    "group_id": sample.group_id,
                }
            )
        (destination / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (destination / "dataset_profile.json").write_text(
        json.dumps(
            {**spec.to_dict(), "mounted_splits": sorted(allowed_splits), "samples": [asdict(s) for s in selected]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination
