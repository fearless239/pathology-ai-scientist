from __future__ import annotations

import ast
import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .config import DatasetConfig


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetSummary:
    sha256: str
    splits: dict[str, int]
    classes: dict[str, tuple[int, ...]]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_npz(
    path: Path,
) -> tuple[dict[str, tuple[tuple[int, ...], str]], dict[str, tuple[int, ...]]]:
    arrays: dict[str, tuple[tuple[int, ...], str]] = {}
    labels: dict[str, tuple[int, ...]] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                data = archive.read(name)
                if data[:6] != b"\x93NUMPY":
                    raise DataValidationError(f"{name} is not a NumPy array")
                header_length = int.from_bytes(data[8:10], "little")
                header = ast.literal_eval(data[10 : 10 + header_length].decode("latin1"))
                shape = tuple(int(value) for value in header["shape"])
                arrays[name.removesuffix(".npy")] = (shape, str(header["descr"]))
                if name.endswith("_labels.npy"):
                    payload = data[10 + header_length :]
                    labels[name.removesuffix(".npy").removesuffix("_labels")] = tuple(
                        payload[: shape[0]]
                    )
    except (OSError, zipfile.BadZipFile, SyntaxError, TypeError, ValueError) as exc:
        raise DataValidationError(f"Unable to inspect {path}: {exc}") from exc
    return arrays, labels


def validate_dataset(config: DatasetConfig) -> DatasetSummary:
    path = config.path
    if not path.is_file():
        raise DataValidationError(f"Dataset is absent: {path}")
    digest = sha256_file(path)
    if digest != config.sha256:
        raise DataValidationError(f"Dataset SHA-256 mismatch: {digest} != {config.sha256}")
    arrays, labels = inspect_npz(path)
    expected_arrays = {
        f"{split}_images": ((count, 64, 64, 3), "|u1")
        for split, count in config.expected_splits.items()
    }
    expected_arrays.update(
        {
            f"{split}_labels": ((count, 1), "|u1")
            for split, count in config.expected_splits.items()
        }
    )
    if arrays != expected_arrays:
        raise DataValidationError(f"Dataset array metadata mismatch: {arrays}")
    summaries: dict[str, tuple[int, ...]] = {}
    for split in config.expected_splits:
        values = labels[split]
        if sorted(set(values)) != list(range(config.classes)):
            raise DataValidationError(f"{split} labels do not contain all classes")
        summaries[split] = tuple(
            values.count(label) for label in range(config.classes)
        )
    return DatasetSummary(
        sha256=digest, splits=dict(config.expected_splits), classes=summaries
    )
