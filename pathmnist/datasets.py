from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset


class PathMNISTDataset(Dataset):
    def __init__(self, archive: dict[str, np.ndarray], split: str, augment: bool) -> None:
        self.images = archive[f"{split}_images"]
        self.labels = archive[f"{split}_labels"].reshape(-1).astype(np.int64, copy=False)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        image = self.images[index].astype(np.float32) / 255.0
        label = int(self.labels[index])
        if self.augment:
            if np.random.rand() < 0.5:
                image = image[::-1].copy()
            if np.random.rand() < 0.5:
                image = image[:, ::-1].copy()
            noise = np.random.normal(0.0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)
        return np.transpose(image, (2, 0, 1)), label


def worker_init(worker_id: int) -> None:
    seed = int(np.random.get_state()[1][0]) + worker_id
    np.random.seed(seed)


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def make_loader(
    archive: dict[str, np.ndarray], split: str, batch_size: int, augment: bool, workers: int, seed: int
) -> DataLoader:
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        PathMNISTDataset(archive, split, augment),
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=workers,
        generator=generator,
        worker_init_fn=worker_init if augment else None,
        persistent_workers=workers > 0,
    )
