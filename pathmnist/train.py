from __future__ import annotations

import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from .config import AppConfig, VariantConfig
from .datasets import make_loader
from .models import SmallResNet
from .training import EpochMetrics, TrainingRun, early_stop, write_run_artifacts


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def variant_learning_rate(config: AppConfig, variant: VariantConfig) -> float:
    return config.experiment.baseline_learning_rate if variant.optimization else 0.001


def variant_weight_decay(config: AppConfig, variant: VariantConfig) -> float:
    raw = config.raw["experiment"]["improvements"]["optimization"]
    return float(raw["weight_decay"]) if variant.optimization else 0.0


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float, list[list[int]]]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    loss_total = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss_total += float(criterion(outputs, labels).item())
            predictions.append(outputs.argmax(1).cpu().numpy())
            targets.append(labels.cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    matrix = confusion_matrix(target, prediction, labels=range(9)).tolist()
    return (
        loss_total / len(target),
        float(f1_score(target, prediction, average="macro", zero_division=0)),
        matrix,
    )


def run_single(
    config: AppConfig,
    variant: VariantConfig,
    seed: int,
    archive: dict[str, np.ndarray],
    output_root: Path,
    device: torch.device,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    save_checkpoint: bool = False,
) -> Path:
    experiment = config.experiment
    output_dir = output_root / variant.name / f"seed_{seed}"
    result_path = output_dir / "run.json"
    if result_path.exists():
        return output_dir

    seed_everything(seed)
    learning_rate = learning_rate or variant_learning_rate(config, variant)
    weight_decay = weight_decay if weight_decay is not None else variant_weight_decay(config, variant)
    raw_optimization = config.raw["experiment"]["improvements"]["optimization"]
    label_smoothing = float(raw_optimization["label_smoothing"]) if variant.optimization else 0.0

    train_loader = make_loader(archive, "train", experiment.batch_size, variant.augmentation, experiment.num_workers, seed)
    val_loader = make_loader(archive, "val", experiment.batch_size, False, experiment.num_workers, seed)
    model = SmallResNet(classes=config.dataset.classes, multiscale=variant.multiscale).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    if variant.optimization:
        steps = experiment.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, total_steps=steps, pct_start=0.2
        )

    started = time.monotonic()
    epochs: list[EpochMetrics] = []
    stop_reason = "max_epochs"
    for epoch in range(1, experiment.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            running_loss += float(loss.item()) * len(labels)
            seen += len(labels)
        validation_loss, macro_f1, matrix = evaluate(model, val_loader, device)
        accuracy = sum(matrix[index][index] for index in range(9)) / sum(sum(row) for row in matrix)
        epochs.append(EpochMetrics(epoch, running_loss / seen, validation_loss, accuracy, macro_f1))
        stop, stop_reason = early_stop(epochs, experiment.early_stop_patience, experiment.epochs)
        if stop:
            break

    training_seconds = time.monotonic() - started
    run = TrainingRun(variant.name, seed, epochs, 0, stop_reason, training_seconds)
    write_run_artifacts(
        run,
        output_dir,
        {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "environment": {
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(device),
                "git_commit": git_commit(),
            },
        },
    )
    if save_checkpoint:
        torch.save(
            {
                "model_state": model.state_dict(),
                "variant": variant.name,
                "seed": seed,
                "best_epoch": run.best().epoch,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            },
            output_dir / "checkpoint.pt",
        )
    return output_dir


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def tune(config: AppConfig, archive: dict[str, np.ndarray], output_root: Path, device: torch.device) -> Path:
    combined = next(item for item in config.experiment.variants if item.name == "combined")
    scores: list[dict[str, Any]] = []
    for learning_rate in config.experiment.learning_rates:
        for weight_decay in config.experiment.weight_decays:
            setting = f"lr_{learning_rate:g}_wd_{weight_decay:g}"
            directory = run_single(
                config,
                combined,
                config.experiment.seeds[0],
                archive,
                output_root / "tuning" / setting,
                device,
                learning_rate,
                weight_decay,
            )
            payload = json.loads((directory / "run.json").read_text())
            scores.append(
                {
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "macro_f1": payload["epochs"][payload["best_epoch"] - 1]["macro_f1"],
                }
            )
    best = max(scores, key=lambda item: item["macro_f1"])
    path = output_root / "tuning" / "result.json"
    path.write_text(json.dumps({"scores": scores, "best": best}, indent=2) + "\n")
    return path
