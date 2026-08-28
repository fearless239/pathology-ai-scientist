import pytest

from pathmnist.training import EpochMetrics, TrainingRun, confusion_metrics, early_stop


def test_model_construction_without_pytorch():
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock

    fake_torch = ModuleType("torch")
    fake_torch.Tensor = MagicMock()
    fake_torch.nn = MagicMock()
    original = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        from pathmnist import models

        assert hasattr(models, "SmallResNet")
    finally:
        if original is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = original


def test_confusion_metrics():
    accuracy, macro_f1 = confusion_metrics([[8, 2], [1, 9]])
    assert accuracy == pytest.approx(0.85)
    assert macro_f1 == pytest.approx((16 / 19 + 18 / 21) / 2)


def test_early_stop_and_best_epoch():
    epochs = [
        EpochMetrics(1, 1.0, 0.8, 0.5, 0.4),
        EpochMetrics(2, 0.8, 0.6, 0.6, 0.7),
        EpochMetrics(3, 0.7, 0.7, 0.6, 0.6),
        EpochMetrics(4, 0.7, 0.8, 0.6, 0.5),
    ]
    assert early_stop(epochs, patience=1, max_epochs=10) == (True, "early_stopping")
    run = TrainingRun("baseline", 7, epochs, 0, "early_stopping", 1.0)
    assert run.best().epoch == 2
