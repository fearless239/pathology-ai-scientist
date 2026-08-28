"""The paid smoke never executes arbitrary generated host code."""
import runpy
from pathlib import Path

import pytest


validate_patch = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/llm-gpu-smoke.py'))['validate_patch']


def test_accepts_literal_augmentation_patch():
    assert 'ColorJitter' in validate_patch('[transforms.ColorJitter(brightness=0.1), transforms.RandomRotation(15)]')


@pytest.mark.parametrize('text', [
    "[__import__('os').system('echo unsafe')]",
    '[transforms.RandomRotation(180)]',
    '[transforms.ColorJitter(brightness=some_function())]',
    '[transforms.RandomHorizontalFlip(**options)]',
    '[]',
    '[transforms.RandomHorizontalFlip()] * 1000000',
])
def test_rejects_executable_or_unbounded_patch(text):
    with pytest.raises(ValueError):
        validate_patch(text)
