from pathlib import Path

import pytest

from pathmnist.candidates import CandidateError, require_inference_candidate


def test_test_approval_blocks_training_only_candidate(tmp_path: Path) -> None:
    frozen = tmp_path / "candidate_frozen"
    frozen.mkdir()
    (frozen / "run.py").write_text('data["train_images"]', encoding="utf-8")
    with pytest.raises(CandidateError, match="inference-only"):
        require_inference_candidate(tmp_path)


def test_test_approval_accepts_checkpoint_backed_dual_mode_candidate(tmp_path: Path) -> None:
    frozen = tmp_path / "candidate_frozen"
    frozen.mkdir()
    (frozen / "run.py").write_text('HAS_TRAIN_SPLIT = False\ntorch.load("/workspace/model_checkpoint.pt")', encoding="utf-8")
    (frozen / "model_checkpoint.pt").write_bytes(b"weights")
    require_inference_candidate(tmp_path)
