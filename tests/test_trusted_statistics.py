import pytest

from pathmnist.trusted_statistics import paired_comparison


def test_paired_statistics_preserve_pre_specified_pairing():
    result = paired_comparison(
        [0.60, 0.61, 0.59, 0.60, 0.62],
        [0.64, 0.65, 0.63, 0.64, 0.66],
        bootstrap_seed=7,
    )
    assert result.mean_difference == pytest.approx(0.04)
    assert result.sign_flip_p_value == pytest.approx(0.0625)
    assert result.confidence_interval_95[0] == pytest.approx(0.04)
