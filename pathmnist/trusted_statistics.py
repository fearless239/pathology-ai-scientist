from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Iterable


class StatisticsError(ValueError):
    """Raised when a requested statistical comparison is not reproducible."""


def comparison_summary(baseline, intervention, *, bootstrap_seed=1701):
    base, method = _values(baseline, 'baseline'), _values(intervention, 'intervention')
    if len(base) == len(method) == 1:
        return {'schema_version': 2, 'n': 1, 'baseline': list(base), 'intervention': list(method),
                'differences': [method[0] - base[0]], 'mean_baseline': base[0],
                'mean_intervention': method[0], 'mean_difference': method[0] - base[0],
                'test': 'not_applicable_single_repeat', 'sign_flip_p_value': None,
                'confidence_interval_95': None, 'standardized_effect': None,
                'limitation': 'Single comparison; no inferential significance or repeat stability claim.'}
    return paired_comparison(base, method, bootstrap_seed=bootstrap_seed).as_dict()


@dataclass(frozen=True)
class PairedComparison:
    baseline: tuple[float, ...]
    intervention: tuple[float, ...]
    differences: tuple[float, ...]
    mean_baseline: float
    mean_intervention: float
    mean_difference: float
    sample_std_difference: float
    standardized_effect: float | None
    confidence_interval_95: tuple[float, float]
    sign_flip_p_value: float

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "baseline": list(self.baseline),
            "intervention": list(self.intervention),
            "differences": list(self.differences),
            "n": len(self.differences),
            "mean_baseline": self.mean_baseline,
            "mean_intervention": self.mean_intervention,
            "mean_difference": self.mean_difference,
            "sample_std_difference": self.sample_std_difference,
            "standardized_effect": self.standardized_effect,
            "confidence_interval_95": list(self.confidence_interval_95),
            "test": "exact_paired_sign_flip",
            "sign_flip_p_value": self.sign_flip_p_value,
        }


def _values(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise StatisticsError(f"{name} must contain finite values")
    return result


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def _sample_std(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def exact_paired_sign_flip_p_value(differences: Iterable[float]) -> float:
    values = _values(differences, "differences")
    observed = abs(_mean(values))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(_mean(tuple(sign * value for sign, value in zip(signs, values))))
        total += 1
        if statistic + 1e-15 >= observed:
            extreme += 1
    return extreme / total


def bootstrap_mean_interval(
    values: Iterable[float], *, samples: int = 10_000, seed: int = 1701
) -> tuple[float, float]:
    source = _values(values, "values")
    if samples < 100:
        raise StatisticsError("bootstrap requires at least 100 resamples")
    generator = random.Random(seed)
    estimates = sorted(
        _mean(tuple(source[generator.randrange(len(source))] for _ in source))
        for _ in range(samples)
    )
    lower = estimates[max(0, round(0.025 * (samples - 1)))]
    upper = estimates[min(samples - 1, round(0.975 * (samples - 1)))]
    return lower, upper


def paired_comparison(
    baseline: Iterable[float], intervention: Iterable[float], *, bootstrap_seed: int = 1701
) -> PairedComparison:
    base = _values(baseline, "baseline")
    method = _values(intervention, "intervention")
    if len(base) != len(method):
        raise StatisticsError("paired comparisons require equal sample counts")
    if len(base) < 2:
        raise StatisticsError("paired comparisons require at least two repeats")
    differences = tuple(right - left for left, right in zip(base, method))
    spread = _sample_std(differences)
    return PairedComparison(
        baseline=base,
        intervention=method,
        differences=differences,
        mean_baseline=_mean(base),
        mean_intervention=_mean(method),
        mean_difference=_mean(differences),
        sample_std_difference=spread,
        standardized_effect=(_mean(differences) / spread if spread > 0 else None),
        confidence_interval_95=bootstrap_mean_interval(differences, seed=bootstrap_seed),
        sign_flip_p_value=exact_paired_sign_flip_p_value(differences),
    )
