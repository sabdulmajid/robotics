from __future__ import annotations

import random
from statistics import mean
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def bootstrap_ci(
    values: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    samples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "low": None, "high": None, "samples": 0}
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(statistic(sample))
    estimates.sort()
    low_idx = max(0, int((alpha / 2.0) * samples) - 1)
    high_idx = min(samples - 1, int((1.0 - alpha / 2.0) * samples))
    return {
        "mean": mean(estimates),
        "low": estimates[low_idx],
        "high": estimates[high_idx],
        "samples": samples,
    }
