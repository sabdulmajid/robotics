from __future__ import annotations

from typing import Sequence


def brier_score(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    if not probabilities:
        return 0.0
    return sum((p - float(y)) ** 2 for p, y in zip(probabilities, labels)) / len(probabilities)


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    *,
    num_bins: int = 10,
) -> float:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    if not probabilities:
        return 0.0
    bins = [[] for _ in range(num_bins)]
    for p, y in zip(probabilities, labels):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {p}")
        idx = min(num_bins - 1, int(p * num_bins))
        bins[idx].append((p, y))
    total = len(probabilities)
    ece = 0.0
    for items in bins:
        if not items:
            continue
        confidence = sum(p for p, _ in items) / len(items)
        accuracy = sum(float(y) for _, y in items) / len(items)
        ece += len(items) / total * abs(confidence - accuracy)
    return ece

