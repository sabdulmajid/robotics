from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from risk_aware_skill_planning.risk.openpi_models import choose_threshold


@dataclass(frozen=True)
class CalibrationSelection:
    threshold: float
    false_negative_rate: float
    coverage: float


def select_openpi_threshold(labels: Sequence[int], probs: Sequence[float]) -> CalibrationSelection:
    threshold = choose_threshold(labels, probs)
    attempted = [(label, prob) for label, prob in zip(labels, probs) if prob < threshold]
    missed_failures = sum(label for label, _ in attempted)
    total_failures = sum(labels)
    return CalibrationSelection(
        threshold=threshold,
        false_negative_rate=missed_failures / total_failures if total_failures else 0.0,
        coverage=len(attempted) / len(labels) if labels else 0.0,
    )
