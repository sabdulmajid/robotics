from __future__ import annotations

from typing import Sequence


def risk_coverage_curve(labels: Sequence[int], probs: Sequence[float]) -> list[dict[str, float]]:
    if len(labels) != len(probs):
        raise ValueError("labels and probs must have equal length")
    rows: list[dict[str, float]] = []
    for threshold in sorted(set(float(prob) for prob in probs) | {0.0, 0.5, 0.8, 1.0}):
        attempted = [label for label, prob in zip(labels, probs) if prob < threshold]
        rows.append(
            {
                "threshold": threshold,
                "coverage": len(attempted) / len(labels) if labels else 0.0,
                "failure_rate_attempted": sum(attempted) / len(attempted) if attempted else 0.0,
            }
        )
    return rows


def expected_utility(
    *,
    success: bool,
    timeout: bool,
    abstained: bool,
    episode_length: int,
    extra_policy_queries: int = 0,
) -> float:
    utility = 1.0 if success else 0.0
    if not success and not timeout and not abstained:
        utility -= 1.0
    if timeout:
        utility -= 0.5
    if abstained:
        utility -= 0.2
    utility -= 0.01 * max(0, episode_length) / 100.0
    utility -= 0.02 * max(0, extra_policy_queries)
    return utility
