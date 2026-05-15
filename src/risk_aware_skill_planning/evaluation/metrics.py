from __future__ import annotations

import random
from statistics import mean
from typing import Callable, Sequence

from risk_aware_skill_planning.contracts import EpisodeResult, TERMINAL_REJECTED


def _safe_mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def bootstrap_rate_ci(
    episodes: Sequence[EpisodeResult],
    predicate: Callable[[EpisodeResult], bool],
    *,
    num_bootstrap: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    if not episodes:
        return (0.0, 0.0)
    rng = random.Random(seed)
    rates: list[float] = []
    n = len(episodes)
    for _ in range(num_bootstrap):
        sample = [episodes[rng.randrange(n)] for _ in range(n)]
        rates.append(sum(predicate(ep) for ep in sample) / n)
    rates.sort()
    low_idx = int(0.025 * (len(rates) - 1))
    high_idx = int(0.975 * (len(rates) - 1))
    return (rates[low_idx], rates[high_idx])


def aggregate_episode_metrics(
    episodes: Sequence[EpisodeResult],
    *,
    success_reward: float = 1.0,
    catastrophe_penalty: float = 2.0,
    rejection_penalty: float = 0.2,
    cost_penalty: float = 0.05,
    num_bootstrap: int = 200,
) -> dict[str, float | int | tuple[float, float]]:
    total = len(episodes)
    if total == 0:
        return {
            "episodes": 0,
            "task_completion_rate": 0.0,
            "attempted_success_rate": 0.0,
            "catastrophic_failure_rate": 0.0,
            "rejection_rate": 0.0,
            "coverage": 0.0,
            "utility": 0.0,
        }

    attempted = [ep for ep in episodes if ep.attempted]
    successes = sum(ep.success for ep in episodes)
    safety_failures = sum(ep.safety_failure for ep in episodes)
    rejections = sum(ep.terminal_label == TERMINAL_REJECTED for ep in episodes)
    utility_values = [
        success_reward * float(ep.success)
        - catastrophe_penalty * float(ep.safety_failure)
        - rejection_penalty * float(ep.terminal_label == TERMINAL_REJECTED)
        - cost_penalty * ep.execution_cost
        for ep in episodes
    ]

    metrics: dict[str, float | int | tuple[float, float]] = {
        "episodes": total,
        "task_completion_rate": successes / total,
        "attempted_success_rate": successes / len(attempted) if attempted else 0.0,
        "catastrophic_failure_rate": safety_failures / total,
        "rejection_rate": rejections / total,
        "coverage": len(attempted) / total,
        "utility": _safe_mean(utility_values),
        "mean_failed_skill_executions": _safe_mean([float(ep.failed_skill_executions) for ep in episodes]),
        "mean_replans": _safe_mean([float(ep.replans) for ep in episodes]),
        "mean_plan_cost": _safe_mean([ep.execution_cost for ep in episodes]),
        "mean_cumulative_predicted_risk": _safe_mean([ep.cumulative_predicted_risk for ep in episodes]),
        "unsafe_rejection_rate": rejections / total,
    }
    metrics["task_completion_rate_ci95"] = bootstrap_rate_ci(
        episodes,
        lambda ep: ep.success,
        num_bootstrap=num_bootstrap,
    )
    metrics["catastrophic_failure_rate_ci95"] = bootstrap_rate_ci(
        episodes,
        lambda ep: ep.safety_failure,
        num_bootstrap=num_bootstrap,
    )
    metrics["rejection_rate_ci95"] = bootstrap_rate_ci(
        episodes,
        lambda ep: ep.terminal_label == TERMINAL_REJECTED,
        num_bootstrap=num_bootstrap,
    )
    return metrics
