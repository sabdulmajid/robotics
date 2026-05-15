from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from risk_aware_skill_planning.evaluation.metrics import aggregate_episode_metrics
from risk_aware_skill_planning.planning.toy_planner import PlannerConfig, ToyPlanner, run_toy_episode


def run_toy_suite(
    scenario_ids: Iterable[str],
    planner_modes: Iterable[str],
    *,
    num_episodes: int,
    seed_start: int = 0,
    lambda_risk: float = 3.0,
    next_skill_threshold: float = 0.85,
    plan_threshold: float = 0.85,
    max_replans: int = 8,
) -> dict[str, dict[str, dict[str, float | int | tuple[float, float]]]]:
    grouped = defaultdict(list)
    for scenario_id in scenario_ids:
        for mode in planner_modes:
            config = PlannerConfig(
                mode=mode,
                lambda_risk=0.0 if mode == "naive_no_risk" else lambda_risk,
                next_skill_threshold=1.0 if mode == "naive_no_risk" else next_skill_threshold,
                plan_threshold=1.0 if mode == "naive_no_risk" else plan_threshold,
                max_replans=max_replans,
            )
            planner = ToyPlanner(config)
            for offset in range(num_episodes):
                seed = seed_start + offset
                episode = run_toy_episode(
                    scenario_id=scenario_id,
                    planner=planner,
                    seed=seed,
                    max_replans=max_replans,
                )
                grouped[(scenario_id, mode)].append(episode)

    summary: dict[str, dict[str, dict[str, float | int | tuple[float, float]]]] = {}
    for (scenario_id, mode), episodes in grouped.items():
        summary.setdefault(scenario_id, {})[mode] = aggregate_episode_metrics(episodes)
    return summary

