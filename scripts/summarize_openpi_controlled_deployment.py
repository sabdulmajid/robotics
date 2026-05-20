#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from risk_aware_skill_planning.evaluation.bootstrap import bootstrap_ci
from risk_aware_skill_planning.evaluation.risk_coverage import expected_utility


DIRECT_LABEL = "direct"
FIXED_LABEL = "fixed"
SIGLIP_0933_LABEL = "siglip_0933"
SIGLIP_0986_LABEL = "siglip_0986"
LABEL_ORDER = [DIRECT_LABEL, FIXED_LABEL, SIGLIP_0933_LABEL, SIGLIP_0986_LABEL]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize same-seed controlled OpenPI runtime deployment jobs")
    parser.add_argument("--manifest", required=True, help="JSONL manifest produced at job submission time")
    parser.add_argument("--output", default="reports/openpi_runtime_controlled_deployment_summary.json")
    parser.add_argument("--random-seeds", type=int, default=1000)
    args = parser.parse_args()

    manifest = read_jsonl(Path(args.manifest))
    episodes_by_label = load_episodes_by_label(manifest)
    summary = summarize_controlled_deployment(
        manifest,
        episodes_by_label,
        random_seeds=int(args.random_seeds),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": summary["ok"], "episodes": summary["episodes"], "output": str(output)}, indent=2))
    return 0 if summary["ok"] else 2


def summarize_controlled_deployment(
    manifest: Sequence[Mapping[str, Any]],
    episodes_by_label: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    random_seeds: int,
) -> dict[str, Any]:
    failures: list[str] = []
    grid_checks = validate_grid(episodes_by_label)
    if not grid_checks["ok"]:
        failures.extend(grid_checks["failures"])
    metrics_by_label = {label: summarize_episode_set(episodes_by_label.get(label, [])) for label in LABEL_ORDER}
    direct_metrics = metrics_by_label[DIRECT_LABEL]
    comparisons = {
        label: comparison_against_direct(metrics, direct_metrics)
        for label, metrics in metrics_by_label.items()
        if label != DIRECT_LABEL
    }
    matched_baselines: dict[str, Any] = {}
    direct_episodes = list(episodes_by_label.get(DIRECT_LABEL, []))
    for label in (SIGLIP_0933_LABEL, SIGLIP_0986_LABEL):
        coverage = float(metrics_by_label[label]["coverage"])
        matched_baselines[label] = {
            "random_abstain_matched_coverage": random_abstain_matched_coverage(
                direct_episodes,
                target_coverage=coverage,
                samples=random_seeds,
            ),
            "oracle_abstain_upper_bound": oracle_abstain_upper_bound(
                direct_episodes,
                target_coverage=coverage,
            ),
        }
    return {
        "ok": not failures,
        "failures": failures,
        "episodes": sum(len(episodes) for episodes in episodes_by_label.values()),
        "fresh_controlled_episodes": sum(len(episodes) for episodes in episodes_by_label.values()),
        "manifest": list(manifest),
        "job_ids": {label: [int(row["job_id"]) for row in manifest if row["label"] == label] for label in LABEL_ORDER},
        "grid_checks": grid_checks,
        "metrics": metrics_by_label,
        "comparisons_vs_direct": comparisons,
        "matched_baselines": matched_baselines,
        "analysis_questions": answer_analysis_questions(metrics_by_label, comparisons, matched_baselines),
    }


def load_episodes_by_label(manifest: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = {label: [] for label in LABEL_ORDER}
    for row in manifest:
        job_id = int(row["job_id"])
        path = Path("datasets/openpi_libero_rollouts") / f"openpi_rollouts_{job_id}.jsonl"
        if not path.exists():
            continue
        label = str(row["label"])
        output.setdefault(label, []).extend(read_jsonl(path))
    return output


def validate_grid(episodes_by_label: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    failures: list[str] = []
    keys_by_label = {label: {episode_key(episode) for episode in episodes_by_label.get(label, [])} for label in LABEL_ORDER}
    direct_keys = keys_by_label.get(DIRECT_LABEL, set())
    for label in LABEL_ORDER:
        if not keys_by_label.get(label):
            failures.append(f"No episodes found for label {label}")
        missing = sorted(direct_keys - keys_by_label.get(label, set()))
        extra = sorted(keys_by_label.get(label, set()) - direct_keys)
        if missing:
            failures.append(f"{label} is missing {len(missing)} direct-grid keys")
        if extra:
            failures.append(f"{label} has {len(extra)} keys outside direct grid")
    return {
        "ok": not failures,
        "failures": failures,
        "label_episode_counts": {label: len(episodes_by_label.get(label, [])) for label in LABEL_ORDER},
        "unique_grid_keys": {label: len(keys) for label, keys in keys_by_label.items()},
        "same_grid": not failures,
    }


def summarize_episode_set(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = [episode_outcome(episode) for episode in episodes]
    return summarize_outcomes(outcomes) | {
        "by_condition": summarize_by_condition(episodes),
        "gpu_models": count_by(episodes, "cuda_device_name"),
        "thresholds": sorted({threshold for threshold in (episode_threshold(episode) for episode in episodes) if threshold is not None}),
    }


def summarize_by_condition(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for episode in episodes:
        grouped.setdefault(condition_key(episode), []).append(episode)
    return {key: summarize_outcomes([episode_outcome(episode) for episode in items]) for key, items in sorted(grouped.items())}


def summarize_outcomes(outcomes: Sequence[Mapping[str, Any]], *, include_ci: bool = True) -> dict[str, Any]:
    total = len(outcomes)
    attempted = [item for item in outcomes if not bool(item["abstained"])]
    attempted_count = len(attempted)
    success_count = sum(int(bool(item["success"])) for item in outcomes)
    failure_count = sum(int(bool(item["failure"])) for item in outcomes)
    timeout_count = sum(int(bool(item["timeout"])) for item in outcomes)
    abstained_count = sum(int(bool(item["abstained"])) for item in outcomes)
    utility_values = [float(item["utility"]) for item in outcomes]
    payload: dict[str, Any] = {
        "episodes": total,
        "coverage": attempted_count / total if total else 0.0,
        "task_completion_rate": success_count / total if total else 0.0,
        "attempted_completion_rate": (
            sum(int(bool(item["success"])) for item in attempted) / attempted_count if attempted_count else 0.0
        ),
        "failure_rate": failure_count / total if total else 0.0,
        "failure_rate_attempted": (
            sum(int(bool(item["failure"])) for item in attempted) / attempted_count if attempted_count else 0.0
        ),
        "timeout_rate": timeout_count / total if total else 0.0,
        "abstention_rate": abstained_count / total if total else 0.0,
        "expected_utility": mean(utility_values) if utility_values else 0.0,
        "mean_policy_queries": mean(float(item["policy_queries"]) for item in outcomes) if outcomes else 0.0,
        "mean_runtime_risk_compute_seconds": mean(
            float(item["risk_compute_seconds"]) for item in outcomes if item["risk_compute_seconds"] is not None
        )
        if any(item["risk_compute_seconds"] is not None for item in outcomes)
        else 0.0,
    }
    if include_ci:
        payload["ci95"] = {
            "coverage": bootstrap_ci(outcomes, coverage_stat, samples=500),
            "task_completion_rate": bootstrap_ci(outcomes, task_completion_stat, samples=500),
            "attempted_completion_rate": bootstrap_ci(outcomes, attempted_completion_stat, samples=500),
            "failure_rate_attempted": bootstrap_ci(outcomes, failure_attempted_stat, samples=500),
            "timeout_rate": bootstrap_ci(outcomes, timeout_stat, samples=500),
            "abstention_rate": bootstrap_ci(outcomes, abstention_stat, samples=500),
            "expected_utility": bootstrap_ci(utility_values, mean_float, samples=500),
        }
    return payload


def comparison_against_direct(metrics: Mapping[str, Any], direct: Mapping[str, Any]) -> dict[str, float]:
    return {
        "utility_delta_vs_direct": float(metrics["expected_utility"]) - float(direct["expected_utility"]),
        "attempted_failure_delta_vs_direct": float(metrics["failure_rate_attempted"]) - float(direct["failure_rate_attempted"]),
        "completion_delta_vs_direct": float(metrics["task_completion_rate"]) - float(direct["task_completion_rate"]),
        "coverage_delta_vs_direct": float(metrics["coverage"]) - float(direct["coverage"]),
    }


def random_abstain_matched_coverage(
    direct_episodes: Sequence[Mapping[str, Any]],
    *,
    target_coverage: float,
    samples: int,
) -> dict[str, Any]:
    total = len(direct_episodes)
    abstain_count = max(0, min(total, int(round((1.0 - target_coverage) * total))))
    seed_metrics = []
    for seed in range(samples):
        rng = random.Random(seed)
        abstain_indexes = set(rng.sample(range(total), abstain_count)) if abstain_count else set()
        outcomes = []
        for idx, episode in enumerate(direct_episodes):
            if idx in abstain_indexes:
                outcomes.append(synthetic_abstain_outcome(episode))
            else:
                outcomes.append(episode_outcome(episode))
        seed_metrics.append(summarize_outcomes(outcomes, include_ci=False))
    return aggregate_seed_metrics(seed_metrics) | {
        "samples": samples,
        "target_coverage": target_coverage,
        "abstain_count": abstain_count,
        "source": "randomly abstains on direct_openpi episodes at matched coverage",
    }


def oracle_abstain_upper_bound(
    direct_episodes: Sequence[Mapping[str, Any]],
    *,
    target_coverage: float,
) -> dict[str, Any]:
    total = len(direct_episodes)
    abstain_count = max(0, min(total, int(round((1.0 - target_coverage) * total))))
    ranked = sorted(range(total), key=lambda idx: oracle_failure_priority(direct_episodes[idx]), reverse=True)
    abstain_indexes = set(ranked[:abstain_count])
    outcomes = [
        synthetic_abstain_outcome(episode) if idx in abstain_indexes else episode_outcome(episode)
        for idx, episode in enumerate(direct_episodes)
    ]
    return summarize_outcomes(outcomes) | {
        "target_coverage": target_coverage,
        "abstain_count": abstain_count,
        "source": "diagnostic upper bound; abstains on actual direct failures first and is not deployable",
    }


def aggregate_seed_metrics(seed_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = [
        "coverage",
        "task_completion_rate",
        "attempted_completion_rate",
        "failure_rate_attempted",
        "timeout_rate",
        "abstention_rate",
        "expected_utility",
        "mean_policy_queries",
    ]
    output = {}
    for key in keys:
        values = [float(row[key]) for row in seed_metrics]
        values_sorted = sorted(values)
        output[key] = mean(values)
        output[f"{key}_ci95"] = {
            "low": values_sorted[int(0.025 * (len(values_sorted) - 1))],
            "high": values_sorted[int(0.975 * (len(values_sorted) - 1))],
            "mean": mean(values),
            "samples": len(values),
        }
    return output


def answer_analysis_questions(
    metrics_by_label: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, float]],
    matched_baselines: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    direct = metrics_by_label[DIRECT_LABEL]
    answers: dict[str, Any] = {}
    for label in (SIGLIP_0933_LABEL, SIGLIP_0986_LABEL):
        random_baseline = matched_baselines[label]["random_abstain_matched_coverage"]
        answers[label] = {
            "beats_direct_on_utility": comparisons[label]["utility_delta_vs_direct"] > 0.0,
            "reduces_attempted_failure_vs_direct": comparisons[label]["attempted_failure_delta_vs_direct"] < 0.0,
            "beats_random_abstain_matched_coverage_on_utility": (
                float(metrics_by_label[label]["expected_utility"]) > float(random_baseline["expected_utility"])
            ),
            "beats_random_abstain_matched_coverage_on_attempted_failure": (
                float(metrics_by_label[label]["failure_rate_attempted"]) < float(random_baseline["failure_rate_attempted"])
            ),
            "high_occlusion_abstention_rate": high_occlusion_abstention_rate(metrics_by_label[label]),
            "utility_delta_vs_direct": comparisons[label]["utility_delta_vs_direct"],
            "attempted_failure_delta_vs_direct": comparisons[label]["attempted_failure_delta_vs_direct"],
        }
    best_tradeoff = max((SIGLIP_0933_LABEL, SIGLIP_0986_LABEL), key=lambda label: float(metrics_by_label[label]["expected_utility"]))
    answers["best_deployable_tradeoff"] = best_tradeoff
    answers["direct_openpi_utility"] = direct["expected_utility"]
    answers["direct_openpi_attempted_failure"] = direct["failure_rate_attempted"]
    answers["claim_strength"] = claim_strength(answers)
    return answers


def high_occlusion_abstention_rate(metrics: Mapping[str, Any]) -> float:
    condition = metrics.get("by_condition", {}).get("occlusion:1.00")
    return float(condition.get("abstention_rate", 0.0)) if isinstance(condition, Mapping) else 0.0


def claim_strength(answers: Mapping[str, Any]) -> str:
    best = str(answers.get("best_deployable_tradeoff"))
    best_answers = answers.get(best, {})
    if (
        isinstance(best_answers, Mapping)
        and best_answers.get("beats_direct_on_utility")
        and best_answers.get("reduces_attempted_failure_vs_direct")
        and best_answers.get("beats_random_abstain_matched_coverage_on_utility")
    ):
        return "strong_runtime_supervision_result"
    if isinstance(best_answers, Mapping) and best_answers.get("reduces_attempted_failure_vs_direct"):
        return "risk_filtering_result_with_tradeoffs"
    return "inconclusive_or_negative"


def episode_outcome(episode: Mapping[str, Any]) -> dict[str, Any]:
    terminal = str(episode.get("terminal_label", episode.get("terminal_reason", "")))
    failure_label = str(episode.get("failure_label", ""))
    abstained = terminal == "abstained" or failure_label == "abstained"
    success = bool(episode.get("success", False)) and not abstained
    timeout = bool(episode.get("timeout", False)) or terminal == "timeout"
    failure = (not success) and (not abstained)
    metadata = episode.get("metadata", {}) if isinstance(episode.get("metadata"), Mapping) else {}
    decision = episode.get("runtime_supervisor_decision") or metadata.get("runtime_supervisor_decision")
    risk_compute_seconds = decision.get("risk_compute_seconds") if isinstance(decision, Mapping) else None
    episode_length = int(episode.get("episode_length", 0) or 0)
    return {
        "success": success,
        "timeout": timeout,
        "abstained": abstained,
        "failure": failure,
        "episode_length": episode_length,
        "policy_queries": int(metadata.get("action_queries", 0) or 0),
        "runtime_risk": decision.get("predicted_risk") if isinstance(decision, Mapping) else None,
        "risk_compute_seconds": risk_compute_seconds,
        "utility": expected_utility(
            success=success,
            timeout=timeout,
            abstained=abstained,
            episode_length=episode_length,
        ),
    }


def synthetic_abstain_outcome(episode: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "success": False,
        "timeout": False,
        "abstained": True,
        "failure": False,
        "episode_length": 10,
        "policy_queries": 2,
        "runtime_risk": None,
        "risk_compute_seconds": 0.0,
        "utility": expected_utility(success=False, timeout=False, abstained=True, episode_length=10),
    }


def oracle_failure_priority(episode: Mapping[str, Any]) -> tuple[int, int, int]:
    outcome = episode_outcome(episode)
    return (
        int(bool(outcome["failure"])),
        int(bool(outcome["timeout"])),
        int(not bool(outcome["success"])),
    )


def episode_threshold(episode: Mapping[str, Any]) -> float | None:
    metadata = episode.get("metadata", {}) if isinstance(episode.get("metadata"), Mapping) else {}
    decision = episode.get("runtime_supervisor_decision") or metadata.get("runtime_supervisor_decision")
    if isinstance(decision, Mapping) and decision.get("threshold") is not None:
        return float(decision["threshold"])
    return None


def episode_key(episode: Mapping[str, Any]) -> tuple[Any, ...]:
    metadata = episode.get("metadata", {}) if isinstance(episode.get("metadata"), Mapping) else {}
    params = episode.get("stressor_params", {}) if isinstance(episode.get("stressor_params"), Mapping) else {}
    return (
        str(episode.get("libero_suite", episode.get("suite", ""))),
        int(episode.get("libero_task_id", episode.get("task_id", -1))),
        str(episode.get("stressor_name", "")),
        round(float(params.get("severity", 0.0)), 2),
        int(episode.get("seed", metadata.get("seed", -1))),
        int(metadata.get("episode_index", -1)),
        int(metadata.get("init_state_index", -1)),
    )


def condition_key(episode: Mapping[str, Any]) -> str:
    params = episode.get("stressor_params", {}) if isinstance(episode.get("stressor_params"), Mapping) else {}
    return f"{episode.get('stressor_name', '')}:{float(params.get('severity', 0.0)):.2f}"


def count_by(episodes: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for episode in episodes:
        value = str(episode.get(key, "unknown"))
        output[value] = output.get(value, 0) + 1
    return dict(sorted(output.items()))


def coverage_stat(outcomes: Sequence[Mapping[str, Any]]) -> float:
    return sum(1.0 for item in outcomes if not bool(item["abstained"])) / len(outcomes) if outcomes else 0.0


def task_completion_stat(outcomes: Sequence[Mapping[str, Any]]) -> float:
    return sum(float(item["success"]) for item in outcomes) / len(outcomes) if outcomes else 0.0


def attempted_completion_stat(outcomes: Sequence[Mapping[str, Any]]) -> float:
    attempted = [item for item in outcomes if not bool(item["abstained"])]
    return sum(float(item["success"]) for item in attempted) / len(attempted) if attempted else 0.0


def failure_attempted_stat(outcomes: Sequence[Mapping[str, Any]]) -> float:
    attempted = [item for item in outcomes if not bool(item["abstained"])]
    return sum(float(item["failure"]) for item in attempted) / len(attempted) if attempted else 0.0


def timeout_stat(outcomes: Sequence[Mapping[str, Any]]) -> float:
    return sum(float(item["timeout"]) for item in outcomes) / len(outcomes) if outcomes else 0.0


def abstention_stat(outcomes: Sequence[Mapping[str, Any]]) -> float:
    return sum(float(item["abstained"]) for item in outcomes) / len(outcomes) if outcomes else 0.0


def mean_float(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
