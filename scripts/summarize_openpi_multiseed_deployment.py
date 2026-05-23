#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_openpi_controlled_deployment import (  # noqa: E402
    DIRECT_LABEL,
    SIGLIP_0933_LABEL,
    SIGLIP_0986_LABEL,
    condition_key,
    episode_key,
    episode_outcome,
    oracle_abstain_upper_bound,
    random_abstain_matched_coverage,
    read_jsonl,
    summarize_episode_set,
)


LABEL_ORDER = [DIRECT_LABEL, "fixed", SIGLIP_0933_LABEL, SIGLIP_0986_LABEL]
SIGLIP_LABELS = [SIGLIP_0933_LABEL, SIGLIP_0986_LABEL]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize multiseed and cross-suite OpenPI runtime deployments")
    parser.add_argument("--manifest", action="append", required=True, help="JSONL manifest path. May be repeated.")
    parser.add_argument("--output", default="reports/openpi_runtime_multiseed_summary.json")
    parser.add_argument("--random-seeds", type=int, default=1000)
    args = parser.parse_args()

    manifest = []
    for path in args.manifest:
        manifest.extend(normalize_manifest_row(row) for row in read_jsonl(Path(path)))
    episodes = load_manifest_episodes(manifest)
    summary = summarize_multiseed_deployment(manifest, episodes, random_seeds=args.random_seeds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": summary["ok"], "episodes": summary["episodes"], "output": str(output)}, indent=2))
    return 0 if summary["ok"] else 2


def summarize_multiseed_deployment(
    manifest: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    *,
    random_seeds: int,
) -> dict[str, Any]:
    failures: list[str] = []
    job_ids = [int(row["job_id"]) for row in manifest]
    missing_jobs = [job_id for job_id in job_ids if not any(int(ep["_manifest"]["job_id"]) == job_id for ep in episodes)]
    if missing_jobs:
        failures.append(f"Missing rollout JSONL for {len(missing_jobs)} jobs: {missing_jobs[:10]}")

    groups = {"all_new": summarize_group(episodes, random_seeds=random_seeds, validate_grid=False)}
    for experiment in sorted({str(ep["_manifest"]["experiment"]) for ep in episodes}):
        groups[experiment] = summarize_group(
            [ep for ep in episodes if ep["_manifest"]["experiment"] == experiment],
            random_seeds=random_seeds,
        )
    failures.extend(group_failures(groups))
    per_seed = summarize_named_groups(
        episodes,
        lambda ep: f"{ep['_manifest']['experiment']}:seed{ep['_manifest']['seed']}",
        random_seeds=random_seeds,
    )
    per_stressor = summarize_named_groups(
        episodes,
        lambda ep: f"{ep['_manifest']['experiment']}:{condition_key(ep)}",
        random_seeds=random_seeds,
    )
    per_suite = summarize_named_groups(
        episodes,
        lambda ep: f"{ep['_manifest']['experiment']}:{ep.get('libero_suite', ep.get('suite', 'unknown'))}",
        random_seeds=random_seeds,
    )

    return {
        "ok": not failures,
        "failures": failures,
        "episodes": len(episodes),
        "fresh_controlled_episodes": len(episodes),
        "manifest_rows": len(manifest),
        "job_ids": job_ids,
        "job_ids_by_experiment_label": job_ids_by_experiment_label(manifest),
        "groups": groups,
        "per_seed": per_seed,
        "per_stressor": per_stressor,
        "per_suite": per_suite,
        "analysis_questions": answer_analysis_questions(groups),
    }


def summarize_named_groups(
    episodes: Sequence[Mapping[str, Any]],
    key_fn: Callable[[Mapping[str, Any]], str],
    *,
    random_seeds: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for episode in episodes:
        grouped.setdefault(key_fn(episode), []).append(episode)
    return {
        key: summarize_group(items, random_seeds=random_seeds)
        for key, items in sorted(grouped.items())
        if items
    }


def summarize_group(
    episodes: Sequence[Mapping[str, Any]],
    *,
    random_seeds: int,
    validate_grid: bool = True,
) -> dict[str, Any]:
    episodes_by_label = group_by_label(episodes)
    labels = [label for label in LABEL_ORDER if episodes_by_label.get(label)]
    metrics = {label: summarize_episode_set(episodes_by_label[label]) for label in labels}
    direct_episodes = episodes_by_label.get(DIRECT_LABEL, [])
    grid_checks = validate_present_grids(episodes_by_label) if validate_grid else descriptive_grid_counts(episodes_by_label)
    comparisons = {}
    matched_baselines = {}
    if direct_episodes:
        for label in labels:
            if label == DIRECT_LABEL:
                continue
            comparisons[label] = compare_against_direct_with_ci(
                episodes_by_label[label],
                direct_episodes,
            )
        for label in SIGLIP_LABELS:
            if label in metrics:
                coverage = float(metrics[label]["coverage"])
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
        "episodes": len(episodes),
        "labels": labels,
        "grid_checks": grid_checks,
        "metrics": metrics,
        "comparisons_vs_direct": comparisons,
        "matched_baselines": matched_baselines,
        "analysis": answer_group_questions(metrics, comparisons, matched_baselines),
    }


def compare_against_direct_with_ci(
    label_episodes: Sequence[Mapping[str, Any]],
    direct_episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    label_by_key = {episode_key(ep): ep for ep in label_episodes}
    direct_by_key = {episode_key(ep): ep for ep in direct_episodes}
    keys = sorted(set(label_by_key) & set(direct_by_key))
    label_outcomes = [episode_outcome(label_by_key[key]) for key in keys]
    direct_outcomes = [episode_outcome(direct_by_key[key]) for key in keys]
    return {
        "paired_keys": len(keys),
        "utility_delta_vs_direct": metric_delta(label_outcomes, direct_outcomes, utility_stat),
        "utility_delta_ci95": paired_delta_ci(label_outcomes, direct_outcomes, utility_stat),
        "attempted_failure_delta_vs_direct": metric_delta(
            label_outcomes,
            direct_outcomes,
            failure_attempted_stat,
        ),
        "attempted_failure_delta_ci95": paired_delta_ci(
            label_outcomes,
            direct_outcomes,
            failure_attempted_stat,
        ),
        "completion_delta_vs_direct": metric_delta(label_outcomes, direct_outcomes, completion_stat),
        "completion_delta_ci95": paired_delta_ci(label_outcomes, direct_outcomes, completion_stat),
        "coverage_delta_vs_direct": metric_delta(label_outcomes, direct_outcomes, coverage_stat),
        "coverage_delta_ci95": paired_delta_ci(label_outcomes, direct_outcomes, coverage_stat),
    }


def paired_delta_ci(
    label_outcomes: Sequence[Mapping[str, Any]],
    direct_outcomes: Sequence[Mapping[str, Any]],
    metric: Callable[[Sequence[Mapping[str, Any]]], float],
    *,
    samples: int = 1000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    if not label_outcomes or len(label_outcomes) != len(direct_outcomes):
        return {"mean": None, "low": None, "high": None, "samples": 0}
    rng = random.Random(seed)
    estimates = []
    indexes = list(range(len(label_outcomes)))
    for _ in range(samples):
        sample_indexes = [rng.choice(indexes) for _ in indexes]
        estimates.append(
            metric([label_outcomes[idx] for idx in sample_indexes])
            - metric([direct_outcomes[idx] for idx in sample_indexes])
        )
    estimates.sort()
    return {
        "mean": mean(estimates),
        "low": estimates[max(0, int(0.025 * samples) - 1)],
        "high": estimates[min(samples - 1, int(0.975 * samples))],
        "samples": samples,
    }


def metric_delta(
    label_outcomes: Sequence[Mapping[str, Any]],
    direct_outcomes: Sequence[Mapping[str, Any]],
    metric: Callable[[Sequence[Mapping[str, Any]]], float],
) -> float:
    return metric(label_outcomes) - metric(direct_outcomes)


def validate_present_grids(episodes_by_label: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    failures = []
    direct_keys = {episode_key(ep) for ep in episodes_by_label.get(DIRECT_LABEL, [])}
    keys_by_label = {label: {episode_key(ep) for ep in episodes} for label, episodes in episodes_by_label.items()}
    if not direct_keys:
        failures.append("No direct_openpi episodes found for group")
    for label, keys in keys_by_label.items():
        if label == DIRECT_LABEL:
            continue
        missing = sorted(direct_keys - keys)
        extra = sorted(keys - direct_keys)
        if missing:
            failures.append(f"{label} is missing {len(missing)} direct-grid keys")
        if extra:
            failures.append(f"{label} has {len(extra)} keys outside direct grid")
    return {
        "ok": not failures,
        "failures": failures,
        "label_episode_counts": {
            label: len(episodes)
            for label, episodes in sorted(episodes_by_label.items())
        },
        "unique_grid_keys": {label: len(keys) for label, keys in sorted(keys_by_label.items())},
    }


def descriptive_grid_counts(episodes_by_label: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    keys_by_label = {label: {episode_key(ep) for ep in episodes} for label, episodes in episodes_by_label.items()}
    return {
        "ok": True,
        "failures": [],
        "label_episode_counts": {
            label: len(episodes)
            for label, episodes in sorted(episodes_by_label.items())
        },
        "unique_grid_keys": {label: len(keys) for label, keys in sorted(keys_by_label.items())},
        "note": "descriptive pooled group; same-grid validation is applied within experiment groups",
    }


def answer_group_questions(
    metrics: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
    matched_baselines: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    siglip_present = [label for label in SIGLIP_LABELS if label in metrics]
    best = max(siglip_present, key=lambda label: float(metrics[label]["expected_utility"])) if siglip_present else None
    answers: dict[str, Any] = {
        "best_deployable_tradeoff": best,
        "siglip_0933_safety_mode": False,
    }
    for label in siglip_present:
        comparison = comparisons.get(label, {})
        random_baseline = matched_baselines.get(label, {}).get("random_abstain_matched_coverage", {})
        answers[label] = {
            "utility_delta_vs_direct": comparison.get("utility_delta_vs_direct"),
            "utility_delta_ci95": comparison.get("utility_delta_ci95"),
            "attempted_failure_delta_vs_direct": comparison.get("attempted_failure_delta_vs_direct"),
            "attempted_failure_delta_ci95": comparison.get("attempted_failure_delta_ci95"),
            "beats_direct_on_utility": (comparison.get("utility_delta_vs_direct") or 0.0) > 0.0,
            "reduces_attempted_failure_vs_direct": (
                (comparison.get("attempted_failure_delta_vs_direct") or 0.0) < 0.0
            ),
            "utility_gain_ci_excludes_zero": ci_excludes_zero_positive(comparison.get("utility_delta_ci95")),
            "failure_reduction_ci_excludes_zero": ci_excludes_zero_negative(
                comparison.get("attempted_failure_delta_ci95")
            ),
            "beats_random_abstain_matched_coverage_on_utility": (
                float(metrics[label]["expected_utility"]) > float(random_baseline.get("expected_utility", -1e9))
            ),
            "beats_random_abstain_matched_coverage_on_attempted_failure": (
                float(metrics[label]["failure_rate_attempted"])
                < float(random_baseline.get("failure_rate_attempted", 1e9))
            ),
        }
    if SIGLIP_0933_LABEL in metrics and SIGLIP_0986_LABEL in metrics:
        answers["siglip_0933_safety_mode"] = (
            float(metrics[SIGLIP_0933_LABEL]["failure_rate_attempted"])
            < float(metrics[SIGLIP_0986_LABEL]["failure_rate_attempted"])
            and float(metrics[SIGLIP_0933_LABEL]["coverage"]) <= float(metrics[SIGLIP_0986_LABEL]["coverage"])
        )
    return answers


def answer_analysis_questions(groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    spatial = groups.get("spatial_multiseed", {}).get("analysis", {})
    cross = groups.get("cross_suite", {}).get("analysis", {})
    return {
        "spatial_best_deployable_tradeoff": spatial.get("best_deployable_tradeoff"),
        "spatial_siglip_0933_safety_mode": spatial.get("siglip_0933_safety_mode"),
        "cross_suite_best_deployable_tradeoff": cross.get("best_deployable_tradeoff"),
        "cross_suite_generalization_holds": cross_suite_generalization_holds(cross),
        "utility_gain_robust": bool(
            spatial.get(SIGLIP_0986_LABEL, {}).get("utility_gain_ci_excludes_zero")
        ),
        "attempted_failure_reduction_robust": bool(
            spatial.get(SIGLIP_0986_LABEL, {}).get("failure_reduction_ci_excludes_zero")
        ),
    }


def cross_suite_generalization_holds(cross_analysis: Mapping[str, Any]) -> bool:
    siglip = cross_analysis.get(SIGLIP_0986_LABEL, {})
    return bool(
        isinstance(siglip, Mapping)
        and siglip.get("reduces_attempted_failure_vs_direct")
        and siglip.get("beats_random_abstain_matched_coverage_on_attempted_failure")
    )


def ci_excludes_zero_positive(ci: Any) -> bool:
    return isinstance(ci, Mapping) and ci.get("low") is not None and float(ci["low"]) > 0.0


def ci_excludes_zero_negative(ci: Any) -> bool:
    return isinstance(ci, Mapping) and ci.get("high") is not None and float(ci["high"]) < 0.0


def group_by_label(episodes: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = {}
    for episode in episodes:
        output.setdefault(str(episode["_manifest"]["label"]), []).append(episode)
    return output


def load_manifest_episodes(manifest: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in manifest:
        path = Path("datasets/openpi_libero_rollouts") / f"openpi_rollouts_{int(row['job_id'])}.jsonl"
        if not path.exists():
            continue
        for episode in read_jsonl(path):
            item = dict(episode)
            item["_manifest"] = dict(row)
            output.append(item)
    return output


def normalize_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row)
    if "suites" not in output:
        output["suites"] = [output.get("suite", "unknown")]
    output["experiment"] = output.get("experiment", "unknown")
    output["label"] = output.get("label", output.get("mode", "unknown"))
    output["seed"] = int(output.get("seed", -1))
    output["job_id"] = int(output["job_id"])
    return output


def job_ids_by_experiment_label(manifest: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[int]]]:
    output: dict[str, dict[str, list[int]]] = {}
    for row in manifest:
        output.setdefault(str(row["experiment"]), {}).setdefault(str(row["label"]), []).append(int(row["job_id"]))
    return {
        experiment: {label: sorted(job_ids) for label, job_ids in sorted(labels.items())}
        for experiment, labels in sorted(output.items())
    }


def group_failures(groups: Mapping[str, Mapping[str, Any]]) -> list[str]:
    failures = []
    for name, group in groups.items():
        for failure in group.get("grid_checks", {}).get("failures", []):
            if name == "cross_suite" and "fixed" in failure:
                continue
            failures.append(f"{name}: {failure}")
    return failures


def utility_stat(outcomes: Sequence[Mapping[str, Any]]) -> float:
    return mean(float(item["utility"]) for item in outcomes) if outcomes else 0.0


def failure_attempted_stat(outcomes: Sequence[Mapping[str, Any]]) -> float:
    attempted = [item for item in outcomes if not bool(item["abstained"])]
    return sum(float(item["failure"]) for item in attempted) / len(attempted) if attempted else 0.0


def completion_stat(outcomes: Sequence[Mapping[str, Any]]) -> float:
    return sum(float(item["success"]) for item in outcomes) / len(outcomes) if outcomes else 0.0


def coverage_stat(outcomes: Sequence[Mapping[str, Any]]) -> float:
    return sum(1.0 for item in outcomes if not bool(item["abstained"])) / len(outcomes) if outcomes else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
