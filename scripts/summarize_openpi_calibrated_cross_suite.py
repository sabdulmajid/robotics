#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_openpi_controlled_deployment import (  # noqa: E402
    condition_key,
    episode_key,
    oracle_abstain_upper_bound,
    random_abstain_matched_coverage,
    read_jsonl,
    summarize_episode_set,
)
from summarize_openpi_multiseed_deployment import compare_against_direct_with_ci  # noqa: E402


DIRECT_LABEL = "direct"
SPATIAL_LABEL = "spatial_0986"
CALIBRATED_LABEL = "cross_calibrated"
LABEL_ORDER = [DIRECT_LABEL, SPATIAL_LABEL, CALIBRATED_LABEL]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the held-out cross-suite threshold deployment")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--output", default="reports/openpi_cross_suite_calibrated_deployment.json")
    parser.add_argument("--random-seeds", type=int, default=1000)
    args = parser.parse_args()

    manifest = read_jsonl(Path(args.manifest))
    calibration = json.loads(Path(args.calibration_summary).read_text(encoding="utf-8"))
    episodes = load_manifest_episodes(manifest)
    summary = summarize_deployment(manifest, calibration, episodes, random_seeds=args.random_seeds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": summary["ok"], "episodes": summary["episodes"], "output": str(output)}, indent=2))
    return 0 if summary["ok"] else 2


def summarize_deployment(
    manifest: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    *,
    random_seeds: int,
) -> dict[str, Any]:
    by_label = group_by_label(episodes)
    grid = validate_grid(by_label)
    failures = list(grid["failures"])
    expected_threshold = calibration.get("selection", {}).get("threshold")
    manifest_thresholds = {
        float(row["threshold"])
        for row in manifest
        if row.get("label") == CALIBRATED_LABEL and row.get("threshold") is not None
    }
    if expected_threshold is None:
        failures.append("The calibration summary does not contain a selected threshold")
    elif manifest_thresholds != {float(expected_threshold)}:
        failures.append("The deployed threshold does not match the frozen calibration threshold")

    overall = summarize_group(by_label, random_seeds=random_seeds)
    per_suite = summarize_named_groups(episodes, lambda episode: str(episode.get("libero_suite", "unknown")), random_seeds)
    per_stressor = summarize_named_groups(episodes, condition_key, random_seeds)
    return {
        "ok": not failures,
        "failures": failures,
        "experiment": "held_out_cross_suite_calibrated_deployment",
        "data_boundary": (
            "The threshold uses tasks 0 through 4 and seed 7500. "
            "This deployment uses tasks 5 through 9 and seed 8000."
        ),
        "episodes": len(episodes),
        "manifest": list(manifest),
        "job_ids": [int(row["job_id"]) for row in manifest],
        "frozen_threshold": expected_threshold,
        "calibration_job_ids": calibration.get("job_ids", []),
        "grid_checks": grid,
        "metrics": overall["metrics"],
        "comparisons": overall["comparisons"],
        "matched_baselines": overall["matched_baselines"],
        "per_suite": per_suite,
        "per_stressor": per_stressor,
        "analysis_questions": answer_questions(overall),
    }


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


def group_by_label(episodes: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    output = {label: [] for label in LABEL_ORDER}
    for episode in episodes:
        output.setdefault(str(episode["_manifest"]["label"]), []).append(episode)
    return output


def validate_grid(by_label: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    failures = []
    keys = {label: {episode_key(episode) for episode in by_label.get(label, [])} for label in LABEL_ORDER}
    direct = keys[DIRECT_LABEL]
    for label in LABEL_ORDER:
        if not keys[label]:
            failures.append(f"No episodes were found for label {label}")
        missing = direct - keys[label]
        extra = keys[label] - direct
        if missing:
            failures.append(f"{label} is missing {len(missing)} direct episode keys")
        if extra:
            failures.append(f"{label} has {len(extra)} extra episode keys")
    return {
        "ok": not failures,
        "failures": failures,
        "episode_counts": {label: len(by_label.get(label, [])) for label in LABEL_ORDER},
        "unique_key_counts": {label: len(keys[label]) for label in LABEL_ORDER},
    }


def summarize_named_groups(
    episodes: Sequence[Mapping[str, Any]],
    key_fn: Any,
    random_seeds: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for episode in episodes:
        grouped.setdefault(str(key_fn(episode)), []).append(episode)
    return {
        key: summarize_group(group_by_label(items), random_seeds=random_seeds)
        for key, items in sorted(grouped.items())
    }


def summarize_group(
    by_label: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    random_seeds: int,
) -> dict[str, Any]:
    metrics = {label: summarize_episode_set(by_label.get(label, [])) for label in LABEL_ORDER}
    comparisons = {
        f"{label}_vs_direct": compare_against_direct_with_ci(
            by_label.get(label, []),
            by_label.get(DIRECT_LABEL, []),
        )
        for label in (SPATIAL_LABEL, CALIBRATED_LABEL)
    }
    comparisons["cross_calibrated_vs_spatial_0986"] = compare_against_direct_with_ci(
        by_label.get(CALIBRATED_LABEL, []),
        by_label.get(SPATIAL_LABEL, []),
    )
    matched = {}
    direct = list(by_label.get(DIRECT_LABEL, []))
    for label in (SPATIAL_LABEL, CALIBRATED_LABEL):
        coverage = float(metrics[label]["coverage"])
        matched[label] = {
            "random_abstain_matched_coverage": random_abstain_matched_coverage(
                direct,
                target_coverage=coverage,
                samples=random_seeds,
            ),
            "oracle_abstain_upper_bound": oracle_abstain_upper_bound(
                direct,
                target_coverage=coverage,
            ),
        }
    return {"metrics": metrics, "comparisons": comparisons, "matched_baselines": matched}


def answer_questions(overall: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = overall["comparisons"]
    calibrated = overall["metrics"][CALIBRATED_LABEL]
    random_baseline = overall["matched_baselines"][CALIBRATED_LABEL]["random_abstain_matched_coverage"]
    direct_comparison = comparisons["cross_calibrated_vs_direct"]
    old_comparison = comparisons["cross_calibrated_vs_spatial_0986"]
    utility_ci = direct_comparison["utility_delta_ci95"]
    failure_ci = direct_comparison["attempted_failure_delta_ci95"]
    return {
        "utility_gain_vs_direct": direct_comparison["utility_delta_vs_direct"] > 0.0,
        "utility_gain_vs_direct_is_robust": utility_ci.get("low") is not None and float(utility_ci["low"]) > 0.0,
        "attempted_failure_reduction_vs_direct": direct_comparison["attempted_failure_delta_vs_direct"] < 0.0,
        "attempted_failure_reduction_vs_direct_is_robust": (
            failure_ci.get("high") is not None and float(failure_ci["high"]) < 0.0
        ),
        "calibration_improves_utility_vs_spatial_threshold": old_comparison["utility_delta_vs_direct"] > 0.0,
        "beats_random_abstention_on_utility": (
            float(calibrated["expected_utility"]) > float(random_baseline["expected_utility"])
        ),
        "beats_random_abstention_on_attempted_failure": (
            float(calibrated["failure_rate_attempted"]) < float(random_baseline["failure_rate_attempted"])
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
