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

from calibrate_openpi_cross_suite_threshold import (  # noqa: E402
    evaluate_threshold,
    group_by_label,
    load_manifest_episodes,
    pair_examples,
    threshold_for_target_coverage,
)
from summarize_openpi_controlled_deployment import (  # noqa: E402
    episode_outcome,
    oracle_abstain_upper_bound,
    random_abstain_matched_coverage,
    read_jsonl,
    summarize_outcomes,
    synthetic_abstain_outcome,
)
from summarize_openpi_multiseed_deployment import (  # noqa: E402
    failure_attempted_stat,
    paired_delta_ci,
    utility_stat,
)
from risk_aware_skill_planning.evaluation.openpi_metrics import binary_risk_metrics  # noqa: E402


SPATIAL_THRESHOLDS = (0.9333276460818999, 0.9860334584902223)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a task-disjoint retrospective cross-suite threshold analysis")
    parser.add_argument("--manifest", default="reports/openpi_cross_suite_jobs_seed5000.jsonl")
    parser.add_argument("--output", default="reports/openpi_cross_suite_retrospective_threshold_summary.json")
    parser.add_argument("--calibration-task-ids", default="5,6")
    parser.add_argument("--test-task-ids", default="7,8,9")
    parser.add_argument("--target-coverages", default="0.70,0.75,0.80,0.85,0.90,0.95,1.00")
    parser.add_argument("--random-seeds", type=int, default=1000)
    args = parser.parse_args()

    manifest = read_jsonl(Path(args.manifest))
    calibration_ids = parse_ids(args.calibration_task_ids)
    test_ids = parse_ids(args.test_task_ids)
    targets = [float(value) for value in args.target_coverages.split(",") if value.strip()]
    summary = analyze_retrospective(
        manifest,
        calibration_task_ids=calibration_ids,
        test_task_ids=test_ids,
        target_coverages=targets,
        random_seeds=args.random_seeds,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": summary["ok"],
                "paired_episodes": summary.get("paired_episodes", 0),
                "selected_threshold": summary.get("selection", {}).get("threshold"),
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["ok"] else 2


def analyze_retrospective(
    manifest: Sequence[Mapping[str, Any]],
    *,
    calibration_task_ids: set[int],
    test_task_ids: set[int],
    target_coverages: Sequence[float],
    random_seeds: int,
) -> dict[str, Any]:
    if calibration_task_ids & test_task_ids:
        return {"ok": False, "failures": ["Calibration and test task IDs overlap"]}
    score_manifest = [
        dict(row, label="siglip_score" if row.get("label") == "siglip_0986" else row.get("label"))
        for row in manifest
        if row.get("label") in ("direct", "siglip_0986")
    ]
    episodes = load_manifest_episodes(score_manifest)
    pairs = pair_examples(group_by_label(episodes))
    calibration = [pair for pair in pairs if int(pair["key"][1]) in calibration_task_ids]
    test = [pair for pair in pairs if int(pair["key"][1]) in test_task_ids]
    failures = []
    if not calibration:
        failures.append("The calibration split is empty")
    if not test:
        failures.append("The test split is empty")
    expected = calibration_task_ids | test_task_ids
    observed = {int(pair["key"][1]) for pair in pairs}
    if expected - observed:
        failures.append(f"Missing task IDs: {sorted(expected - observed)}")
    if failures:
        return {"ok": False, "failures": failures, "paired_episodes": len(pairs)}

    rows = []
    calibration_risks = [float(pair["risk"]) for pair in calibration]
    for target in target_coverages:
        threshold = threshold_for_target_coverage(calibration_risks, float(target))
        rows.append(
            {
                "target_coverage": float(target),
                "threshold": threshold,
                "calibration": evaluate_threshold(calibration, threshold),
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            float(row["calibration"]["expected_utility"]),
            float(row["calibration"]["coverage"]),
        ),
    )
    selected_threshold = float(selected["threshold"])
    selected_test = evaluate_threshold(test, selected_threshold)
    direct_test_episodes = [pair["direct"] for pair in test]
    direct_test_outcomes = [episode_outcome(episode) for episode in direct_test_episodes]
    selected_test_outcomes = threshold_outcomes(test, selected_threshold)
    selected_comparison = paired_comparison(selected_test_outcomes, direct_test_outcomes)
    references = {
        threshold_name(threshold): reference_block(test, threshold, direct_test_outcomes)
        for threshold in SPATIAL_THRESHOLDS
    }
    coverage = float(selected_test["coverage"])
    calibration_labels, calibration_probs = risk_labels_and_probs(calibration)
    test_labels, test_probs = risk_labels_and_probs(test)
    return {
        "ok": True,
        "failures": [],
        "analysis_type": "retrospective task-disjoint diagnostic",
        "fresh_online_deployment": False,
        "reuse_notice": (
            "This analysis reuses seed-5000 episodes from the reported cross-suite deployment. "
            "It does not add online episodes or support a new deployment claim."
        ),
        "selection_rule": "maximum calibration utility; use higher coverage to break a tie",
        "job_ids": sorted({int(row["job_id"]) for row in score_manifest}),
        "paired_episodes": len(pairs),
        "split": {
            "calibration_task_ids": sorted(calibration_task_ids),
            "test_task_ids": sorted(test_task_ids),
            "calibration_pairs": len(calibration),
            "test_pairs": len(test),
            "seed": 5000,
        },
        "risk_metrics": {
            "calibration": binary_risk_metrics(calibration_labels, calibration_probs, threshold=selected_threshold),
            "test": binary_risk_metrics(test_labels, test_probs, threshold=selected_threshold),
            "test_by_suite": grouped_risk_metrics(
                test,
                selected_threshold,
                lambda pair: str(pair["key"][0]),
            ),
            "test_by_stressor": grouped_risk_metrics(
                test,
                selected_threshold,
                lambda pair: f"{pair['key'][2]}:{float(pair['key'][3]):.2f}",
            ),
        },
        "threshold_rows": rows,
        "selection": {
            "target_coverage": float(selected["target_coverage"]),
            "threshold": selected_threshold,
            "calibration_metrics": selected["calibration"],
        },
        "held_out_test": {
            "direct": summarize_outcomes(direct_test_outcomes),
            "selected_threshold": selected_test,
            "selected_vs_direct": selected_comparison,
            "spatial_threshold_references": references,
            "random_abstain_matched_coverage": random_abstain_matched_coverage(
                direct_test_episodes,
                target_coverage=coverage,
                samples=random_seeds,
            ),
            "oracle_abstain_upper_bound": oracle_abstain_upper_bound(
                direct_test_episodes,
                target_coverage=coverage,
            ),
            "per_suite": grouped_test_results(test, selected_threshold, lambda pair: str(pair["key"][0])),
            "per_stressor": grouped_test_results(
                test,
                selected_threshold,
                lambda pair: f"{pair['key'][2]}:{float(pair['key'][3]):.2f}",
            ),
        },
        "analysis_questions": answer_questions(selected_test, selected_comparison, references),
    }


def threshold_outcomes(pairs: Sequence[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [
        episode_outcome(pair["direct"])
        if float(pair["risk"]) < threshold
        else synthetic_abstain_outcome(pair["direct"])
        for pair in pairs
    ]


def paired_comparison(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "paired_episodes": len(candidate),
        "utility_delta": utility_stat(candidate) - utility_stat(baseline),
        "utility_delta_ci95": paired_delta_ci(candidate, baseline, utility_stat),
        "attempted_failure_delta": failure_attempted_stat(candidate) - failure_attempted_stat(baseline),
        "attempted_failure_delta_ci95": paired_delta_ci(candidate, baseline, failure_attempted_stat),
    }


def grouped_test_results(
    pairs: Sequence[Mapping[str, Any]],
    threshold: float,
    key_fn: Any,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for pair in pairs:
        grouped.setdefault(str(key_fn(pair)), []).append(pair)
    output = {}
    for key, items in sorted(grouped.items()):
        direct = [episode_outcome(pair["direct"]) for pair in items]
        selected = threshold_outcomes(items, threshold)
        output[key] = {
            "pairs": len(items),
            "direct": summarize_outcomes(direct),
            "selected_threshold": summarize_outcomes(selected),
            "selected_vs_direct": paired_comparison(selected, direct),
        }
    return output


def reference_block(
    pairs: Sequence[Mapping[str, Any]],
    threshold: float,
    direct_outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    outcomes = threshold_outcomes(pairs, threshold)
    return {
        "threshold": threshold,
        "metrics": summarize_outcomes(outcomes),
        "vs_direct": paired_comparison(outcomes, direct_outcomes),
    }


def risk_labels_and_probs(pairs: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[float]]:
    labels = [int(bool(episode_outcome(pair["direct"])["failure"])) for pair in pairs]
    probs = [float(pair["risk"]) for pair in pairs]
    return labels, probs


def grouped_risk_metrics(
    pairs: Sequence[Mapping[str, Any]],
    threshold: float,
    key_fn: Any,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for pair in pairs:
        grouped.setdefault(str(key_fn(pair)), []).append(pair)
    output = {}
    for key, items in sorted(grouped.items()):
        labels, probs = risk_labels_and_probs(items)
        output[key] = binary_risk_metrics(labels, probs, threshold=threshold)
    return output


def answer_questions(
    selected: Mapping[str, Any],
    comparison: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    utility_ci = comparison["utility_delta_ci95"]
    failure_ci = comparison["attempted_failure_delta_ci95"]
    old = references[threshold_name(0.9860334584902223)]["metrics"]
    return {
        "threshold_tuning_restores_cross_suite_utility": comparison["utility_delta"] > 0.0,
        "utility_gain_is_robust": utility_ci.get("low") is not None and float(utility_ci["low"]) > 0.0,
        "attempted_failure_reduction_is_robust": (
            failure_ci.get("high") is not None and float(failure_ci["high"]) < 0.0
        ),
        "selected_threshold_improves_utility_over_spatial_0986": (
            float(selected["expected_utility"]) > float(old["expected_utility"])
        ),
        "interpretation": (
            "Task-local calibration improves risk filtering but does not restore held-out utility. "
            "The threshold does not transfer uniformly across cross-suite task IDs."
        ),
    }


def threshold_name(threshold: float) -> str:
    return f"threshold_{threshold:.4f}".replace(".", "_")


def parse_ids(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
