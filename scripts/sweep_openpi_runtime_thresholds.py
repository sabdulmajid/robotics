#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from risk_aware_skill_planning.evaluation.bootstrap import bootstrap_ci
from risk_aware_skill_planning.evaluation.risk_coverage import expected_utility


DIRECT_MODE = "direct_openpi"
FIXED_MODE = "fixed_task_prior_selective"
VISION_MODE = "vision_language_risk_selective"


def main() -> int:
    parser = argparse.ArgumentParser(description="Runtime threshold sweep for paired OpenPI/LIBERO SigLIP risk logs")
    parser.add_argument("--input", action="append", required=True, help="Input JSONL path or glob; may be repeated")
    parser.add_argument("--output", default="reports/openpi_runtime_siglip_eval_summary.json")
    parser.add_argument("--target-coverages", default="0.70,0.75,0.80,0.85,0.90,0.95,1.00")
    parser.add_argument("--calibration-task-max", type=int, default=4)
    args = parser.parse_args()

    paths = expand_inputs(args.input)
    episodes = [episode for path in paths for episode in read_jsonl(path)]
    target_coverages = [float(item) for item in args.target_coverages.split(",") if item.strip()]
    sweep = build_threshold_sweep(
        episodes,
        target_coverages=target_coverages,
        calibration_task_max=int(args.calibration_task_max),
    )
    output = Path(args.output)
    summary = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    summary["runtime_threshold_sweep"] = sweep
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": sweep["ok"],
                "paired_episodes": sweep["paired_episodes"],
                "output": str(output),
                "best_utility_target_coverage": sweep["best_operating_points"]["best_utility"]["target_coverage"],
                "best_utility": sweep["best_operating_points"]["best_utility"]["expected_utility"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if sweep["ok"] else 2


def build_threshold_sweep(
    episodes: Sequence[Mapping[str, Any]],
    *,
    target_coverages: Sequence[float],
    calibration_task_max: int,
) -> dict[str, Any]:
    examples = pair_runtime_examples(episodes)
    calibration = [example for example in examples if int(example["task_id"]) <= calibration_task_max]
    test = [example for example in examples if int(example["task_id"]) > calibration_task_max]
    if not calibration or not test:
        return {
            "ok": False,
            "paired_episodes": len(examples),
            "error": "Calibration/test split is empty",
        }
    direct_test = evaluate_actual_mode(test, DIRECT_MODE)
    direct_queries = max(float(direct_test["mean_policy_queries"]), 1e-9)
    baselines = {
        "all": baseline_block(examples),
        "calibration": baseline_block(calibration),
        "test": baseline_block(test),
    }
    threshold_rows = []
    for target in target_coverages:
        threshold = threshold_for_target_coverage([float(example["risk"]) for example in calibration], target)
        calibration_metrics = evaluate_threshold(calibration, threshold)
        test_metrics = evaluate_threshold(test, threshold)
        add_comparison_fields(test_metrics, direct_test, direct_queries)
        threshold_rows.append(
            {
                "target_coverage": float(target),
                "threshold": threshold,
                "threshold_source": "runtime_calibration_split",
                "calibration": calibration_metrics,
                "test": test_metrics,
            }
        )
    return {
        "ok": True,
        "evaluation_type": "paired runtime SigLIP risk scores with paired real direct OpenPI outcomes for accepted attempts",
        "counterfactual_note": (
            "Thresholds are selected from runtime calibration risk scores. Test outcomes use paired direct_openpi "
            "episodes when a threshold would attempt, and a 10-step runtime-prefix abstention proxy when it would reject."
        ),
        "paired_episodes": len(examples),
        "target_coverages": [float(value) for value in target_coverages],
        "split": {
            "method": "task_disjoint",
            "calibration_task_ids": sorted({int(example["task_id"]) for example in calibration}),
            "test_task_ids": sorted({int(example["task_id"]) for example in test}),
            "calibration_episodes": len(calibration),
            "test_episodes": len(test),
            "threshold_selection": "target coverage thresholds selected only on calibration examples",
        },
        "baselines": baselines,
        "threshold_rows": threshold_rows,
        "best_operating_points": best_operating_points(threshold_rows, direct_test),
    }


def pair_runtime_examples(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_mode_key: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for episode in episodes:
        by_mode_key[(str(episode.get("mode")),) + episode_key(episode)] = episode
    keys = sorted({episode_key(episode) for episode in episodes})
    examples = []
    for key in keys:
        direct = by_mode_key.get((DIRECT_MODE,) + key)
        fixed = by_mode_key.get((FIXED_MODE,) + key)
        vision = by_mode_key.get((VISION_MODE,) + key)
        if not direct or not fixed or not vision:
            continue
        decision = vision.get("runtime_supervisor_decision")
        if not isinstance(decision, Mapping):
            metadata = vision.get("metadata", {})
            decision = metadata.get("runtime_supervisor_decision") if isinstance(metadata, Mapping) else None
        if not isinstance(decision, Mapping) or decision.get("predicted_risk") is None:
            continue
        examples.append(
            {
                "key": key,
                "suite": key[0],
                "task_id": int(key[1]),
                "stressor_name": key[2],
                "stressor_severity": float(key[3]),
                "episode_index": int(key[4]),
                "init_state_index": int(key[5]),
                "risk": float(decision["predicted_risk"]),
                "threshold_logged": float(decision.get("threshold", 0.0) or 0.0),
                "direct": direct,
                "fixed": fixed,
                "vision": vision,
            }
        )
    return examples


def episode_key(episode: Mapping[str, Any]) -> tuple[Any, ...]:
    metadata = episode.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    params = episode.get("stressor_params", {})
    severity = float(params.get("severity", 0.0)) if isinstance(params, Mapping) else 0.0
    return (
        str(episode.get("libero_suite", episode.get("suite", ""))),
        int(episode.get("libero_task_id", episode.get("task_id", -1))),
        str(episode.get("stressor_name", "")),
        round(severity, 2),
        int(metadata.get("episode_index", -1)),
        int(metadata.get("init_state_index", -1)),
    )


def threshold_for_target_coverage(risks: Sequence[float], target: float) -> float:
    if not risks:
        return 1.0
    ordered = sorted(float(value) for value in risks)
    accepted = max(1, min(len(ordered), int(math.ceil(float(target) * len(ordered)))))
    base = ordered[accepted - 1]
    return float(base + max(1e-12, abs(base) * 1e-12))


def baseline_block(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    direct = evaluate_actual_mode(examples, DIRECT_MODE)
    fixed = evaluate_actual_mode(examples, FIXED_MODE)
    vision = evaluate_actual_mode(examples, VISION_MODE)
    direct_queries = max(float(direct["mean_policy_queries"]), 1e-9)
    for payload in (direct, fixed, vision):
        payload["policy_query_overhead_vs_direct"] = float(payload["mean_policy_queries"]) / direct_queries
    return {
        DIRECT_MODE: direct,
        FIXED_MODE: fixed,
        VISION_MODE: vision,
    }


def evaluate_actual_mode(examples: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    episode_field = {
        DIRECT_MODE: "direct",
        FIXED_MODE: "fixed",
        VISION_MODE: "vision",
    }[mode]
    outcomes = [episode_outcome(example[episode_field]) for example in examples]
    return summarize_outcomes(outcomes)


def evaluate_threshold(examples: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    outcomes = []
    for example in examples:
        if float(example["risk"]) < threshold:
            outcome = episode_outcome(example["direct"])
            outcome["threshold_decision"] = "attempt"
        else:
            outcome = synthetic_abstain_outcome(example)
            outcome["threshold_decision"] = "abstain"
        outcomes.append(outcome)
    return summarize_outcomes(outcomes)


def episode_outcome(episode: Mapping[str, Any]) -> dict[str, Any]:
    terminal = str(episode.get("terminal_label", episode.get("terminal_reason", "")))
    failure_label = str(episode.get("failure_label", ""))
    abstained = terminal == "abstained" or failure_label == "abstained"
    success = bool(episode.get("success", False)) and not abstained
    timeout = bool(episode.get("timeout", False)) or terminal == "timeout"
    failure = (not success) and (not abstained)
    metadata = episode.get("metadata", {}) if isinstance(episode.get("metadata"), Mapping) else {}
    episode_length = int(episode.get("episode_length", 0) or 0)
    return {
        "success": success,
        "timeout": timeout,
        "abstained": abstained,
        "failure": failure,
        "episode_length": episode_length,
        "policy_queries": int(metadata.get("action_queries", 0) or 0),
    }


def synthetic_abstain_outcome(example: Mapping[str, Any]) -> dict[str, Any]:
    vision = example["vision"]
    decision = vision.get("runtime_supervisor_decision")
    if not isinstance(decision, Mapping):
        metadata = vision.get("metadata", {})
        decision = metadata.get("runtime_supervisor_decision", {}) if isinstance(metadata, Mapping) else {}
    prefix_steps = int(decision.get("prefix_steps_observed", 10) if isinstance(decision, Mapping) else 10)
    replan_steps = int(vision.get("n_action_steps", vision.get("action_horizon", 5)) or 5)
    return {
        "success": False,
        "timeout": False,
        "abstained": True,
        "failure": False,
        "episode_length": prefix_steps,
        "policy_queries": int(math.ceil(prefix_steps / max(replan_steps, 1))),
    }


def summarize_outcomes(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(outcomes)
    attempted = [item for item in outcomes if not bool(item["abstained"])]
    attempted_count = len(attempted)
    success_count = sum(int(bool(item["success"])) for item in outcomes)
    failure_count = sum(int(bool(item["failure"])) for item in outcomes)
    timeout_count = sum(int(bool(item["timeout"])) for item in outcomes)
    abstained_count = sum(int(bool(item["abstained"])) for item in outcomes)
    utility_values = [
        expected_utility(
            success=bool(item["success"]),
            timeout=bool(item["timeout"]),
            abstained=bool(item["abstained"]),
            episode_length=int(item["episode_length"]),
        )
        for item in outcomes
    ]
    payload = {
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
        "ci95": {
            "coverage": bootstrap_ci(outcomes, coverage_stat, samples=500),
            "task_completion_rate": bootstrap_ci(outcomes, task_completion_stat, samples=500),
            "attempted_completion_rate": bootstrap_ci(outcomes, attempted_completion_stat, samples=500),
            "failure_rate_attempted": bootstrap_ci(outcomes, failure_attempted_stat, samples=500),
            "timeout_rate": bootstrap_ci(outcomes, timeout_stat, samples=500),
            "abstention_rate": bootstrap_ci(outcomes, abstention_stat, samples=500),
            "expected_utility": bootstrap_ci(utility_values, mean_float, samples=500),
        },
    }
    return payload


def add_comparison_fields(metrics: dict[str, Any], direct_test: Mapping[str, Any], direct_queries: float) -> None:
    direct_failure = float(direct_test["failure_rate_attempted"])
    metrics["failure_attempted_reduction_vs_direct"] = direct_failure - float(metrics["failure_rate_attempted"])
    metrics["utility_delta_vs_direct"] = float(metrics["expected_utility"]) - float(direct_test["expected_utility"])
    metrics["policy_query_overhead_vs_direct"] = float(metrics["mean_policy_queries"]) / direct_queries


def best_operating_points(rows: Sequence[Mapping[str, Any]], direct_test: Mapping[str, Any]) -> dict[str, Any]:
    test_rows = [flatten_row(row) for row in rows]
    best_utility = max(test_rows, key=lambda row: float(row["expected_utility"]))
    safest = min(test_rows, key=lambda row: float(row["failure_rate_attempted"]))
    most_productive = max(test_rows, key=lambda row: (float(row["task_completion_rate"]), float(row["coverage"])))
    return {
        "direct_openpi_test_utility": direct_test["expected_utility"],
        "direct_openpi_test_failure_rate_attempted": direct_test["failure_rate_attempted"],
        "best_utility": best_utility,
        "safest_mode": safest,
        "most_productive_mode": most_productive,
        "best_failure_reduction_at_or_above_85_coverage": best_failure_reduction(test_rows, 0.85),
        "best_failure_reduction_at_or_above_90_coverage": best_failure_reduction(test_rows, 0.90),
        "best_failure_reduction_at_or_above_95_coverage": best_failure_reduction(test_rows, 0.95),
        "beats_direct_openpi_utility": float(best_utility["expected_utility"]) > float(direct_test["expected_utility"]),
    }


def flatten_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        "target_coverage": row["target_coverage"],
        "threshold": row["threshold"],
    }
    output.update(row["test"])
    return output


def best_failure_reduction(rows: Sequence[Mapping[str, Any]], min_coverage: float) -> dict[str, Any] | None:
    candidates = [row for row in rows if float(row["coverage"]) >= min_coverage]
    return min(candidates, key=lambda row: float(row["failure_rate_attempted"])) if candidates else None


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


def expand_inputs(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(Path(match) for match in matches)
    return sorted(dict.fromkeys(paths))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
