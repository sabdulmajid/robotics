#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_openpi_controlled_deployment import (  # noqa: E402
    episode_key,
    episode_outcome,
    oracle_abstain_upper_bound,
    random_abstain_matched_coverage,
    read_jsonl,
    summarize_outcomes,
    synthetic_abstain_outcome,
)
from risk_aware_skill_planning.evaluation.openpi_metrics import binary_risk_metrics  # noqa: E402


DIRECT_LABEL = "direct"
SCORE_LABEL = "siglip_score"


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a cross-suite SigLIP threshold on calibration tasks")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="reports/openpi_cross_suite_calibration_summary.json")
    parser.add_argument("--target-coverages", default="0.70,0.75,0.80,0.85,0.90,0.95,1.00")
    parser.add_argument("--random-seeds", type=int, default=1000)
    args = parser.parse_args()

    manifest = read_jsonl(Path(args.manifest))
    episodes = load_manifest_episodes(manifest)
    targets = [float(value) for value in args.target_coverages.split(",") if value.strip()]
    summary = calibrate_threshold(manifest, episodes, targets, random_seeds=args.random_seeds)
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


def calibrate_threshold(
    manifest: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    target_coverages: Sequence[float],
    *,
    random_seeds: int,
) -> dict[str, Any]:
    by_label = group_by_label(episodes)
    grid = validate_grid(by_label)
    pairs = pair_examples(by_label)
    failures = list(grid["failures"])
    if not pairs:
        failures.append("No paired direct and SigLIP score episodes were found")
    if failures:
        return {
            "ok": False,
            "failures": failures,
            "manifest": list(manifest),
            "grid_checks": grid,
            "paired_episodes": len(pairs),
        }

    direct_episodes = [pair["direct"] for pair in pairs]
    risks = [float(pair["risk"]) for pair in pairs]
    labels = [int(bool(episode_outcome(pair["direct"])["failure"])) for pair in pairs]
    rows = []
    for target in target_coverages:
        threshold = threshold_for_target_coverage(risks, float(target))
        metrics = evaluate_threshold(pairs, threshold)
        random_baseline = random_abstain_matched_coverage(
            direct_episodes,
            target_coverage=float(metrics["coverage"]),
            samples=random_seeds,
        )
        rows.append(
            {
                "target_coverage": float(target),
                "threshold": threshold,
                "metrics": metrics,
                "random_abstain_matched_coverage": random_baseline,
                "oracle_abstain_upper_bound": oracle_abstain_upper_bound(
                    direct_episodes,
                    target_coverage=float(metrics["coverage"]),
                ),
            }
        )

    selected = max(
        rows,
        key=lambda row: (
            float(row["metrics"]["expected_utility"]),
            float(row["metrics"]["coverage"]),
        ),
    )
    threshold = float(selected["threshold"])
    return {
        "ok": True,
        "failures": [],
        "experiment": "cross_suite_threshold_calibration",
        "selection_rule": "maximum calibration utility; use higher coverage to break a tie",
        "data_boundary": "The selection uses calibration tasks only. It does not use held-out deployment outcomes.",
        "manifest": list(manifest),
        "job_ids": [int(row["job_id"]) for row in manifest],
        "grid_checks": grid,
        "paired_episodes": len(pairs),
        "risk_metrics": binary_risk_metrics(labels, risks, threshold=threshold),
        "direct_metrics": summarize_outcomes([episode_outcome(episode) for episode in direct_episodes]),
        "target_coverages": [float(value) for value in target_coverages],
        "threshold_rows": rows,
        "selection": {
            "target_coverage": float(selected["target_coverage"]),
            "threshold": threshold,
            "calibration_metrics": selected["metrics"],
            "random_abstain_matched_coverage": selected["random_abstain_matched_coverage"],
            "oracle_abstain_upper_bound": selected["oracle_abstain_upper_bound"],
        },
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
    output: dict[str, list[Mapping[str, Any]]] = {}
    for episode in episodes:
        output.setdefault(str(episode["_manifest"]["label"]), []).append(episode)
    return output


def validate_grid(by_label: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    failures = []
    keys = {label: {episode_key(episode) for episode in episodes} for label, episodes in by_label.items()}
    for label in (DIRECT_LABEL, SCORE_LABEL):
        if not keys.get(label):
            failures.append(f"No episodes were found for label {label}")
    if keys.get(DIRECT_LABEL) and keys.get(SCORE_LABEL):
        missing = keys[DIRECT_LABEL] - keys[SCORE_LABEL]
        extra = keys[SCORE_LABEL] - keys[DIRECT_LABEL]
        if missing:
            failures.append(f"SigLIP score data is missing {len(missing)} direct episode keys")
        if extra:
            failures.append(f"SigLIP score data has {len(extra)} extra episode keys")
    return {
        "ok": not failures,
        "failures": failures,
        "episode_counts": {label: len(items) for label, items in sorted(by_label.items())},
        "unique_key_counts": {label: len(items) for label, items in sorted(keys.items())},
    }


def pair_examples(by_label: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    direct = {episode_key(episode): episode for episode in by_label.get(DIRECT_LABEL, [])}
    scores = {episode_key(episode): episode for episode in by_label.get(SCORE_LABEL, [])}
    output = []
    for key in sorted(set(direct) & set(scores)):
        risk = episode_risk(scores[key])
        if risk is not None:
            output.append({"key": key, "direct": direct[key], "score_episode": scores[key], "risk": risk})
    return output


def episode_risk(episode: Mapping[str, Any]) -> float | None:
    decision = episode.get("runtime_supervisor_decision")
    if not isinstance(decision, Mapping):
        metadata = episode.get("metadata", {})
        decision = metadata.get("runtime_supervisor_decision") if isinstance(metadata, Mapping) else None
    if not isinstance(decision, Mapping) or decision.get("predicted_risk") is None:
        return None
    return float(decision["predicted_risk"])


def threshold_for_target_coverage(risks: Sequence[float], target: float) -> float:
    ordered = sorted(float(value) for value in risks)
    if not ordered:
        return 1.0
    accepted = max(1, min(len(ordered), int(math.ceil(target * len(ordered)))))
    value = ordered[accepted - 1]
    return float(value + max(1e-12, abs(value) * 1e-12))


def evaluate_threshold(pairs: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    outcomes = []
    for pair in pairs:
        episode = pair["direct"]
        outcomes.append(
            episode_outcome(episode)
            if float(pair["risk"]) < threshold
            else synthetic_abstain_outcome(episode)
        )
    return summarize_outcomes(outcomes)


if __name__ == "__main__":
    raise SystemExit(main())
