#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from risk_aware_skill_planning.evaluation.openpi_metrics import binary_risk_metrics, reliability_bins
from risk_aware_skill_planning.risk.openpi_dataset import (
    class_balance,
    load_openpi_risk_examples,
    split_examples,
)
from risk_aware_skill_planning.risk.openpi_models import choose_threshold


AUDIT_START = "<!-- OPENPI_METRICS_AUDIT_START -->"
AUDIT_END = "<!-- OPENPI_METRICS_AUDIT_END -->"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit OpenPI/LIBERO risk metrics from raw JSONL")
    parser.add_argument("--risk-summary", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--raw-glob", default="datasets/openpi_libero_rollouts/openpi_rollouts_*.jsonl")
    parser.add_argument("--audit-json", default="reports/openpi_metrics_audit.json")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    summary = json.loads(Path(args.risk_summary).read_text(encoding="utf-8"))
    audit = audit_summary(summary, raw_glob=args.raw_glob, tolerance=args.tolerance)
    audit_path = Path(args.audit_json)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    upsert_report_audit(Path(args.report), audit, audit_path)
    print(json.dumps({"ok": audit["ok"], "audit_json": str(audit_path), "failures": audit["failures"]}, indent=2))
    return 0 if audit["ok"] else 2


def audit_summary(summary: Mapping[str, Any], *, raw_glob: str, tolerance: float = 1e-9) -> dict[str, Any]:
    failures: list[str] = []
    raw_paths = sorted(Path().glob(raw_glob))
    raw_episodes = [episode for path in raw_paths for episode in read_jsonl(path)]
    raw_counts = summarize_raw_episodes(raw_episodes)

    input_paths = [Path(path) for path in summary.get("input_paths", [])]
    training_episodes = [episode for path in input_paths for episode in read_jsonl(path)]
    supervisor_training_modes = sorted({episode.get("mode") for episode in training_episodes if episode.get("mode") != "direct_openpi"})
    if supervisor_training_modes:
        failures.append(f"Supervisor/non-direct modes present in training inputs: {supervisor_training_modes}")

    examples = load_openpi_risk_examples(input_paths, prefix_steps=int(summary.get("prefix_steps", 10)))
    splits = split_examples(examples)
    split_keys = {
        name: {(example.run_id, example.episode_id) for example in items}
        for name, items in splits.items()
    }
    overlap = {
        "train_calibration": sorted(split_keys["train"].intersection(split_keys["calibration"])),
        "train_test": sorted(split_keys["train"].intersection(split_keys["test"])),
        "calibration_test": sorted(split_keys["calibration"].intersection(split_keys["test"])),
    }
    if any(overlap.values()):
        failures.append(f"Train/calibration/test split overlap found: {overlap}")

    recomputed_balance = {name: class_balance(items) for name, items in splits.items()} | {"all": class_balance(examples)}
    compare_mapping("class_balance", recomputed_balance, summary.get("class_balance", {}), failures, tolerance)

    threshold = float(summary["calibration"]["threshold"])
    probs_by_split = {name: predict_from_summary(summary, items) for name, items in splits.items()}
    labels_by_split = {name: [example.label_failure for example in items] for name, items in splits.items()}
    calibration_threshold = choose_threshold(labels_by_split["calibration"], probs_by_split["calibration"])
    if not close(calibration_threshold, threshold, tolerance):
        failures.append(
            f"Threshold mismatch: recomputed from calibration={calibration_threshold}, summary={threshold}"
        )

    metrics = {}
    baseline_metrics = {"global_prior": {}, "fixed_task_prior": {}}
    global_prior = sum(labels_by_split["train"]) / len(labels_by_split["train"]) if labels_by_split["train"] else 0.0
    task_priors = task_priors_from_examples(splits["train"], default=global_prior)
    for split_name, items in splits.items():
        labels = labels_by_split[split_name]
        probs = probs_by_split[split_name]
        metrics[split_name] = binary_risk_metrics(labels, probs, threshold=threshold)
        metrics[split_name]["reliability_bins"] = reliability_bins(labels, probs)
        baseline_metrics["global_prior"][split_name] = binary_risk_metrics(
            labels,
            [global_prior for _ in items],
            threshold=threshold,
        )
        baseline_metrics["fixed_task_prior"][split_name] = binary_risk_metrics(
            labels,
            [task_priors.get((example.suite, example.task_id), global_prior) for example in items],
            threshold=threshold,
        )

    compare_mapping("metrics", metrics, summary.get("metrics", {}), failures, tolerance)
    compare_mapping("baselines", baseline_metrics, summary.get("baselines", {}), failures, tolerance)

    variant_checks: dict[str, Any] = {}
    for variant_name, variant in summary.get("model_variants", {}).items():
        if not isinstance(variant, Mapping) or not variant.get("ok"):
            variant_checks[variant_name] = {
                "ok": False,
                "status": variant.get("status", "blocked") if isinstance(variant, Mapping) else "invalid",
                "blocker": variant.get("blocker") if isinstance(variant, Mapping) else "Variant entry is not a mapping",
            }
            continue
        variant_probs_by_split = {name: predict_from_summary(variant, items) for name, items in splits.items()}
        variant_threshold = float(variant["calibration"]["threshold"])
        variant_calibration_threshold = choose_threshold(
            labels_by_split["calibration"],
            variant_probs_by_split["calibration"],
        )
        if not close(variant_calibration_threshold, variant_threshold, tolerance):
            failures.append(
                f"{variant_name} threshold mismatch: "
                f"recomputed from calibration={variant_calibration_threshold}, summary={variant_threshold}"
            )
        variant_metrics = {}
        variant_baselines = {"global_prior": {}, "fixed_task_prior": {}}
        for split_name, items in splits.items():
            labels = labels_by_split[split_name]
            probs = variant_probs_by_split[split_name]
            variant_metrics[split_name] = binary_risk_metrics(labels, probs, threshold=variant_threshold)
            variant_metrics[split_name]["reliability_bins"] = reliability_bins(labels, probs)
            variant_baselines["global_prior"][split_name] = binary_risk_metrics(
                labels,
                [global_prior for _ in items],
                threshold=variant_threshold,
            )
            variant_baselines["fixed_task_prior"][split_name] = binary_risk_metrics(
                labels,
                [task_priors.get((example.suite, example.task_id), global_prior) for example in items],
                threshold=variant_threshold,
            )
        compare_mapping(f"model_variants.{variant_name}.metrics", variant_metrics, variant.get("metrics", {}), failures, tolerance)
        compare_mapping(
            f"model_variants.{variant_name}.baselines",
            variant_baselines,
            variant.get("baselines", {}),
            failures,
            tolerance,
        )
        variant_checks[variant_name] = {
            "ok": True,
            "threshold": variant_threshold,
            "calibration_threshold_recomputed": variant_calibration_threshold,
            "test_metrics": variant_metrics.get("test", {}),
        }

    global_test = baseline_metrics["global_prior"]["test"]
    global_probs = [global_prior for _ in splits["test"]]
    if len(set(global_probs)) <= 1 and global_test["auprc"] is not None:
        expected_tied_auprc = global_test["positive_rate"]
        if not close(float(global_test["auprc"]), float(expected_tied_auprc), tolerance):
            failures.append(
                "Global-prior AUPRC tie handling is suspect: "
                f"auprc={global_test['auprc']}, positive_rate={expected_tied_auprc}"
            )

    return {
        "ok": not failures,
        "failures": failures,
        "raw_episode_counts": raw_counts,
        "training_input_paths": [str(path) for path in input_paths],
        "training_modes": sorted({str(episode.get("mode")) for episode in training_episodes}),
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "split_overlap": overlap,
        "calibration_threshold_recomputed": calibration_threshold,
        "summary_threshold": threshold,
        "global_prior_test_auprc": global_test["auprc"],
        "global_prior_test_positive_rate": global_test["positive_rate"],
        "metric_checks": {
            "class_balance": recomputed_balance,
            "test_metrics": metrics.get("test", {}),
            "test_baselines": {name: values.get("test", {}) for name, values in baseline_metrics.items()},
            "variants": variant_checks,
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_raw_episodes(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(episodes)
    by_mode: dict[str, int] = {}
    by_suite: dict[str, int] = {}
    by_stressor: dict[str, int] = {}
    by_stressor_severity: dict[str, int] = {}
    success = 0
    timeout = 0
    abstained = 0
    for episode in items:
        mode = str(episode.get("mode", "unknown"))
        suite = str(episode.get("libero_suite", episode.get("suite", "unknown")))
        stressor = str(episode.get("stressor_name", "unknown"))
        stressor_params = episode.get("stressor_params", {})
        severity = float(stressor_params.get("severity", episode.get("metadata", {}).get("stressor_severity", 0.0)))
        by_mode[mode] = by_mode.get(mode, 0) + 1
        by_suite[suite] = by_suite.get(suite, 0) + 1
        by_stressor[stressor] = by_stressor.get(stressor, 0) + 1
        severity_key = f"{stressor}:{severity:.2f}"
        by_stressor_severity[severity_key] = by_stressor_severity.get(severity_key, 0) + 1
        success += int(bool(episode.get("success", False)))
        timeout += int(bool(episode.get("timeout", False)))
        abstained += int(str(episode.get("failure_label", "")) == "abstained")
    return {
        "episodes": len(items),
        "successes": success,
        "timeouts": timeout,
        "abstained": abstained,
        "by_mode": by_mode,
        "by_suite": by_suite,
        "by_stressor": by_stressor,
        "by_stressor_severity": by_stressor_severity,
    }


def predict_from_summary(summary: Mapping[str, Any], examples) -> list[float]:
    names = list(summary["feature_names"])
    weights = [float(summary["weights"][name]) for name in names]
    mean = [float(summary["normalization"]["mean"][name]) for name in names]
    std = [max(float(summary["normalization"]["std"][name]), 1e-6) for name in names]
    temperature = max(float(summary["calibration"]["temperature"]), 1e-6)
    probs = []
    for example in examples:
        feature_lookup = dict(zip(example.feature_names, example.features))
        logit = 0.0
        for name, weight, mu, sigma in zip(names, weights, mean, std):
            value = feature_lookup[name]
            logit += weight * ((float(value) - mu) / sigma)
        probs.append(sigmoid(logit / temperature))
    return probs


def task_priors_from_examples(examples, *, default: float) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[int]] = {}
    for example in examples:
        grouped.setdefault((example.suite, example.task_id), []).append(example.label_failure)
    return {key: sum(values) / len(values) for key, values in grouped.items()} or {("", -1): default}


def compare_mapping(
    name: str,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    failures: list[str],
    tolerance: float,
    path: str = "",
) -> None:
    for key, actual_value in actual.items():
        child_path = f"{path}.{key}" if path else str(key)
        if key not in expected:
            failures.append(f"{name} missing expected key {child_path}")
            continue
        expected_value = expected[key]
        if isinstance(actual_value, Mapping) and isinstance(expected_value, Mapping):
            compare_mapping(name, actual_value, expected_value, failures, tolerance, child_path)
        elif isinstance(actual_value, list) and isinstance(expected_value, list):
            if len(actual_value) != len(expected_value):
                failures.append(f"{name}.{child_path} length mismatch: {len(actual_value)} != {len(expected_value)}")
            else:
                for index, (actual_item, expected_item) in enumerate(zip(actual_value, expected_value)):
                    compare_mapping(
                        name,
                        {str(index): actual_item},
                        {str(index): expected_item},
                        failures,
                        tolerance,
                        child_path,
                    )
        elif not values_match(actual_value, expected_value, tolerance):
            failures.append(f"{name}.{child_path} mismatch: actual={actual_value} expected={expected_value}")


def values_match(actual: Any, expected: Any, tolerance: float) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return close(float(actual), float(expected), tolerance)
    return actual == expected


def close(left: float, right: float, tolerance: float) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def upsert_report_audit(report_path: Path, audit: Mapping[str, Any], audit_path: Path) -> None:
    status = "PASS" if audit["ok"] else "FAIL"
    section = "\n".join(
        [
            AUDIT_START,
            "## Metrics Audit",
            "",
            f"Status: `{status}`",
            f"Audit JSON: `{audit_path}`",
            "",
            "```json",
            json.dumps(
                {
                    "ok": audit["ok"],
                    "failures": audit["failures"],
                    "raw_episode_counts": audit["raw_episode_counts"],
                    "split_sizes": audit["split_sizes"],
                    "summary_threshold": audit["summary_threshold"],
                    "calibration_threshold_recomputed": audit["calibration_threshold_recomputed"],
                    "global_prior_test_auprc": audit["global_prior_test_auprc"],
                    "global_prior_test_positive_rate": audit["global_prior_test_positive_rate"],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            AUDIT_END,
            "",
        ]
    )
    original = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if AUDIT_START in original and AUDIT_END in original:
        before = original.split(AUDIT_START, 1)[0].rstrip()
        after = original.split(AUDIT_END, 1)[1].lstrip()
        text = f"{before}\n\n{section}{after}"
    else:
        text = f"{original.rstrip()}\n\n{section}"
    report_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
