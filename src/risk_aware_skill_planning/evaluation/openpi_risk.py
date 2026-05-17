from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from risk_aware_skill_planning.evaluation.openpi_metrics import binary_risk_metrics, reliability_bins
from risk_aware_skill_planning.risk.openpi_dataset import (
    class_balance,
    expand_input_paths,
    load_openpi_risk_examples,
    split_examples,
)
from risk_aware_skill_planning.risk.openpi_models import (
    calibrate_temperature,
    choose_threshold,
    predict_examples,
    train_logistic_risk_model,
)


def run_openpi_risk_training(
    input_patterns: Sequence[str],
    *,
    prefix_steps: int = 10,
) -> dict[str, Any]:
    paths = expand_input_paths(input_patterns)
    examples = load_openpi_risk_examples(paths, prefix_steps=prefix_steps)
    splits = split_examples(examples)
    summary: dict[str, Any] = {
        "input_paths": [str(path) for path in paths],
        "prefix_steps": prefix_steps,
        "class_balance": {name: class_balance(items) for name, items in splits.items()} | {"all": class_balance(examples)},
        "feature_names": list(examples[0].feature_names) if examples else [],
    }
    if len(examples) < 3 or len({example.label_failure for example in examples}) < 2:
        summary["ok"] = False
        summary["blocker"] = "Need at least three examples and both success/failure labels to train a risk critic"
        return summary
    model = train_logistic_risk_model(splits["train"])
    calibrated = calibrate_temperature(model, splits["calibration"])
    calibration_probs = predict_examples(calibrated, splits["calibration"])
    calibration_labels = [example.label_failure for example in splits["calibration"]]
    threshold = choose_threshold(calibration_labels, calibration_probs)
    split_metrics = {}
    baseline_metrics = {"global_prior": {}, "fixed_task_prior": {}}
    global_prior = _mean_label(splits["train"])
    task_priors = _task_priors(splits["train"], default=global_prior)
    for split_name, items in splits.items():
        probs = predict_examples(calibrated, items)
        labels = [example.label_failure for example in items]
        split_metrics[split_name] = binary_risk_metrics(labels, probs, threshold=threshold)
        split_metrics[split_name]["reliability_bins"] = reliability_bins(labels, probs)
        baseline_metrics["global_prior"][split_name] = binary_risk_metrics(
            labels,
            [global_prior for _ in items],
            threshold=threshold,
        )
        baseline_metrics["fixed_task_prior"][split_name] = binary_risk_metrics(
            labels,
            [_lookup_task_prior(task_priors, example, global_prior) for example in items],
            threshold=threshold,
        )
    summary.update(
        {
            "ok": True,
            "model": "logistic_state_action_stress_risk",
            "calibration": {
                "method": "temperature_scaling_grid",
                "temperature": calibrated.temperature,
                "threshold": threshold,
            },
            "weights": dict(zip(calibrated.feature_names, calibrated.weights)),
            "metrics": split_metrics,
            "baselines": baseline_metrics,
            "supervisor_offline": {
                "mode": "selective_openpi",
                "threshold_source": "calibration_split",
                "threshold": threshold,
                "interpretation": "Episodes with calibrated p_failure >= threshold are abstained in this offline coverage analysis.",
                "test_coverage": split_metrics["test"]["coverage_at_threshold"],
                "test_failure_rate_attempted": split_metrics["test"]["failure_rate_attempted"],
            },
        }
    )
    return summary


def write_openpi_risk_outputs(summary: dict[str, Any], summary_path: str | Path, report_path: str | Path) -> None:
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# OpenPI/LIBERO Risk Training",
        "",
        f"Status: `{'PASS' if summary.get('ok') else 'BLOCKED'}`",
        "",
        "## Dataset",
        "",
        "```json",
        json.dumps(summary.get("class_balance", {}), indent=2, sort_keys=True),
        "```",
        "",
    ]
    if not summary.get("ok"):
        lines.extend(["## Blocker", "", str(summary.get("blocker")), ""])
    else:
        lines.extend(
            [
                "## Calibration",
                "",
                "```json",
                json.dumps(summary["calibration"], indent=2, sort_keys=True),
                "```",
                "",
                "## Metrics",
                "",
                "```json",
                json.dumps(summary["metrics"], indent=2, sort_keys=True),
                "```",
                "",
                "## Baselines",
                "",
                "```json",
                json.dumps(summary["baselines"], indent=2, sort_keys=True),
                "```",
                "",
                "## Offline Supervisor",
                "",
                "```json",
                json.dumps(summary["supervisor_offline"], indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def _mean_label(examples) -> float:
    return sum(example.label_failure for example in examples) / len(examples) if examples else 0.0


def _task_priors(examples, *, default: float) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[int]] = {}
    for example in examples:
        grouped.setdefault((example.suite, example.task_id), []).append(example.label_failure)
    return {key: sum(values) / len(values) for key, values in grouped.items()} or {("", -1): default}


def _lookup_task_prior(priors: dict[tuple[str, int], float], example, default: float) -> float:
    return priors.get((example.suite, example.task_id), default)
