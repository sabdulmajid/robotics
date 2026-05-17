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
    if len({example.label_failure for example in splits["train"]}) < 2:
        summary["ok"] = False
        summary["blocker"] = "Training split needs both success and failure examples; collect more failure-rich stress rollouts"
        return summary
    calibration_items = splits["calibration"]
    if len({example.label_failure for example in calibration_items}) < 2:
        calibration_items = splits["train"]
        summary["calibration_warning"] = "Calibration split lacked both classes; reused train split for preliminary calibration"
    model = train_logistic_risk_model(splits["train"])
    calibrated = calibrate_temperature(model, calibration_items)
    calibration_probs = predict_examples(calibrated, calibration_items)
    calibration_labels = [example.label_failure for example in calibration_items]
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
            "normalization": {
                "mean": dict(zip(calibrated.feature_names, calibrated.mean)),
                "std": dict(zip(calibrated.feature_names, calibrated.std)),
            },
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
    test_metrics = summary.get("metrics", {}).get("test", {})
    fixed_test = summary.get("baselines", {}).get("fixed_task_prior", {}).get("test", {})
    global_test = summary.get("baselines", {}).get("global_prior", {}).get("test", {})
    calibration = summary.get("calibration", {})
    supervisor = summary.get("supervisor_offline", {})
    lines = [
        "# OpenPI/LIBERO Risk Training",
        "",
        f"Status: `{'PASS' if summary.get('ok') else 'BLOCKED'}`",
        "",
        "This report is the current robot-foundation-policy checkpoint for the project. OpenPI `pi05_libero` is used as the vision-language-action policy, LIBERO supplies the manipulation tasks, and this layer learns a rollout-level failure-risk model used for selective execution and adaptive action chunking.",
        "",
        "## Dataset",
        "",
        "The risk dataset is built from direct OpenPI/LIBERO rollouts. Each row is one episode converted into initial/task/stressor features plus early rollout progress statistics; labels mark any terminal failure or timeout.",
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
                f"Temperature scaling selected `T={calibration.get('temperature')}` and planner threshold `{calibration.get('threshold')}` on the calibration split.",
                "",
                "```json",
                json.dumps(calibration, indent=2, sort_keys=True),
                "```",
                "",
                "## Test Metrics",
                "",
                "| Model | AUROC | AUPRC | Brier | NLL | ECE | Coverage @ threshold | Failure rate attempted |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                _metric_row("global prior", global_test),
                _metric_row("fixed task prior", fixed_test),
                _metric_row("logistic state/progress risk", test_metrics),
                "",
                "The test split is still small, so these numbers should be treated as an engineering checkpoint rather than a final benchmark claim. The fixed task prior is a strong baseline because the current stress suite makes task identity informative; the next step is to add richer image/language and transition features so the learned critic can beat fixed priors under held-out perturbations.",
                "",
                "## Offline Supervisor",
                "",
                "```json",
                json.dumps(supervisor, indent=2, sort_keys=True),
                "```",
                "",
                "## Reproduce",
                "",
                "```bash",
                "SUITES=\"libero_spatial\" TASK_IDS=\"0 1 2\" NUM_TRIALS=3 STRESSORS=\"none\" sbatch slurm/openpi_libero_rollouts.sbatch",
                "SUITES=\"libero_spatial\" TASK_IDS=\"0 1 2\" NUM_TRIALS=3 STRESSORS=\"occlusion\" STRESSOR_SEVERITY=1.0 sbatch slurm/openpi_libero_rollouts.sbatch",
                "PYTHONPATH=src python scripts/train_openpi_risk.py --config configs/openpi/train_risk.yaml",
                "MODE=adaptive_chunk_openpi RISK_SUMMARY=reports/openpi_libero_risk_summary.json SUITES=\"libero_spatial\" TASK_IDS=\"0 1 2\" NUM_TRIALS=2 STRESSORS=\"occlusion\" STRESSOR_SEVERITY=1.0 sbatch slurm/openpi_libero_rollouts.sbatch",
                "```",
                "",
                "## Limitations",
                "",
                "- The current OpenPI result is a small rollout study, not a benchmark-scale LIBERO evaluation.",
                "- The present risk features include stressor metadata for controlled stress testing; the production path should replace this with directly observed image/language/progress features.",
                "- The learned risk critic is a transparent logistic baseline. Neural VLM/world-model features are planned after this executable risk-supervision loop is stable.",
                "",
            ]
        )
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def _metric_row(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {_fmt(metrics.get('auroc'))} | {_fmt(metrics.get('auprc'))} | "
        f"{_fmt(metrics.get('brier'))} | {_fmt(metrics.get('nll'))} | {_fmt(metrics.get('ece'))} | "
        f"{_fmt(metrics.get('coverage_at_threshold'))} | {_fmt(metrics.get('failure_rate_attempted'))} |"
    )


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _mean_label(examples) -> float:
    return sum(example.label_failure for example in examples) / len(examples) if examples else 0.0


def _task_priors(examples, *, default: float) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[int]] = {}
    for example in examples:
        grouped.setdefault((example.suite, example.task_id), []).append(example.label_failure)
    return {key: sum(values) / len(values) for key, values in grouped.items()} or {("", -1): default}


def _lookup_task_prior(priors: dict[tuple[str, int], float], example, default: float) -> float:
    return priors.get((example.suite, example.task_id), default)
