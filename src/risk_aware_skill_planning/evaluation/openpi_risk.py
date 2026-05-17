from __future__ import annotations

import json
import html
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from risk_aware_skill_planning.evaluation.bootstrap import bootstrap_ci
from risk_aware_skill_planning.evaluation.openpi_metrics import binary_risk_metrics, reliability_bins
from risk_aware_skill_planning.evaluation.risk_coverage import expected_utility
from risk_aware_skill_planning.risk.openpi_dataset import (
    OpenPIRiskExample,
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


STRESSOR_FEATURES = {
    "stressor_severity",
    "stressor_none",
    "stressor_occlusion",
    "stressor_action_noise",
    "stressor_gaussian_noise",
    "stressor_brightness",
    "stressor_action_delay",
    "stressor_action_precision",
}

VLM_FEATURES = {
    "vlm_image_embedding",
    "vision_language_embedding",
    "openclip_embedding",
}

COVERAGE_TARGETS = (0.25, 0.5, 0.75, 0.9, 1.0)


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
        "dataset_summary": _dataset_summary(examples),
    }
    if len(examples) < 3 or len({example.label_failure for example in examples}) < 2:
        summary["ok"] = False
        summary["blocker"] = "Need at least three examples and both success/failure labels to train a risk critic"
        return summary
    if len({example.label_failure for example in splits["train"]}) < 2:
        summary["ok"] = False
        summary["blocker"] = "Training split needs both success and failure examples; collect more failure-rich stress rollouts"
        return summary
    all_feature_names = tuple(examples[0].feature_names)
    structured_feature_names = tuple(name for name in all_feature_names if name not in STRESSOR_FEATURES)
    model_variants = {
        "metadata_oracle_risk": _fit_openpi_variant(
            examples,
            splits,
            feature_names=all_feature_names,
            description=(
                "Diagnostic upper-bound model that is allowed to see controlled stressor metadata. "
                "It should not be treated as deployable risk perception."
            ),
            uses_stressor_metadata=True,
        ),
        "structured_progress_risk": _fit_openpi_variant(
            examples,
            splits,
            feature_names=structured_feature_names,
            description=(
                "Deployable structured baseline using task/language hashes, action horizon, and early "
                "rollout progress statistics, with hidden stressor metadata removed."
            ),
            uses_stressor_metadata=False,
        ),
        "vision_language_risk": _vision_language_variant_status(examples),
    }
    primary_name = "structured_progress_risk"
    primary = model_variants[primary_name]
    if not primary.get("ok"):
        primary_name = "metadata_oracle_risk"
        primary = model_variants[primary_name]
    if not primary.get("ok"):
        summary["ok"] = False
        summary["blocker"] = str(primary.get("blocker", "No trainable OpenPI risk variant"))
        summary["model_variants"] = model_variants
        return summary

    summary.update(_primary_summary_fields(primary))
    summary.update(
        {
            "ok": True,
            "model": primary_name,
            "primary_model": primary_name,
            "model_variants": model_variants,
            "offline_policy_comparison": _offline_policy_comparison(model_variants, primary),
            "supervisor_offline": {
                "mode": "selective_openpi",
                "threshold_source": "calibration_split",
                "threshold": primary["calibration"]["threshold"],
                "interpretation": "Episodes with calibrated p_failure >= threshold are abstained in this offline coverage analysis.",
                "test_coverage": primary["metrics"]["test"]["coverage_at_threshold"],
                "test_failure_rate_attempted": primary["metrics"]["test"]["failure_rate_attempted"],
                "coverage_curve": primary["coverage_curves"]["test"],
            },
            "open_source_stack": {
                "openpi": "Primary robot foundation/VLA policy, using pi05_libero.",
                "libero": "Manipulation benchmark and task suite.",
                "lerobot": "Planned export/baseline path; not used for the current risk numbers.",
                "vlm_features": "Planned frozen image/language embedding ablation; current rollouts did not save embeddings.",
                "world_model_features": "Current structured model uses early transition/progress statistics as a lightweight world-model proxy; learned predictive dynamics are a planned ablation.",
            },
        }
    )
    return summary


def write_openpi_risk_outputs(summary: dict[str, Any], summary_path: str | Path, report_path: str | Path) -> None:
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if summary.get("ok"):
        _write_openpi_figures(summary)
    test_metrics = summary.get("metrics", {}).get("test", {})
    fixed_test = summary.get("baselines", {}).get("fixed_task_prior", {}).get("test", {})
    global_test = summary.get("baselines", {}).get("global_prior", {}).get("test", {})
    calibration = summary.get("calibration", {})
    supervisor = summary.get("supervisor_offline", {})
    variants = summary.get("model_variants", {})
    dataset_summary = summary.get("dataset_summary", {})
    lines = [
        "# OpenPI/LIBERO Risk Training",
        "",
        f"Status: `{'PASS' if summary.get('ok') else 'BLOCKED'}`",
        "",
        "This report is the current robot-foundation-policy checkpoint for the project. OpenPI `pi05_libero` is used as the vision-language-action policy, LIBERO supplies the manipulation tasks, and this layer learns rollout-level failure-risk models for selective execution and adaptive action chunking.",
        "",
        "## Dataset",
        "",
        "The risk dataset is built from direct OpenPI/LIBERO rollouts. Each row is one episode converted into initial/task/stressor features plus early rollout progress statistics; labels mark any terminal failure or timeout.",
        "",
        "```json",
        json.dumps(summary.get("class_balance", {}), indent=2, sort_keys=True),
        "```",
        "",
        "Dataset coverage:",
        "",
        "```json",
        json.dumps(dataset_summary, indent=2, sort_keys=True),
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
                _metric_row(str(summary.get("primary_model", "primary risk")), test_metrics),
                "",
                "Model ablations:",
                "",
                _variant_table(variants),
                "",
                "Offline policy comparison at matched coverage:",
                "",
                _offline_policy_table(summary.get("offline_policy_comparison", {})),
                "",
                "Bootstrap confidence intervals for selected supervisor metrics are stored in the risk summary. Compact view:",
                "",
                _offline_policy_ci_table(summary.get("offline_policy_comparison", {})),
                "",
                "![OpenPI risk reliability](figures/openpi_risk_reliability.svg)",
                "",
                "![OpenPI coverage vs failure](figures/openpi_coverage_failure.svg)",
                "",
                "The metadata-aware model is diagnostic because it can observe the injected stressor. The structured/progress model is the deployable baseline in this report because it excludes hidden stressor metadata. VLM embedding and learned world-model ablations are tracked explicitly but are not counted until RGB embeddings or predictive dynamics features are generated.",
                "",
                "VLM feasibility note: the completed rollout JSONL files do not contain saved frame paths, the current Python environment has no cached SigLIP/DINOv2 checkpoint, and `transformers` could not load `google/siglip-base-patch16-224` from cache or Hugging Face during this run. The evaluator and SLURM wrapper now support `SAVE_IMAGES=1`, so the next step is an image-logging subset plus cached/fetched frozen embeddings.",
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
                "SUITES=\"libero_spatial libero_object libero_goal libero_10\" TASK_IDS=\"0 1 2 3 4 5 6 7 8 9\" NUM_TRIALS=10 STRESSORS=\"none\" sbatch slurm/openpi_libero_rollouts.sbatch",
                "SUITES=\"libero_spatial\" TASK_IDS=\"0 1 2 3 4 5 6 7 8 9\" NUM_TRIALS=7 STRESSORS=\"occlusion action_noise\" STRESSOR_SEVERITY=0.6 sbatch slurm/openpi_libero_rollouts.sbatch",
                "PYTHONPATH=src python scripts/train_openpi_risk.py --config configs/openpi/train_risk.yaml",
                "MODE=adaptive_chunk_openpi RISK_SUMMARY=reports/openpi_libero_risk_summary.json SUITES=\"libero_spatial\" TASK_IDS=\"0 1 2\" NUM_TRIALS=2 STRESSORS=\"occlusion\" STRESSOR_SEVERITY=1.0 sbatch slurm/openpi_libero_rollouts.sbatch",
                "```",
                "",
                "## Limitations",
                "",
                "- This is an OpenPI/LIBERO execution-risk study, not a formal safety guarantee.",
                "- The deployable structured model excludes injected stressor metadata; the metadata-aware model is reported only as a diagnostic upper bound.",
                "- VLM image embeddings and learned world-model dynamics are not claimed until image/embedding artifacts are present and audited.",
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


def _fit_openpi_variant(
    examples: Sequence[OpenPIRiskExample],
    splits: dict[str, list[OpenPIRiskExample]],
    *,
    feature_names: Sequence[str],
    description: str,
    uses_stressor_metadata: bool,
) -> dict[str, Any]:
    projected_splits = {name: _project_examples(items, feature_names) for name, items in splits.items()}
    train_items = projected_splits["train"]
    if len({example.label_failure for example in train_items}) < 2:
        return {
            "ok": False,
            "description": description,
            "uses_stressor_metadata": uses_stressor_metadata,
            "feature_names": list(feature_names),
            "blocker": "Training split needs both success and failure labels",
        }

    calibration_items = projected_splits["calibration"]
    calibration_warning = None
    if len({example.label_failure for example in calibration_items}) < 2:
        calibration_items = train_items
        calibration_warning = "Calibration split lacked both classes; reused train split for preliminary calibration"

    model = train_logistic_risk_model(train_items)
    calibrated = calibrate_temperature(model, calibration_items)
    calibration_probs = predict_examples(calibrated, calibration_items)
    calibration_labels = [example.label_failure for example in calibration_items]
    threshold = choose_threshold(calibration_labels, calibration_probs)
    split_metrics = {}
    baseline_metrics = {"global_prior": {}, "fixed_task_prior": {}}
    coverage_curves = {}
    prediction_rows = {}
    global_prior = _mean_label(projected_splits["train"])
    task_priors = _task_priors(projected_splits["train"], default=global_prior)
    for split_name, items in projected_splits.items():
        probs = predict_examples(calibrated, items)
        labels = [example.label_failure for example in items]
        split_metrics[split_name] = binary_risk_metrics(labels, probs, threshold=threshold)
        split_metrics[split_name]["reliability_bins"] = reliability_bins(labels, probs)
        coverage_curves[split_name] = _coverage_curve(labels, probs)
        prediction_rows[split_name] = _prediction_rows(items, probs)
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

    payload: dict[str, Any] = {
        "ok": True,
        "description": description,
        "uses_stressor_metadata": uses_stressor_metadata,
        "feature_names": list(calibrated.feature_names),
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
        "baseline_coverage_curves": {
            "global_prior": _baseline_coverage_curves(projected_splits, global_prior=global_prior, task_priors=task_priors, mode="global"),
            "fixed_task_prior": _baseline_coverage_curves(projected_splits, global_prior=global_prior, task_priors=task_priors, mode="fixed_task"),
        },
        "baseline_prediction_rows": {
            "test": {
                "global_prior": _prediction_rows(projected_splits["test"], [global_prior for _ in projected_splits["test"]]),
                "fixed_task_prior": _prediction_rows(
                    projected_splits["test"],
                    [_lookup_task_prior(task_priors, example, global_prior) for example in projected_splits["test"]],
                ),
            }
        },
        "coverage_curves": coverage_curves,
        "prediction_rows": {"test": prediction_rows["test"]},
    }
    if calibration_warning:
        payload["calibration_warning"] = calibration_warning
    return payload


def _vision_language_variant_status(examples: Sequence[OpenPIRiskExample]) -> dict[str, Any]:
    available = sorted(set(examples[0].feature_names).intersection(VLM_FEATURES)) if examples else []
    image_paths = sum(
        1
        for example in examples
        if bool(example.metadata.get("first_image_path")) or bool(example.metadata.get("embedding_path"))
    )
    if not available and image_paths == 0:
        return {
            "ok": False,
            "status": "skipped",
            "description": (
                "Frozen VLM/image embedding risk ablation. Current rollout JSONL does not contain saved RGB "
                "frame paths or embedding vectors, so this variant is explicitly not trained."
            ),
            "blocker": "Rerun a subset with --save-images or add an embedding extraction pass before claiming VLM risk results.",
        }
    return {
        "ok": False,
        "status": "blocked",
        "description": "VLM/image evidence is partially present, but no embedding feature adapter is implemented yet.",
        "available_feature_names": available,
        "episodes_with_image_or_embedding_paths": image_paths,
    }


def _primary_summary_fields(primary: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_names": primary["feature_names"],
        "calibration": primary["calibration"],
        "weights": primary["weights"],
        "normalization": primary["normalization"],
        "metrics": primary["metrics"],
        "baselines": primary["baselines"],
    }


def _project_examples(examples: Sequence[OpenPIRiskExample], feature_names: Sequence[str]) -> list[OpenPIRiskExample]:
    names = tuple(feature_names)
    output = []
    for example in examples:
        lookup = dict(zip(example.feature_names, example.features))
        missing = [name for name in names if name not in lookup]
        if missing:
            raise ValueError(f"Example {example.episode_id} is missing requested features: {missing}")
        output.append(replace(example, feature_names=names, features=tuple(float(lookup[name]) for name in names)))
    return output


def _coverage_curve(labels: Sequence[int], probs: Sequence[float]) -> list[dict[str, Any]]:
    if not labels:
        return []
    ranked = sorted(zip(labels, probs), key=lambda item: item[1])
    rows = []
    total = len(labels)
    for target in COVERAGE_TARGETS:
        attempted_count = max(1, min(total, round(total * target)))
        attempted = ranked[:attempted_count]
        failures = sum(label for label, _ in attempted)
        successes = attempted_count - failures
        threshold = max(prob for _, prob in attempted)
        rows.append(
            {
                "target_coverage": target,
                "coverage": attempted_count / total,
                "threshold": threshold,
                "attempted_success_rate": successes / attempted_count,
                "task_completion_rate": successes / total,
                "failure_rate_attempted": failures / attempted_count,
                "rejection_rate": 1.0 - attempted_count / total,
            }
        )
    return rows


def _baseline_coverage_curves(
    splits: dict[str, list[OpenPIRiskExample]],
    *,
    global_prior: float,
    task_priors: dict[tuple[str, int], float],
    mode: str,
) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for split_name, items in splits.items():
        labels = [example.label_failure for example in items]
        if mode == "global":
            probs = [global_prior for _ in items]
        elif mode == "fixed_task":
            probs = [_lookup_task_prior(task_priors, example, global_prior) for example in items]
        else:
            raise ValueError(f"Unknown baseline coverage mode: {mode}")
        output[split_name] = _coverage_curve(labels, probs)
    return output


def _offline_policy_comparison(model_variants: dict[str, Any], primary: dict[str, Any]) -> dict[str, Any]:
    test_metrics = primary.get("metrics", {}).get("test", {})
    direct_failure_rate = test_metrics.get("positive_rate")
    primary_rows = primary.get("prediction_rows", {}).get("test", [])
    comparison: dict[str, Any] = {
        "direct_openpi": _direct_policy_metrics(primary_rows, direct_failure_rate=direct_failure_rate),
        "global_prior_selective": _selective_policy_metrics(
            primary.get("baseline_prediction_rows", {}).get("test", {}).get("global_prior", []),
            target_coverage=0.9,
            note="Offline selective execution using the global training failure prior.",
        ),
        "fixed_task_prior_selective": _selective_policy_metrics(
            primary.get("baseline_prediction_rows", {}).get("test", {}).get("fixed_task_prior", []),
            target_coverage=0.9,
            note="Offline selective execution using per-suite/task training priors.",
        ),
    }
    for name in ("metadata_oracle_risk", "structured_progress_risk"):
        payload = model_variants.get(name, {})
        comparison[f"{name}_selective"] = _selective_policy_metrics(
            payload.get("prediction_rows", {}).get("test", []),
            target_coverage=0.9,
            note=f"Offline selective execution using `{name}` risk scores.",
        )
    comparison["adaptive_chunk_openpi_offline"] = _adaptive_chunk_metrics(primary_rows)
    comparison["early_abort_on_no_progress_offline"] = _early_abort_metrics(primary_rows)
    comparison["adaptive_chunk_plus_abort_offline"] = _adaptive_plus_abort_metrics(primary_rows)
    vision = model_variants.get("vision_language_risk", {})
    comparison["vision_language_risk_selective"] = {
        "status": vision.get("status", "blocked"),
        "blocker": vision.get("blocker"),
    }
    return comparison


def _curve_at_target(rows: Sequence[dict[str, Any]], target: float) -> dict[str, Any]:
    if not rows:
        return {"status": "missing"}
    row = min(rows, key=lambda item: abs(float(item.get("target_coverage", 0.0)) - target))
    return {"status": "evaluated", **row}


def _prediction_rows(examples: Sequence[OpenPIRiskExample], probs: Sequence[float]) -> list[dict[str, Any]]:
    rows = []
    for example, prob in zip(examples, probs):
        feature_lookup = dict(zip(example.feature_names, example.features))
        n_action_steps = int(round(float(feature_lookup.get("n_action_steps_scaled", 0.25)) * 20.0))
        rows.append(
            {
                "run_id": example.run_id,
                "episode_id": example.episode_id,
                "suite": example.suite,
                "task_id": example.task_id,
                "label_failure": example.label_failure,
                "label_timeout": example.label_timeout,
                "predicted_risk": float(prob),
                "episode_length": int(example.metadata.get("episode_length") or 0),
                "n_action_steps": max(1, n_action_steps),
                "prefix_no_progress_mean": float(feature_lookup.get("prefix_no_progress_mean", 0.0)),
                "prefix_reward_sum": float(feature_lookup.get("prefix_reward_sum", 0.0)),
            }
        )
    return rows


def _direct_policy_metrics(rows: Sequence[dict[str, Any]], *, direct_failure_rate: Any) -> dict[str, Any]:
    if not rows:
        return {
            "status": "evaluated",
            "coverage": 1.0,
            "task_completion_rate": None if direct_failure_rate is None else 1.0 - float(direct_failure_rate),
            "failure_rate_attempted": direct_failure_rate,
            "rejection_rate": 0.0,
        }
    outcomes = []
    for row in rows:
        success = int(row["label_failure"]) == 0
        timeout = int(row.get("label_timeout", row["label_failure"])) == 1
        length = int(row.get("episode_length", 0))
        direct_queries = _direct_queries(row)
        outcomes.append(
            {
                "success": success,
                "failure": not success,
                "timeout": timeout,
                "abstained": False,
                "utility": expected_utility(success=success, timeout=timeout, abstained=False, episode_length=length),
                "policy_queries": direct_queries,
                "direct_policy_queries": direct_queries,
                "extra_policy_queries": 0,
            }
        )
    return _policy_summary(outcomes, status="evaluated", note="Observed direct OpenPI test episodes.")


def _selective_policy_metrics(rows: Sequence[dict[str, Any]], *, target_coverage: float, note: str) -> dict[str, Any]:
    if not rows:
        return {"status": "missing", "note": note}
    total = len(rows)
    attempted_count = max(1, min(total, round(total * target_coverage)))
    ranked = sorted(rows, key=lambda row: float(row["predicted_risk"]))
    attempted_ids = {(row["run_id"], row["episode_id"]) for row in ranked[:attempted_count]}
    outcomes = []
    threshold = max(float(row["predicted_risk"]) for row in ranked[:attempted_count])
    for row in rows:
        attempted = (row["run_id"], row["episode_id"]) in attempted_ids
        success = attempted and int(row["label_failure"]) == 0
        timeout = attempted and int(row.get("label_timeout", row["label_failure"])) == 1
        length = int(row.get("episode_length", 0)) if attempted else 0
        direct_queries = _direct_queries(row)
        outcomes.append(
            {
                "success": success,
                "failure": attempted and not success,
                "timeout": timeout,
                "abstained": not attempted,
                "utility": expected_utility(success=success, timeout=timeout, abstained=not attempted, episode_length=length),
                "policy_queries": direct_queries if attempted else 0,
                "direct_policy_queries": direct_queries,
                "extra_policy_queries": -direct_queries if not attempted else 0,
            }
        )
    summary = _policy_summary(outcomes, status="evaluated", note=note)
    summary["target_coverage"] = target_coverage
    summary["threshold"] = threshold
    return summary


def _adaptive_chunk_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    outcomes = []
    for row in rows:
        success = int(row["label_failure"]) == 0
        timeout = int(row.get("label_timeout", row["label_failure"])) == 1
        length = int(row.get("episode_length", 0))
        direct_queries = _direct_queries(row)
        adaptive_queries = math.ceil(max(1, length) / _adaptive_horizon(float(row["predicted_risk"])))
        extra_queries = adaptive_queries - direct_queries
        outcomes.append(
            {
                "success": success,
                "failure": not success,
                "timeout": timeout,
                "abstained": False,
                "utility": expected_utility(
                    success=success,
                    timeout=timeout,
                    abstained=False,
                    episode_length=length,
                    extra_policy_queries=max(0, extra_queries),
                ),
                "policy_queries": adaptive_queries,
                "direct_policy_queries": direct_queries,
                "extra_policy_queries": extra_queries,
            }
        )
    return _policy_summary(
        outcomes,
        status="offline_counterfactual",
        note="Risk changes estimated action horizon and policy-query overhead only; success labels are not resimulated.",
    )


def _early_abort_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    outcomes = []
    for row in rows:
        abort = _would_abort_no_progress(row)
        success = (not abort) and int(row["label_failure"]) == 0
        timeout = (not abort) and int(row.get("label_timeout", row["label_failure"])) == 1
        original_length = int(row.get("episode_length", 0))
        length = min(original_length, 10) if abort else original_length
        direct_queries = _direct_queries(row)
        outcomes.append(
            {
                "success": success,
                "failure": (not abort) and not success,
                "timeout": timeout,
                "abstained": abort,
                "utility": expected_utility(success=success, timeout=timeout, abstained=abort, episode_length=length),
                "policy_queries": math.ceil(max(1, length) / max(1, int(row.get("n_action_steps", 5)))),
                "direct_policy_queries": direct_queries,
                "extra_policy_queries": 0,
            }
        )
    return _policy_summary(
        outcomes,
        status="offline_counterfactual",
        note="Aborts episodes whose first logged prefix has high no-progress; not a resimulated controller result.",
    )


def _adaptive_plus_abort_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    outcomes = []
    for row in rows:
        abort = _would_abort_no_progress(row)
        success = (not abort) and int(row["label_failure"]) == 0
        timeout = (not abort) and int(row.get("label_timeout", row["label_failure"])) == 1
        original_length = int(row.get("episode_length", 0))
        length = min(original_length, 10) if abort else original_length
        direct_queries = _direct_queries(row)
        adaptive_queries = math.ceil(max(1, length) / _adaptive_horizon(float(row["predicted_risk"])))
        extra_queries = adaptive_queries - direct_queries
        outcomes.append(
            {
                "success": success,
                "failure": (not abort) and not success,
                "timeout": timeout,
                "abstained": abort,
                "utility": expected_utility(
                    success=success,
                    timeout=timeout,
                    abstained=abort,
                    episode_length=length,
                    extra_policy_queries=max(0, extra_queries),
                ),
                "policy_queries": adaptive_queries,
                "direct_policy_queries": direct_queries,
                "extra_policy_queries": extra_queries,
            }
        )
    return _policy_summary(
        outcomes,
        status="offline_counterfactual",
        note="Combines adaptive horizon overhead estimate with the same prefix no-progress abort rule.",
    )


def _policy_summary(outcomes: Sequence[dict[str, Any]], *, status: str, note: str) -> dict[str, Any]:
    if not outcomes:
        return {"status": "missing", "note": note}
    total = len(outcomes)
    success_values = [float(item["success"]) for item in outcomes]
    failure_values = [float(item["failure"]) for item in outcomes]
    timeout_values = [float(item["timeout"]) for item in outcomes]
    abstain_values = [float(item["abstained"]) for item in outcomes]
    utility_values = [float(item["utility"]) for item in outcomes]
    query_values = [float(item["policy_queries"]) for item in outcomes]
    extra_query_values = [float(item["extra_policy_queries"]) for item in outcomes]
    return {
        "status": status,
        "note": note,
        "episodes": total,
        "coverage": 1.0 - sum(abstain_values) / total,
        "task_completion_rate": sum(success_values) / total,
        "failure_rate": sum(failure_values) / total,
        "failure_rate_attempted": sum(failure_values) / max(1.0, total - sum(abstain_values)),
        "timeout_rate": sum(timeout_values) / total,
        "abstention_rate": sum(abstain_values) / total,
        "rejection_rate": sum(abstain_values) / total,
        "expected_utility": sum(utility_values) / total,
        "mean_policy_queries": sum(query_values) / total,
        "mean_extra_policy_queries": sum(extra_query_values) / total,
        "policy_query_overhead": (sum(query_values) / total) / max(1e-9, sum(_direct_queries_from_outcome(item) for item in outcomes) / total),
        "ci95": {
            "task_completion_rate": bootstrap_ci(success_values, _mean_float, samples=500),
            "failure_rate": bootstrap_ci(failure_values, _mean_float, samples=500),
            "timeout_rate": bootstrap_ci(timeout_values, _mean_float, samples=500),
            "abstention_rate": bootstrap_ci(abstain_values, _mean_float, samples=500),
            "expected_utility": bootstrap_ci(utility_values, _mean_float, samples=500),
        },
    }


def _direct_queries(row: Mapping[str, Any]) -> int:
    return math.ceil(max(1, int(row.get("episode_length", 0))) / max(1, int(row.get("n_action_steps", 5))))


def _direct_queries_from_outcome(outcome: Mapping[str, Any]) -> float:
    return max(1.0, float(outcome.get("direct_policy_queries", outcome["policy_queries"])))


def _adaptive_horizon(risk: float) -> int:
    if risk < 0.35:
        return 10
    if risk < 0.65:
        return 5
    if risk < 0.85:
        return 2
    return 1


def _would_abort_no_progress(row: Mapping[str, Any]) -> bool:
    return float(row.get("prefix_no_progress_mean", 0.0)) >= 0.8 and float(row.get("prefix_reward_sum", 0.0)) <= 0.0


def _mean_float(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _dataset_summary(examples: Sequence[OpenPIRiskExample]) -> dict[str, Any]:
    by_suite: dict[str, int] = {}
    by_stressor: dict[str, int] = {}
    by_suite_stressor: dict[str, dict[str, int]] = {}
    by_stressor_severity: dict[str, int] = {}
    by_task: dict[str, int] = {}
    gpu_models: dict[str, int] = {}
    run_ids = set()
    for example in examples:
        by_suite[example.suite] = by_suite.get(example.suite, 0) + 1
        by_stressor[example.stressor_name] = by_stressor.get(example.stressor_name, 0) + 1
        severity_key = f"{example.stressor_name}:{example.stressor_severity:.2f}"
        by_stressor_severity[severity_key] = by_stressor_severity.get(severity_key, 0) + 1
        task_key = f"{example.suite}:task{example.task_id:02d}"
        by_task[task_key] = by_task.get(task_key, 0) + 1
        by_suite_stressor.setdefault(example.suite, {})
        by_suite_stressor[example.suite][example.stressor_name] = by_suite_stressor[example.suite].get(example.stressor_name, 0) + 1
        run_ids.add(example.run_id)
        gpu = str(example.metadata.get("cuda_device_name") or example.metadata.get("gpu_model") or "")
        if gpu:
            gpu_models[gpu] = gpu_models.get(gpu, 0) + 1
    return {
        "episodes": len(examples),
        "failures": sum(example.label_failure for example in examples),
        "successes": len(examples) - sum(example.label_failure for example in examples),
        "run_ids": len(run_ids),
        "by_suite": dict(sorted(by_suite.items())),
        "by_stressor": dict(sorted(by_stressor.items())),
        "by_stressor_severity": dict(sorted(by_stressor_severity.items())),
        "by_suite_stressor": {suite: dict(sorted(values.items())) for suite, values in sorted(by_suite_stressor.items())},
        "by_task": dict(sorted(by_task.items())),
        "gpu_models": dict(sorted(gpu_models.items())),
    }


def _variant_table(variants: dict[str, Any]) -> str:
    lines = [
        "| Variant | Status | Stressor metadata | Test AUROC | Test AUPRC | Test ECE | Coverage @ threshold | Note |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, payload in variants.items():
        metrics = payload.get("metrics", {}).get("test", {})
        status = "trained" if payload.get("ok") else str(payload.get("status", "blocked"))
        note = str(payload.get("blocker") or payload.get("description", "")).replace("|", "/")
        lines.append(
            f"| `{name}` | {status} | {payload.get('uses_stressor_metadata', False)} | "
            f"{_fmt(metrics.get('auroc'))} | {_fmt(metrics.get('auprc'))} | {_fmt(metrics.get('ece'))} | "
            f"{_fmt(metrics.get('coverage_at_threshold'))} | {note} |"
        )
    return "\n".join(lines)


def _offline_policy_table(comparison: dict[str, Any]) -> str:
    lines = [
        "| Policy | Status | Coverage | Task completion | Failure attempted | Timeout | Abstain | Utility | Query overhead | Note |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, payload in comparison.items():
        status = str(payload.get("status", "unknown"))
        note = str(payload.get("blocker", "") or payload.get("note", "") or "")
        lines.append(
            f"| `{name}` | {status} | {_fmt(payload.get('coverage'))} | "
            f"{_fmt(payload.get('task_completion_rate'))} | {_fmt(payload.get('failure_rate_attempted'))} | "
            f"{_fmt(payload.get('timeout_rate'))} | {_fmt(payload.get('abstention_rate'))} | "
            f"{_fmt(payload.get('expected_utility'))} | {_fmt(payload.get('policy_query_overhead'))} | {note} |"
        )
    return "\n".join(lines)


def _offline_policy_ci_table(comparison: dict[str, Any]) -> str:
    selected = [
        "direct_openpi",
        "fixed_task_prior_selective",
        "structured_progress_risk_selective",
        "metadata_oracle_risk_selective",
        "adaptive_chunk_openpi_offline",
        "early_abort_on_no_progress_offline",
        "adaptive_chunk_plus_abort_offline",
    ]
    lines = [
        "| Policy | Success CI | Failure CI | Timeout CI | Abstain CI | Utility CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in selected:
        payload = comparison.get(name, {})
        ci = payload.get("ci95", {}) if isinstance(payload, dict) else {}
        lines.append(
            f"| `{name}` | {_ci_fmt(ci.get('task_completion_rate'))} | {_ci_fmt(ci.get('failure_rate'))} | "
            f"{_ci_fmt(ci.get('timeout_rate'))} | {_ci_fmt(ci.get('abstention_rate'))} | "
            f"{_ci_fmt(ci.get('expected_utility'))} |"
        )
    return "\n".join(lines)


def _ci_fmt(ci: Any) -> str:
    if not isinstance(ci, dict) or ci.get("low") is None or ci.get("high") is None:
        return "n/a"
    return f"{float(ci['mean']):.3f} [{float(ci['low']):.3f}, {float(ci['high']):.3f}]"


def _write_openpi_figures(summary: dict[str, Any]) -> None:
    figures_dir = Path("reports/figures")
    reliability = summary.get("metrics", {}).get("test", {}).get("reliability_bins", [])
    coverage = summary.get("supervisor_offline", {}).get("coverage_curve", [])
    if reliability:
        _write_reliability_svg(
            reliability,
            figures_dir / "openpi_risk_reliability.svg",
            title=f"OpenPI {summary.get('primary_model', 'risk')} Reliability",
        )
    if coverage:
        _write_coverage_svg(
            coverage,
            figures_dir / "openpi_coverage_failure.svg",
            title="OpenPI Selective Execution: Coverage vs Failure",
        )


def _write_reliability_svg(bins: Sequence[dict[str, Any]], output_path: Path, *, title: str) -> None:
    width, height = 720, 460
    left, top, plot = 76, 64, 320
    bottom = top + plot
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 30, title, size=18, anchor="middle"),
        f'<line x1="{left}" y1="{bottom}" x2="{left + plot}" y2="{bottom}" stroke="#111"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#111"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{left + plot}" y2="{top}" stroke="#999" stroke-dasharray="4 4"/>',
    ]
    for tick in range(6):
        value = tick / 5
        x = left + value * plot
        y = bottom - value * plot
        elements.extend(
            [
                f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" y2="{bottom + 5}" stroke="#111"/>',
                f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#111"/>',
                _svg_text(x, bottom + 22, f"{value:.1f}", size=11, anchor="middle"),
                _svg_text(left - 10, y + 4, f"{value:.1f}", size=11, anchor="end"),
            ]
        )
    bin_width = plot / max(1, len(bins))
    for index, item in enumerate(bins):
        count = int(item.get("count", 0))
        if count == 0:
            continue
        predicted = float(item.get("mean_predicted_risk", 0.0))
        empirical = float(item.get("empirical_failure_rate", 0.0))
        x = left + predicted * plot
        y = bottom - empirical * plot
        radius = 4 + min(13, count / 25)
        bar_height = empirical * plot
        bar_x = left + index * bin_width
        elements.append(
            f'<rect x="{bar_x + 2:.1f}" y="{bottom - bar_height:.1f}" width="{max(1, bin_width - 4):.1f}" height="{bar_height:.1f}" fill="#90cdf4" fill-opacity="0.35"/>'
        )
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="#2563eb" fill-opacity="0.82"/>')
    elements.extend(
        [
            _svg_text(left + plot / 2, height - 28, "Predicted failure risk", size=13, anchor="middle"),
            f'<text x="18" y="{top + plot / 2:.1f}" font-size="13" text-anchor="middle" transform="rotate(-90 18 {top + plot / 2:.1f})">Empirical failure rate</text>',
            _svg_text(left + plot + 44, top + 34, "Dashed line: perfect calibration", size=12),
            _svg_text(left + plot + 44, top + 58, "Circle size: bin count", size=12),
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")


def _write_coverage_svg(rows: Sequence[dict[str, Any]], output_path: Path, *, title: str) -> None:
    width, height = 760, 460
    left, top, plot_w, plot_h = 76, 64, 500, 300
    bottom = top + plot_h
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 30, title, size=18, anchor="middle"),
        f'<line x1="{left}" y1="{bottom}" x2="{left + plot_w}" y2="{bottom}" stroke="#111"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#111"/>',
    ]
    for tick in range(6):
        value = tick / 5
        x = left + value * plot_w
        y = bottom - value * plot_h
        elements.extend(
            [
                f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" y2="{bottom + 5}" stroke="#111"/>',
                f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#111"/>',
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eee"/>',
                _svg_text(x, bottom + 22, f"{value:.1f}", size=11, anchor="middle"),
                _svg_text(left - 10, y + 4, f"{value:.1f}", size=11, anchor="end"),
            ]
        )
    points = []
    for row in rows:
        coverage = float(row.get("coverage", 0.0))
        failure = float(row.get("failure_rate_attempted", 0.0))
        x = left + coverage * plot_w
        y = bottom - failure * plot_h
        points.append((x, y, coverage, failure))
    if points:
        path = " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y, _, _) in enumerate(points))
        elements.append(f'<path d="{path}" fill="none" stroke="#dc2626" stroke-width="2.5"/>')
        for x, y, coverage, failure in points:
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#dc2626"/>')
            elements.append(_svg_text(x, y - 9, f"{coverage:.2f}/{failure:.2f}", size=10, anchor="middle"))
    elements.extend(
        [
            _svg_text(left + plot_w / 2, height - 28, "Coverage attempted", size=13, anchor="middle"),
            f'<text x="18" y="{top + plot_h / 2:.1f}" font-size="13" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2:.1f})">Failure rate among attempts</text>',
            _svg_text(left + plot_w + 34, top + 44, "Lower is better at a", size=12),
            _svg_text(left + plot_w + 34, top + 64, "given coverage.", size=12),
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")


def _svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}">{html.escape(text)}</text>'


def _mean_label(examples) -> float:
    return sum(example.label_failure for example in examples) / len(examples) if examples else 0.0


def _task_priors(examples, *, default: float) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[int]] = {}
    for example in examples:
        grouped.setdefault((example.suite, example.task_id), []).append(example.label_failure)
    return {key: sum(values) / len(values) for key, values in grouped.items()} or {("", -1): default}


def _lookup_task_prior(priors: dict[tuple[str, int], float], example, default: float) -> float:
    return priors.get((example.suite, example.task_id), default)
