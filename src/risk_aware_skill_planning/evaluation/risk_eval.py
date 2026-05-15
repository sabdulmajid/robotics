from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from risk_aware_skill_planning.calibration.metrics import brier_score, expected_calibration_error
from risk_aware_skill_planning.contracts import RiskModel
from risk_aware_skill_planning.datasets.toy_risk import (
    ToyRiskExample,
    generate_toy_risk_examples,
    toy_risk_split_stats,
)
from risk_aware_skill_planning.envs.toy import ground_truth_toy_risk
from risk_aware_skill_planning.evaluation.metrics import aggregate_episode_metrics
from risk_aware_skill_planning.planning.toy_planner import PlannerConfig, ToyPlanner, run_toy_episode
from risk_aware_skill_planning.risk.learned import (
    EmpiricalPerSkillRiskModel,
    GlobalPriorRiskModel,
    LogisticRiskModel,
    fit_temperature,
    negative_log_likelihood,
)
from risk_aware_skill_planning.risk.models import OracleToyRiskModel


def auroc(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    positives = [p for p, label in zip(probabilities, labels) if label]
    negatives = [p for p, label in zip(probabilities, labels) if not label]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def auprc(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    pairs = sorted(zip(probabilities, labels), key=lambda item: item[0], reverse=True)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    precision_recall_points = [(1.0, 0.0)]
    true_positives = 0
    false_positives = 0
    for _, label in pairs:
        if label:
            true_positives += 1
        else:
            false_positives += 1
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / positives
        precision_recall_points.append((precision, recall))
    area = 0.0
    previous_recall = 0.0
    for precision, recall in precision_recall_points[1:]:
        area += precision * (recall - previous_recall)
        previous_recall = recall
    return area


def selective_metrics(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    *,
    threshold: float,
) -> dict[str, float]:
    accepted = [(p, label) for p, label in zip(probabilities, labels) if p <= threshold]
    failures = [label for label in labels if label]
    false_negatives = [label for p, label in zip(probabilities, labels) if label and p <= threshold]
    return {
        "threshold": threshold,
        "coverage_at_threshold": len(accepted) / len(labels) if labels else 0.0,
        "selective_success_at_threshold": (
            sum(not label for _, label in accepted) / len(accepted) if accepted else 0.0
        ),
        "false_negative_rate_at_threshold": (
            len(false_negatives) / len(failures) if failures else 0.0
        ),
    }


def reliability_bins(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    *,
    num_bins: int = 10,
) -> list[dict[str, float | int]]:
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(num_bins)]
    for probability, label in zip(probabilities, labels):
        idx = min(num_bins - 1, int(probability * num_bins))
        bins[idx].append((probability, label))
    result: list[dict[str, float | int]] = []
    for index, items in enumerate(bins):
        if items:
            mean_probability = sum(probability for probability, _ in items) / len(items)
            empirical_rate = sum(label for _, label in items) / len(items)
        else:
            mean_probability = (index + 0.5) / num_bins
            empirical_rate = 0.0
        result.append(
            {
                "bin": index,
                "n": len(items),
                "mean_probability": mean_probability,
                "empirical_failure_rate": empirical_rate,
            }
        )
    return result


def evaluate_risk_model(
    name: str,
    model: RiskModel,
    examples: Sequence[ToyRiskExample],
    *,
    threshold: float,
) -> dict[str, object]:
    probabilities = [model.predict(example.state, example.skill).p_any_failure for example in examples]
    labels = [example.any_failure for example in examples]
    metrics: dict[str, object] = {
        "model": name,
        "n": len(examples),
        "brier": brier_score(probabilities, labels),
        "nll": negative_log_likelihood(probabilities, labels),
        "ece": expected_calibration_error(probabilities, labels),
        "auroc": auroc(probabilities, labels),
        "auprc": auprc(probabilities, labels),
        "mean_predicted_risk": sum(probabilities) / len(probabilities),
        "empirical_failure_rate": sum(labels) / len(labels),
        "reliability_bins": reliability_bins(probabilities, labels),
    }
    metrics.update(selective_metrics(probabilities, labels, threshold=threshold))
    return metrics


class TrueRiskOracleModel:
    def predict(self, state_features, skill):
        return ground_truth_toy_risk(state_features, skill)


def _format_float(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _metrics_markdown_table(metrics: Mapping[str, Mapping[str, object]]) -> str:
    rows = [
        "| Model | Brier | NLL | ECE | AUROC | AUPRC | Coverage @ threshold | Selective success | FNR @ threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in metrics.items():
        rows.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    _format_float(values["brier"]),
                    _format_float(values["nll"]),
                    _format_float(values["ece"]),
                    _format_float(values["auroc"]),
                    _format_float(values["auprc"]),
                    _format_float(values["coverage_at_threshold"]),
                    _format_float(values["selective_success_at_threshold"]),
                    _format_float(values["false_negative_rate_at_threshold"]),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _planner_markdown_table(metrics: Mapping[str, Mapping[str, object]]) -> str:
    rows = [
        "| Scenario | Planner | Task completion | Catastrophic failure | Coverage | Rejection |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scenario_id, mode_metrics in metrics.items():
        for planner_mode, values in mode_metrics.items():
            rows.append(
                "| "
                + " | ".join(
                    [
                        f"`{scenario_id}`",
                        f"`{planner_mode}`",
                        _format_float(values["task_completion_rate"]),
                        _format_float(values["catastrophic_failure_rate"]),
                        _format_float(values["coverage"]),
                        _format_float(values["rejection_rate"]),
                    ]
                )
                + " |"
            )
    return "\n".join(rows)


def run_toy_risk_learning(
    *,
    seed: int,
    train_examples: int,
    calibration_examples: int,
    test_examples: int,
    planner_eval_episodes: int,
    risk_threshold: float,
) -> dict[str, object]:
    train = generate_toy_risk_examples(num_examples=train_examples, seed=seed, split="train")
    calibration = generate_toy_risk_examples(
        num_examples=calibration_examples,
        seed=seed + 100_000,
        split="calibration",
    )
    test = generate_toy_risk_examples(num_examples=test_examples, seed=seed + 200_000, split="test")

    logistic = LogisticRiskModel.fit(train)
    calibrated = fit_temperature(logistic, calibration)
    models: dict[str, RiskModel] = {
        "global_prior": GlobalPriorRiskModel.fit(train),
        "per_skill_prior": EmpiricalPerSkillRiskModel.fit(train),
        "logistic_state_risk": logistic,
        "calibrated_logistic_state_risk": calibrated,
        "oracle_true_risk": TrueRiskOracleModel(),
    }
    risk_metrics = {
        name: evaluate_risk_model(name, model, test, threshold=risk_threshold)
        for name, model in models.items()
    }
    risk_metrics["calibrated_logistic_state_risk"]["temperature"] = calibrated.temperature

    scenario_ids = ("direct_pick_blocked_by_distractor", "far_bin_high_drop_risk")
    planner_models: dict[str, RiskModel] = {
        "naive_no_risk": models["global_prior"],
        "per_skill_prior": models["per_skill_prior"],
        "calibrated_logistic_state_risk": calibrated,
        "oracle_true_risk": OracleToyRiskModel(),
    }
    planner_metrics: dict[str, dict[str, object]] = {}
    for scenario_id in scenario_ids:
        planner_metrics[scenario_id] = {}
        for planner_mode, model in planner_models.items():
            if planner_mode == "naive_no_risk":
                config = PlannerConfig(
                    mode="naive_no_risk",
                    lambda_risk=0.0,
                    next_skill_threshold=1.0,
                    plan_threshold=1.0,
                )
                planner = ToyPlanner(config)
            else:
                config = PlannerConfig(
                    mode=planner_mode,
                    lambda_risk=3.0,
                    next_skill_threshold=0.85,
                    plan_threshold=0.85,
                )
                planner = ToyPlanner(config, risk_model=model)
            episodes = [
                run_toy_episode(scenario_id, planner, seed=seed + episode_index)
                for episode_index in range(planner_eval_episodes)
            ]
            planner_metrics[scenario_id][planner_mode] = aggregate_episode_metrics(episodes)

    return {
        "seed": seed,
        "risk_threshold": risk_threshold,
        "splits": {
            "train": toy_risk_split_stats(train),
            "calibration": toy_risk_split_stats(calibration),
            "test": toy_risk_split_stats(test),
        },
        "risk_metrics": risk_metrics,
        "planner_metrics": planner_metrics,
    }


def write_toy_risk_report(summary: Mapping[str, object], path: str | Path) -> None:
    risk_metrics = summary["risk_metrics"]
    planner_metrics = summary["planner_metrics"]
    content = "\n".join(
        [
            "# Toy Learned Risk Validation",
            "",
            "This note reports the first learned-risk sanity check in the toy domain.",
            "",
            "Scope: these models are trained on stochastic toy rollouts. They are not robosuite policies or manipulation risk critics.",
            "",
            f"Risk threshold for selective metrics: `{summary['risk_threshold']}`.",
            "",
            "## Skill-Level Risk Metrics",
            "",
            _metrics_markdown_table(risk_metrics),
            "",
            "## Planner Impact Check",
            "",
            _planner_markdown_table(planner_metrics),
            "",
            "Interpretation: the calibrated state-conditioned logistic model should beat the state-independent per-skill prior on Brier/ECE and should choose safer plans in the same scenarios where oracle risk helps.",
            "",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")


def write_json_summary(summary: Mapping[str, object], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

