from __future__ import annotations

from risk_aware_skill_planning.evaluation.risk_eval import run_toy_risk_learning


def test_toy_learned_state_risk_beats_per_skill_prior() -> None:
    summary = run_toy_risk_learning(
        seed=13,
        train_examples=1200,
        calibration_examples=500,
        test_examples=500,
        planner_eval_episodes=120,
        risk_threshold=0.35,
    )
    metrics = summary["risk_metrics"]
    per_skill = metrics["per_skill_prior"]
    calibrated = metrics["calibrated_logistic_state_risk"]
    assert calibrated["brier"] < per_skill["brier"]
    assert calibrated["auroc"] > per_skill["auroc"]


def test_toy_learned_state_risk_improves_planner_behavior() -> None:
    summary = run_toy_risk_learning(
        seed=13,
        train_examples=1200,
        calibration_examples=500,
        test_examples=500,
        planner_eval_episodes=120,
        risk_threshold=0.35,
    )
    for scenario_id, scenario_metrics in summary["planner_metrics"].items():
        naive = scenario_metrics["naive_no_risk"]
        learned = scenario_metrics["calibrated_logistic_state_risk"]
        assert learned["task_completion_rate"] > naive["task_completion_rate"]
        assert learned["catastrophic_failure_rate"] < naive["catastrophic_failure_rate"]
