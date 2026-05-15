from __future__ import annotations

from risk_aware_skill_planning.envs.toy import TOY_FEATURE_SPEC, ToySymbolicEnv
from risk_aware_skill_planning.evaluation.toy_eval import run_toy_suite
from risk_aware_skill_planning.planning.toy_planner import PlannerConfig, ToyPlanner, run_toy_episode
from risk_aware_skill_planning.skills.toy_skills import MOVE_DISTRACTOR


def test_feature_spec_validates() -> None:
    TOY_FEATURE_SPEC.validate()


def test_seeded_reset_is_deterministic() -> None:
    env = ToySymbolicEnv()
    first = env.reset(seed=123, scenario_id="direct_pick_blocked_by_distractor")
    second = env.reset(seed=123, scenario_id="direct_pick_blocked_by_distractor")
    assert first == second
    assert first.feature_vector() == second.feature_vector()


def test_oracle_planner_chooses_move_distractor_when_pick_blocked() -> None:
    env = ToySymbolicEnv("direct_pick_blocked_by_distractor")
    state = env.reset(seed=0)
    planner = ToyPlanner(PlannerConfig(mode="oracle_risk"))
    decision = planner.plan(state)
    assert not decision.rejected
    assert decision.selected_next_skill() == MOVE_DISTRACTOR
    rejected_actions = {
        log.plan[0].action_id
        for log in decision.candidate_logs
        if log.rejected_by_threshold and log.rejection_reason is not None
    }
    assert "direct_pick" in rejected_actions


def test_naive_planner_logs_zero_risk() -> None:
    env = ToySymbolicEnv("direct_pick_blocked_by_distractor")
    state = env.reset(seed=0)
    planner = ToyPlanner(PlannerConfig(mode="naive_no_risk", lambda_risk=3.0))
    decision = planner.plan(state)
    assert not decision.rejected
    assert decision.selected_next_skill() is not None
    assert all(log.risk_union_bound_score == 0.0 for log in decision.candidate_logs)
    assert all(log.tail_risk_score == 0.0 for log in decision.candidate_logs)


def test_oracle_replans_to_slow_place_for_far_target() -> None:
    env = ToySymbolicEnv("far_bin_high_drop_risk")
    state = env.reset(seed=0).with_updates(holding_object=True, gripper_empty=False)
    planner = ToyPlanner(PlannerConfig(mode="oracle_risk"))
    decision = planner.plan(state)
    assert not decision.rejected
    next_skill = decision.selected_next_skill()
    assert next_skill is not None
    assert next_skill.action_id == "slow_place"


def test_strict_threshold_rejects_all_plans() -> None:
    env = ToySymbolicEnv("direct_pick_blocked_by_distractor")
    state = env.reset(seed=0)
    planner = ToyPlanner(PlannerConfig(mode="oracle_risk", plan_threshold=0.20))
    decision = planner.plan(state)
    assert decision.rejected
    assert decision.rejection_reason == "all_candidate_plans_exceed_threshold"


def test_oracle_beats_naive_and_fixed_on_frozen_scenarios() -> None:
    summary = run_toy_suite(
        ["direct_pick_blocked_by_distractor", "far_bin_high_drop_risk"],
        ["naive_no_risk", "fixed_per_skill_risk", "oracle_risk"],
        num_episodes=500,
        seed_start=0,
    )
    for scenario_id in ["direct_pick_blocked_by_distractor", "far_bin_high_drop_risk"]:
        naive = summary[scenario_id]["naive_no_risk"]
        fixed = summary[scenario_id]["fixed_per_skill_risk"]
        oracle = summary[scenario_id]["oracle_risk"]
        assert oracle["task_completion_rate"] > naive["task_completion_rate"] + 0.25
        assert oracle["task_completion_rate"] > fixed["task_completion_rate"] + 0.25
        assert oracle["catastrophic_failure_rate"] < naive["catastrophic_failure_rate"] - 0.15
        assert oracle["catastrophic_failure_rate"] < fixed["catastrophic_failure_rate"] - 0.15
        assert oracle["coverage"] == 1.0
        assert oracle["rejection_rate"] == 0.0


def test_episode_logs_candidate_plans_and_replans() -> None:
    planner = ToyPlanner(PlannerConfig(mode="oracle_risk"))
    episode = run_toy_episode("direct_pick_blocked_by_distractor", planner, seed=7)
    assert episode.planning_logs
    assert episode.planning_logs[0].candidate_logs
    assert episode.executed_skills[0] == MOVE_DISTRACTOR
