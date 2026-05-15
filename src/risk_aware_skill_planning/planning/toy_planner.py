from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from risk_aware_skill_planning.contracts import (
    CandidatePlanLog,
    EpisodeResult,
    PlanningDecision,
    RiskModel,
    SkillCall,
    TERMINAL_REJECTED,
    TERMINAL_SUCCESS,
    TERMINAL_TIMEOUT,
    risk_union_bound_score,
    success_product_score,
)
from risk_aware_skill_planning.envs.toy import ToyState, ToySymbolicEnv, apply_toy_success_effect
from risk_aware_skill_planning.risk.models import FixedPerSkillRiskModel, make_toy_risk_model
from risk_aware_skill_planning.skills.toy_skills import (
    CONSERVATIVE_PICK,
    DIRECT_PICK,
    FAST_PLACE_FAR,
    FAST_PLACE_NEAR,
    MOVE_DISTRACTOR,
    SLOW_PLACE_FAR,
    SLOW_PLACE_NEAR,
    get_toy_skill_cost,
)


@dataclass(frozen=True)
class PlannerConfig:
    mode: str
    lambda_risk: float = 3.0
    next_skill_threshold: float = 0.85
    plan_threshold: float = 0.85
    max_replans: int = 8


def _place_options(state: ToyState) -> tuple[SkillCall, SkillCall]:
    if state.object_far:
        return (FAST_PLACE_FAR, SLOW_PLACE_FAR)
    return (FAST_PLACE_NEAR, SLOW_PLACE_NEAR)


def _valid_projected_plan(state: ToyState, plan: Iterable[SkillCall]) -> tuple[SkillCall, ...] | None:
    projected = state
    result: list[SkillCall] = []
    for skill in plan:
        action = skill.action_id
        if action in {"direct_pick", "conservative_pick"}:
            if not (projected.gripper_empty and not projected.holding_object and not projected.object_at_goal):
                return None
        elif action in {"fast_place", "slow_place"}:
            if not (projected.holding_object and not projected.object_at_goal):
                return None
        elif action == "move_distractor":
            if not (projected.gripper_empty and not projected.distractor_clear):
                return None
        projected = apply_toy_success_effect(projected, skill)
        result.append(skill)
    return tuple(result)


def enumerate_toy_candidate_plans(state: ToyState) -> tuple[tuple[SkillCall, ...], ...]:
    if state.object_at_goal:
        return ((),)

    if state.holding_object:
        return tuple((place,) for place in _place_options(state))

    pick_options = (DIRECT_PICK, CONSERVATIVE_PICK)
    place_options = _place_options(state)
    raw_plans: list[tuple[SkillCall, ...]] = []
    for pick in pick_options:
        for place in place_options:
            raw_plans.append((pick, place))
    if not state.distractor_clear:
        for pick in pick_options:
            for place in place_options:
                raw_plans.append((MOVE_DISTRACTOR, pick, place))

    valid_plans: list[tuple[SkillCall, ...]] = []
    for plan in raw_plans:
        valid = _valid_projected_plan(state, plan)
        if valid is not None:
            valid_plans.append(valid)
    return tuple(valid_plans)


class ToyPlanner:
    def __init__(self, config: PlannerConfig, risk_model: RiskModel | None = None) -> None:
        self.config = config
        self.risk_model = risk_model if risk_model is not None else make_toy_risk_model(config.mode)
        self.tail_risk_model = FixedPerSkillRiskModel()

    def plan(self, state: ToyState) -> PlanningDecision:
        candidate_plans = enumerate_toy_candidate_plans(state)
        if candidate_plans == ((),):
            return PlanningDecision(selected_plan=(), candidate_logs=(), rejected=False)

        logs = tuple(self._score_plan(state, plan) for plan in candidate_plans)
        accepted = [log for log in logs if not log.rejected_by_threshold]
        if not accepted:
            return PlanningDecision(
                selected_plan=None,
                candidate_logs=logs,
                rejected=True,
                rejection_reason="all_candidate_plans_exceed_threshold",
            )
        selected = min(accepted, key=lambda log: (log.total_cost, len(log.plan)))
        return PlanningDecision(selected_plan=selected.plan, candidate_logs=logs, rejected=False)

    def _score_plan(self, state: ToyState, plan: tuple[SkillCall, ...]) -> CandidatePlanLog:
        action_cost_sum = sum(get_toy_skill_cost(skill) for skill in plan)
        risks: list[float] = []
        next_skill_risk = 0.0
        tail_risk = 0.0
        if plan:
            next_skill_risk = self.risk_model.predict(state, plan[0]).p_any_failure
            risks.append(next_skill_risk)
            if self.config.mode != "naive_no_risk":
                for tail_skill in plan[1:]:
                    risk = self.tail_risk_model.predict(None, tail_skill).p_any_failure
                    tail_risk += risk
                    risks.append(risk)

        plan_risk = risk_union_bound_score(risks)
        total_cost = action_cost_sum + self.config.lambda_risk * plan_risk
        rejected = False
        reason = None
        if next_skill_risk > self.config.next_skill_threshold:
            rejected = True
            reason = "next_skill_risk_exceeds_threshold"
        elif plan_risk > self.config.plan_threshold:
            rejected = True
            reason = "plan_risk_exceeds_threshold"

        return CandidatePlanLog(
            plan=plan,
            action_cost_sum=action_cost_sum,
            next_skill_risk=next_skill_risk,
            tail_risk_score=tail_risk,
            risk_union_bound_score=plan_risk,
            success_product_score=success_product_score(risks),
            total_cost=total_cost,
            rejected_by_threshold=rejected,
            rejection_reason=reason,
        )


def _selected_plan_risk(decision: PlanningDecision) -> float:
    if decision.selected_plan is None:
        return 0.0
    selected = decision.selected_plan
    for log in decision.candidate_logs:
        if log.plan == selected:
            return log.risk_union_bound_score
    return 0.0


def run_toy_episode(
    scenario_id: str,
    planner: ToyPlanner,
    seed: int,
    max_replans: int | None = None,
) -> EpisodeResult:
    env = ToySymbolicEnv(scenario_id)
    env.reset(seed=seed, scenario_id=scenario_id)
    max_steps = max_replans if max_replans is not None else planner.config.max_replans

    executed_skills: list[SkillCall] = []
    outcomes = []
    planning_logs = []
    execution_cost = 0.0
    cumulative_predicted_risk = 0.0
    attempted = False
    failed_skill_executions = 0

    for _ in range(max_steps):
        if env.state.object_at_goal:
            return EpisodeResult(
                scenario_id=scenario_id,
                planner_mode=planner.config.mode,
                seed=seed,
                terminal_label=TERMINAL_SUCCESS,
                attempted=attempted,
                rejected=False,
                rejection_reason=None,
                execution_cost=execution_cost,
                cumulative_predicted_risk=cumulative_predicted_risk,
                replans=len(planning_logs),
                failed_skill_executions=failed_skill_executions,
                executed_skills=executed_skills,
                outcomes=outcomes,
                planning_logs=planning_logs,
            )

        decision = planner.plan(env.state)
        planning_logs.append(decision)
        if decision.rejected or decision.selected_plan is None:
            return EpisodeResult(
                scenario_id=scenario_id,
                planner_mode=planner.config.mode,
                seed=seed,
                terminal_label=TERMINAL_REJECTED,
                attempted=attempted,
                rejected=True,
                rejection_reason=decision.rejection_reason,
                execution_cost=execution_cost,
                cumulative_predicted_risk=cumulative_predicted_risk,
                replans=len(planning_logs),
                failed_skill_executions=failed_skill_executions,
                executed_skills=executed_skills,
                outcomes=outcomes,
                planning_logs=planning_logs,
            )
        next_skill = decision.selected_next_skill()
        if next_skill is None:
            continue
        attempted = True
        cumulative_predicted_risk += _selected_plan_risk(decision)
        execution_cost += get_toy_skill_cost(next_skill)
        executed_skills.append(next_skill)
        _, outcome = env.execute(next_skill)
        outcomes.append(outcome)
        if outcome.terminal_label != TERMINAL_SUCCESS:
            failed_skill_executions += 1
            return EpisodeResult(
                scenario_id=scenario_id,
                planner_mode=planner.config.mode,
                seed=seed,
                terminal_label=outcome.terminal_label,
                attempted=attempted,
                rejected=False,
                rejection_reason=None,
                execution_cost=execution_cost,
                cumulative_predicted_risk=cumulative_predicted_risk,
                replans=len(planning_logs),
                failed_skill_executions=failed_skill_executions,
                executed_skills=executed_skills,
                outcomes=outcomes,
                planning_logs=planning_logs,
            )

    return EpisodeResult(
        scenario_id=scenario_id,
        planner_mode=planner.config.mode,
        seed=seed,
        terminal_label=TERMINAL_TIMEOUT,
        attempted=attempted,
        rejected=False,
        rejection_reason=None,
        execution_cost=execution_cost,
        cumulative_predicted_risk=cumulative_predicted_risk,
        replans=len(planning_logs),
        failed_skill_executions=failed_skill_executions,
        executed_skills=executed_skills,
        outcomes=outcomes,
        planning_logs=planning_logs,
    )
