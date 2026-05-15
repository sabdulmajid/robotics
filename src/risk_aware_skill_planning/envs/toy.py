from __future__ import annotations

import random
from dataclasses import asdict, dataclass, replace
from typing import Mapping

from risk_aware_skill_planning.contracts import (
    EventFlags,
    FeatureSpec,
    RiskEstimate,
    RolloutOutcome,
    SkillCall,
    TERMINAL_SAFETY_FAILURE,
    TERMINAL_TASK_FAILURE,
)


@dataclass(frozen=True)
class ToyState:
    object_blocked: bool
    object_far: bool
    gripper_empty: bool
    holding_object: bool
    at_safe_pose: bool
    distractor_clear: bool
    object_at_goal: bool = False
    timestep: int = 0

    def feature_vector(self) -> tuple[float, ...]:
        return (
            float(self.object_blocked),
            float(self.object_far),
            float(self.gripper_empty),
            float(self.holding_object),
            float(self.at_safe_pose),
            float(self.distractor_clear),
            float(self.object_at_goal),
        )

    def to_dict(self) -> dict[str, bool | int]:
        return asdict(self)

    def with_updates(self, **kwargs: bool | int) -> "ToyState":
        return replace(self, **kwargs)


TOY_FIELD_NAMES = (
    "object_blocked",
    "object_far",
    "gripper_empty",
    "holding_object",
    "at_safe_pose",
    "distractor_clear",
    "object_at_goal",
)

TOY_FEATURE_SPEC = FeatureSpec(
    version="toy-v1",
    field_names=TOY_FIELD_NAMES,
    field_shapes={name: (1,) for name in TOY_FIELD_NAMES},
    normalization_mean={name: 0.0 for name in TOY_FIELD_NAMES},
    normalization_std={name: 1.0 for name in TOY_FIELD_NAMES},
    angle_representation="none",
    object_id_encoding={"none": 0, "can": 1, "distractor": 2},
    target_id_encoding={"none": 0, "near": 1, "far": 2, "safe_pose": 3},
    variant_id_encoding={"none": 0, "direct": 1, "conservative": 2, "fast": 3, "slow": 4},
)


@dataclass(frozen=True)
class ToyScenario:
    scenario_id: str
    description: str
    initial_state: ToyState


FROZEN_TOY_SCENARIOS: Mapping[str, ToyScenario] = {
    "direct_pick_blocked_by_distractor": ToyScenario(
        scenario_id="direct_pick_blocked_by_distractor",
        description="Can is blocked by a close distractor; direct pick is short but collision-prone.",
        initial_state=ToyState(
            object_blocked=True,
            object_far=False,
            gripper_empty=True,
            holding_object=False,
            at_safe_pose=True,
            distractor_clear=False,
        ),
    ),
    "far_bin_high_drop_risk": ToyScenario(
        scenario_id="far_bin_high_drop_risk",
        description="Can starts clear, but the far target makes fast placement drop-prone.",
        initial_state=ToyState(
            object_blocked=False,
            object_far=True,
            gripper_empty=True,
            holding_object=False,
            at_safe_pose=True,
            distractor_clear=True,
        ),
    ),
}


def ground_truth_toy_risk(state: ToyState, skill: SkillCall) -> RiskEstimate:
    action = skill.action_id
    if action == "direct_pick":
        if not state.gripper_empty or state.holding_object:
            return RiskEstimate(0.95, p_task_failure=0.95, p_safety_failure=0.0)
        if state.object_blocked and not state.distractor_clear:
            return RiskEstimate(0.75, p_task_failure=0.30, p_safety_failure=0.45)
        return RiskEstimate(0.12, p_task_failure=0.08, p_safety_failure=0.04)

    if action == "conservative_pick":
        if not state.gripper_empty or state.holding_object:
            return RiskEstimate(0.95, p_task_failure=0.95, p_safety_failure=0.0)
        if state.object_blocked and not state.distractor_clear:
            return RiskEstimate(0.28, p_task_failure=0.18, p_safety_failure=0.10)
        return RiskEstimate(0.10, p_task_failure=0.08, p_safety_failure=0.02)

    if action == "move_distractor":
        if state.distractor_clear:
            return RiskEstimate(0.03, p_task_failure=0.03, p_safety_failure=0.0)
        return RiskEstimate(0.07, p_task_failure=0.05, p_safety_failure=0.02)

    if action == "fast_place":
        if not state.holding_object:
            return RiskEstimate(0.95, p_task_failure=0.95, p_safety_failure=0.0)
        if state.object_far:
            return RiskEstimate(0.60, p_task_failure=0.25, p_safety_failure=0.35)
        return RiskEstimate(0.08, p_task_failure=0.05, p_safety_failure=0.03)

    if action == "slow_place":
        if not state.holding_object:
            return RiskEstimate(0.95, p_task_failure=0.95, p_safety_failure=0.0)
        if state.object_far:
            return RiskEstimate(0.15, p_task_failure=0.07, p_safety_failure=0.08)
        return RiskEstimate(0.06, p_task_failure=0.04, p_safety_failure=0.02)

    if action == "recover":
        return RiskEstimate(0.02, p_task_failure=0.02, p_safety_failure=0.0)

    raise KeyError(f"Unknown toy action: {action}")


def apply_toy_success_effect(state: ToyState, skill: SkillCall) -> ToyState:
    action = skill.action_id
    next_time = state.timestep + 1
    if action in {"direct_pick", "conservative_pick"}:
        return state.with_updates(
            holding_object=True,
            gripper_empty=False,
            at_safe_pose=False,
            timestep=next_time,
        )
    if action in {"fast_place", "slow_place"}:
        return state.with_updates(
            holding_object=False,
            gripper_empty=True,
            object_at_goal=True,
            at_safe_pose=False,
            timestep=next_time,
        )
    if action == "move_distractor":
        return state.with_updates(
            object_blocked=False,
            distractor_clear=True,
            at_safe_pose=False,
            timestep=next_time,
        )
    if action == "recover":
        return state.with_updates(at_safe_pose=True, timestep=next_time)
    raise KeyError(f"Unknown toy action: {action}")


def _failure_outcome(action: str, safety_failure: bool, timestep: int) -> RolloutOutcome:
    if safety_failure:
        if action in {"direct_pick", "conservative_pick", "move_distractor"}:
            flags = EventFlags(high_force_collision=True)
            first_event = "high_force_collision"
        else:
            flags = EventFlags(drop=True)
            first_event = "drop"
        return RolloutOutcome(
            terminal_label=TERMINAL_SAFETY_FAILURE,
            event_flags=flags,
            first_failure_event=first_event,
            failure_time=timestep,
        )

    if action in {"direct_pick", "conservative_pick"}:
        flags = EventFlags(miss=True)
        first_event = "miss"
    elif action in {"fast_place", "slow_place"}:
        flags = EventFlags(wrong_location=True)
        first_event = "wrong_location"
    else:
        flags = EventFlags(no_progress=True)
        first_event = "no_progress"
    return RolloutOutcome(
        terminal_label=TERMINAL_TASK_FAILURE,
        event_flags=flags,
        first_failure_event=first_event,
        failure_time=timestep,
    )


class ToySymbolicEnv:
    def __init__(self, scenario_id: str = "direct_pick_blocked_by_distractor") -> None:
        self._rng = random.Random()
        self.scenario_id = scenario_id
        self.state = FROZEN_TOY_SCENARIOS[scenario_id].initial_state

    @property
    def scenarios(self) -> Mapping[str, ToyScenario]:
        return FROZEN_TOY_SCENARIOS

    def reset(self, seed: int | None = None, scenario_id: str | None = None) -> ToyState:
        if seed is not None:
            self._rng = random.Random(seed)
        if scenario_id is not None:
            if scenario_id not in FROZEN_TOY_SCENARIOS:
                raise KeyError(f"Unknown toy scenario {scenario_id!r}")
            self.scenario_id = scenario_id
        self.state = FROZEN_TOY_SCENARIOS[self.scenario_id].initial_state
        return self.state

    def execute(self, skill: SkillCall) -> tuple[ToyState, RolloutOutcome]:
        estimate = ground_truth_toy_risk(self.state, skill)
        action = skill.action_id
        safety_p = estimate.p_safety_failure or 0.0
        task_p = estimate.p_task_failure
        if task_p is None:
            task_p = max(0.0, estimate.p_any_failure - safety_p)
        draw = self._rng.random()
        if draw < safety_p:
            outcome = _failure_outcome(action, safety_failure=True, timestep=self.state.timestep)
            return self.state, outcome
        if draw < safety_p + task_p:
            outcome = _failure_outcome(action, safety_failure=False, timestep=self.state.timestep)
            return self.state, outcome
        self.state = apply_toy_success_effect(self.state, skill)
        return self.state, RolloutOutcome.success()

