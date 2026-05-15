from __future__ import annotations

from risk_aware_skill_planning.contracts import SkillCall, SkillSpec
from risk_aware_skill_planning.envs.toy import ToyState


ACTION_COSTS: dict[str, float] = {
    "direct_pick": 1.0,
    "conservative_pick": 1.6,
    "move_distractor": 0.8,
    "fast_place": 1.0,
    "slow_place": 1.7,
    "recover": 0.4,
}


DIRECT_PICK = SkillCall("pick", object_id="can", subgoal_id="grasp", variant_id="direct")
CONSERVATIVE_PICK = SkillCall("pick", object_id="can", subgoal_id="grasp", variant_id="conservative")
FAST_PLACE_NEAR = SkillCall("place", object_id="can", subgoal_id="near", variant_id="fast")
SLOW_PLACE_NEAR = SkillCall("place", object_id="can", subgoal_id="near", variant_id="slow")
FAST_PLACE_FAR = SkillCall("place", object_id="can", subgoal_id="far", variant_id="fast")
SLOW_PLACE_FAR = SkillCall("place", object_id="can", subgoal_id="far", variant_id="slow")
MOVE_DISTRACTOR = SkillCall("move_distractor", object_id="distractor", subgoal_id="clear")
RECOVER_SAFE_POSE = SkillCall("recover_safe_pose", subgoal_id="safe_pose")


def get_toy_skill_cost(skill: SkillCall) -> float:
    return ACTION_COSTS[skill.action_id]


def _pick_precondition(state: ToyState) -> bool:
    return state.gripper_empty and not state.holding_object and not state.object_at_goal


def _pick_postcondition(state: ToyState) -> bool:
    return state.holding_object and not state.gripper_empty


def _place_precondition(state: ToyState) -> bool:
    return state.holding_object and not state.object_at_goal


def _place_postcondition(state: ToyState) -> bool:
    return state.object_at_goal and state.gripper_empty and not state.holding_object


def _move_distractor_precondition(state: ToyState) -> bool:
    return not state.distractor_clear and state.gripper_empty


def _move_distractor_postcondition(state: ToyState) -> bool:
    return state.distractor_clear and not state.object_blocked


def _recover_precondition(state: ToyState) -> bool:
    return True


def _recover_postcondition(state: ToyState) -> bool:
    return state.at_safe_pose


def build_toy_skill_specs() -> dict[str, SkillSpec]:
    specs = {
        "direct_pick": SkillSpec(DIRECT_PICK, ACTION_COSTS["direct_pick"], _pick_precondition, _pick_postcondition),
        "conservative_pick": SkillSpec(
            CONSERVATIVE_PICK,
            ACTION_COSTS["conservative_pick"],
            _pick_precondition,
            _pick_postcondition,
        ),
        "move_distractor": SkillSpec(
            MOVE_DISTRACTOR,
            ACTION_COSTS["move_distractor"],
            _move_distractor_precondition,
            _move_distractor_postcondition,
        ),
        "fast_place_near": SkillSpec(FAST_PLACE_NEAR, ACTION_COSTS["fast_place"], _place_precondition, _place_postcondition),
        "slow_place_near": SkillSpec(SLOW_PLACE_NEAR, ACTION_COSTS["slow_place"], _place_precondition, _place_postcondition),
        "fast_place_far": SkillSpec(FAST_PLACE_FAR, ACTION_COSTS["fast_place"], _place_precondition, _place_postcondition),
        "slow_place_far": SkillSpec(SLOW_PLACE_FAR, ACTION_COSTS["slow_place"], _place_precondition, _place_postcondition),
        "recover": SkillSpec(RECOVER_SAFE_POSE, ACTION_COSTS["recover"], _recover_precondition, _recover_postcondition),
    }
    return specs

