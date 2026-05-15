from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Iterable

from risk_aware_skill_planning.contracts import SkillCall
from risk_aware_skill_planning.envs.toy import ToyState, ground_truth_toy_risk
from risk_aware_skill_planning.skills.toy_skills import (
    CONSERVATIVE_PICK,
    DIRECT_PICK,
    FAST_PLACE_FAR,
    FAST_PLACE_NEAR,
    MOVE_DISTRACTOR,
    RECOVER_SAFE_POSE,
    SLOW_PLACE_FAR,
    SLOW_PLACE_NEAR,
)


TOY_RISK_SKILLS: tuple[SkillCall, ...] = (
    DIRECT_PICK,
    CONSERVATIVE_PICK,
    MOVE_DISTRACTOR,
    FAST_PLACE_NEAR,
    SLOW_PLACE_NEAR,
    FAST_PLACE_FAR,
    SLOW_PLACE_FAR,
    RECOVER_SAFE_POSE,
)


@dataclass(frozen=True)
class ToyRiskExample:
    state: ToyState
    skill: SkillCall
    true_risk: float
    any_failure: bool
    seed: int
    split: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model_inputs": {
                "initial_state_features": self.state.to_dict(),
                "skill": self.skill.to_dict(),
            },
            "labels": {
                "any_failure": self.any_failure,
                "true_risk": self.true_risk,
            },
            "analysis_metadata": {
                "seed": self.seed,
                "split": self.split,
            },
        }


def sample_toy_state(rng: random.Random) -> ToyState:
    holding_object = rng.random() < 0.45
    distractor_clear = rng.random() < 0.55
    object_blocked = (not distractor_clear and rng.random() < 0.75) or rng.random() < 0.10
    return ToyState(
        object_blocked=object_blocked,
        object_far=rng.random() < 0.50,
        gripper_empty=not holding_object,
        holding_object=holding_object,
        at_safe_pose=rng.random() < 0.60,
        distractor_clear=distractor_clear,
        object_at_goal=False,
        timestep=rng.randrange(4),
    )


def generate_toy_risk_examples(
    *,
    num_examples: int,
    seed: int,
    split: str,
    skills: Iterable[SkillCall] = TOY_RISK_SKILLS,
) -> list[ToyRiskExample]:
    rng = random.Random(seed)
    skill_list = tuple(skills)
    examples: list[ToyRiskExample] = []
    for index in range(num_examples):
        state = sample_toy_state(rng)
        skill = skill_list[rng.randrange(len(skill_list))]
        risk = ground_truth_toy_risk(state, skill).p_any_failure
        label = rng.random() < risk
        examples.append(
            ToyRiskExample(
                state=state,
                skill=skill,
                true_risk=risk,
                any_failure=label,
                seed=seed + index,
                split=split,
            )
        )
    return examples


def toy_risk_dataset_to_dict(examples: Iterable[ToyRiskExample]) -> list[dict[str, object]]:
    return [example.to_dict() for example in examples]


def toy_risk_split_stats(examples: Iterable[ToyRiskExample]) -> dict[str, object]:
    items = list(examples)
    by_skill: dict[str, dict[str, int]] = {}
    for example in items:
        stats = by_skill.setdefault(example.skill.action_id, {"n": 0, "failures": 0})
        stats["n"] += 1
        stats["failures"] += int(example.any_failure)
    return {
        "n": len(items),
        "failure_rate": sum(example.any_failure for example in items) / len(items) if items else 0.0,
        "by_skill": by_skill,
    }

