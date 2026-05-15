from __future__ import annotations

from dataclasses import dataclass

from risk_aware_skill_planning.contracts import SkillCall
from risk_aware_skill_planning.envs.toy import ToyState


@dataclass(frozen=True)
class ScriptedToyPolicy:
    """Scripted policy placeholder used by the symbolic toy harness."""

    def action(self, state: ToyState, skill: SkillCall) -> str:
        return f"execute:{skill.action_id}:t={state.timestep}"

