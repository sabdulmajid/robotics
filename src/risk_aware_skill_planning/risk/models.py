from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from risk_aware_skill_planning.contracts import RiskEstimate, SkillCall
from risk_aware_skill_planning.envs.toy import ToyState, ground_truth_toy_risk


DEFAULT_FIXED_RISKS: Mapping[str, float] = {
    "direct_pick": 0.12,
    "conservative_pick": 0.10,
    "move_distractor": 0.07,
    "fast_place": 0.12,
    "slow_place": 0.10,
    "recover": 0.02,
}


@dataclass(frozen=True)
class ZeroRiskModel:
    def predict(self, state_features: Any, skill: SkillCall) -> RiskEstimate:
        return RiskEstimate(0.0, p_task_failure=0.0, p_safety_failure=0.0)


@dataclass(frozen=True)
class FixedPerSkillRiskModel:
    fixed_risks: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_FIXED_RISKS))

    def predict(self, state_features: Any, skill: SkillCall) -> RiskEstimate:
        risk = self.fixed_risks[skill.action_id]
        return RiskEstimate(risk)


@dataclass(frozen=True)
class OracleToyRiskModel:
    """Ground-truth state-conditioned toy risk model.

    This is an oracle for validating the planner and scenario design before any
    learned neural risk critic is introduced.
    """

    def predict(self, state_features: ToyState, skill: SkillCall) -> RiskEstimate:
        if not isinstance(state_features, ToyState):
            raise TypeError("OracleToyRiskModel expects a ToyState")
        return ground_truth_toy_risk(state_features, skill)


def make_toy_risk_model(mode: str):
    if mode == "naive_no_risk":
        return ZeroRiskModel()
    if mode == "fixed_per_skill_risk":
        return FixedPerSkillRiskModel()
    if mode in {"oracle_risk", "calibrated_state_risk", "uncalibrated_state_risk"}:
        return OracleToyRiskModel()
    raise ValueError(f"Unknown planner/risk mode: {mode}")
