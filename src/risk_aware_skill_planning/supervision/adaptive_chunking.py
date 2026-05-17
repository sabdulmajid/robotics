from __future__ import annotations

from risk_aware_skill_planning.backends.openpi.action_horizon import ActionHorizonPolicy


def adaptive_horizon_for_risk(risk: float, policy: ActionHorizonPolicy | None = None) -> int:
    return (policy or ActionHorizonPolicy()).horizon_for_risk(risk)
