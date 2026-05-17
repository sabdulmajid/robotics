from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from risk_aware_skill_planning.openpi_libero.supervisor import (
    RiskSupervisorConfig,
    SupervisorDecision,
    decide_supervisor_action,
)


@dataclass(frozen=True)
class OpenPISupervisorConfig:
    mode: str = "adaptive_chunk_openpi"
    risk: RiskSupervisorConfig = RiskSupervisorConfig()


def decide_openpi_supervision(
    predicted_risk: float | None,
    *,
    no_progress_steps: int = 0,
    config: OpenPISupervisorConfig | None = None,
) -> dict[str, Any]:
    cfg = config or OpenPISupervisorConfig()
    if predicted_risk is None or cfg.mode in ("direct_openpi", "learned_risk_openpi"):
        return {
            "action": cfg.mode,
            "action_horizon": cfg.risk.low_risk_horizon,
            "reason": "no_runtime_intervention",
            "predicted_risk": predicted_risk,
            "should_abstain": False,
            "should_replan": False,
        }
    decision: SupervisorDecision = decide_supervisor_action(
        predicted_risk,
        no_progress_steps=no_progress_steps,
        config=cfg.risk,
    )
    if cfg.mode == "selective_openpi" and predicted_risk < cfg.risk.abstain_threshold:
        return {
            "action": "selective_openpi",
            "action_horizon": cfg.risk.low_risk_horizon,
            "reason": "risk_below_abstain_threshold",
            "predicted_risk": predicted_risk,
            "should_abstain": False,
            "should_replan": False,
        }
    return decision.to_dict()
