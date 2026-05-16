from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RiskSupervisorConfig:
    low_threshold: float = 0.35
    medium_threshold: float = 0.65
    high_threshold: float = 0.85
    abstain_threshold: float = 0.95
    low_risk_horizon: int = 10
    medium_risk_horizon: int = 5
    high_risk_horizon: int = 2
    extreme_risk_horizon: int = 1
    no_progress_patience: int = 8

    def validate(self) -> None:
        thresholds = [self.low_threshold, self.medium_threshold, self.high_threshold, self.abstain_threshold]
        if thresholds != sorted(thresholds):
            raise ValueError("Risk thresholds must be non-decreasing")
        if self.low_threshold < 0.0 or self.abstain_threshold > 1.0:
            raise ValueError("Risk thresholds must lie in [0, 1]")
        for horizon in (
            self.low_risk_horizon,
            self.medium_risk_horizon,
            self.high_risk_horizon,
            self.extreme_risk_horizon,
        ):
            if horizon <= 0:
                raise ValueError("Action horizons must be positive")


@dataclass(frozen=True)
class SupervisorDecision:
    action: str
    action_horizon: int
    reason: str
    predicted_risk: float
    should_abstain: bool = False
    should_replan: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def decide_supervisor_action(
    predicted_risk: float,
    *,
    no_progress_steps: int = 0,
    config: RiskSupervisorConfig | None = None,
) -> SupervisorDecision:
    cfg = config or RiskSupervisorConfig()
    cfg.validate()
    if not 0.0 <= predicted_risk <= 1.0:
        raise ValueError("predicted_risk must be in [0, 1]")
    if no_progress_steps >= cfg.no_progress_patience and predicted_risk >= cfg.medium_threshold:
        return SupervisorDecision(
            action="no_progress_replan",
            action_horizon=cfg.extreme_risk_horizon,
            reason="no_progress_with_elevated_risk",
            predicted_risk=predicted_risk,
            should_replan=True,
        )
    if predicted_risk >= cfg.abstain_threshold:
        return SupervisorDecision(
            action="abstain",
            action_horizon=cfg.extreme_risk_horizon,
            reason="risk_exceeds_abstain_threshold",
            predicted_risk=predicted_risk,
            should_abstain=True,
        )
    if predicted_risk >= cfg.high_threshold:
        return SupervisorDecision(
            action="adaptive_chunk_openpi",
            action_horizon=cfg.extreme_risk_horizon,
            reason="extreme_risk_short_horizon",
            predicted_risk=predicted_risk,
        )
    if predicted_risk >= cfg.medium_threshold:
        return SupervisorDecision(
            action="adaptive_chunk_openpi",
            action_horizon=cfg.high_risk_horizon,
            reason="high_risk_short_horizon",
            predicted_risk=predicted_risk,
        )
    if predicted_risk >= cfg.low_threshold:
        return SupervisorDecision(
            action="adaptive_chunk_openpi",
            action_horizon=cfg.medium_risk_horizon,
            reason="medium_risk_medium_horizon",
            predicted_risk=predicted_risk,
        )
    return SupervisorDecision(
        action="direct_openpi",
        action_horizon=cfg.low_risk_horizon,
        reason="low_risk_default_horizon",
        predicted_risk=predicted_risk,
    )
