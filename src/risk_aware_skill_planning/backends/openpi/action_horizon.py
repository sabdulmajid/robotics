from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionHorizonPolicy:
    low_threshold: float = 0.35
    medium_threshold: float = 0.65
    high_threshold: float = 0.85
    abstain_threshold: float = 0.95
    low_horizon: int = 10
    medium_horizon: int = 5
    high_horizon: int = 2
    extreme_horizon: int = 1

    def validate(self) -> None:
        thresholds = [self.low_threshold, self.medium_threshold, self.high_threshold, self.abstain_threshold]
        if thresholds != sorted(thresholds):
            raise ValueError("Action-horizon risk thresholds must be non-decreasing")
        if self.low_threshold < 0.0 or self.abstain_threshold > 1.0:
            raise ValueError("Action-horizon thresholds must lie in [0, 1]")
        for horizon in (self.low_horizon, self.medium_horizon, self.high_horizon, self.extreme_horizon):
            if horizon <= 0:
                raise ValueError("Action horizons must be positive")

    def horizon_for_risk(self, risk: float) -> int:
        self.validate()
        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk must lie in [0, 1]")
        if risk < self.low_threshold:
            return self.low_horizon
        if risk < self.medium_threshold:
            return self.medium_horizon
        if risk < self.high_threshold:
            return self.high_horizon
        return self.extreme_horizon

    def should_abstain(self, risk: float) -> bool:
        self.validate()
        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk must lie in [0, 1]")
        return risk >= self.abstain_threshold
