from __future__ import annotations

from dataclasses import dataclass

from risk_aware_skill_planning.contracts import RolloutOutcome


@dataclass(frozen=True)
class ToyMonitor:
    """Thin monitor shim for the toy harness.

    The toy simulator emits terminal event flags directly. This class preserves
    the monitor interface that the manipulation environment will expand later.
    """

    def triggered(self, outcome: RolloutOutcome) -> bool:
        return outcome.event_flags.any()

