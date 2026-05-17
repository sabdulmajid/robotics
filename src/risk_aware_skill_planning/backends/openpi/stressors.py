from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_STRESSORS = {
    "none",
    "occlusion",
    "gaussian_noise",
    "brightness",
    "action_noise",
    "action_delay",
    "action_precision",
}


@dataclass(frozen=True)
class StressorConfig:
    name: str = "none"
    severity: float = 0.0

    def validate(self) -> None:
        if self.name not in SUPPORTED_STRESSORS:
            raise ValueError(f"Unsupported stressor {self.name!r}; expected one of {sorted(SUPPORTED_STRESSORS)}")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("stressor severity must lie in [0, 1]")
        if self.name == "none" and self.severity != 0.0:
            raise ValueError("stressor 'none' must use severity 0.0")

    def to_cli_args(self) -> list[str]:
        self.validate()
        return ["--stressor-name", self.name, "--stressor-severity", str(self.severity)]
