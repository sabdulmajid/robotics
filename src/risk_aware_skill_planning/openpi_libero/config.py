from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class OpenPILiberoSmokeConfig:
    experiment_id: str
    openpi_repo_url: str
    openpi_root: Path
    policy_config: str
    checkpoint: str
    suite: str
    task_id: int
    episodes: int
    seed: int
    default_action_horizon: int
    server_command: tuple[str, ...]
    client_command: tuple[str, ...]
    status_path: Path
    report_path: Path

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "OpenPILiberoSmokeConfig":
        openpi = _require_mapping(config, "openpi")
        libero = _require_mapping(config, "libero")
        outputs = _require_mapping(config, "outputs")
        return cls(
            experiment_id=str(config["experiment_id"]),
            openpi_repo_url=str(openpi["repo_url"]),
            openpi_root=Path(openpi["root"]),
            policy_config=str(openpi["policy_config"]),
            checkpoint=str(openpi["checkpoint"]),
            suite=str(libero["suite"]),
            task_id=int(libero["task_id"]),
            episodes=int(libero["episodes"]),
            seed=int(libero["seed"]),
            default_action_horizon=int(libero["default_action_horizon"]),
            server_command=tuple(str(part) for part in openpi["server_command"]),
            client_command=tuple(str(part) for part in libero["client_command"]),
            status_path=Path(outputs["status_path"]),
            report_path=Path(outputs["report_path"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "openpi_repo_url": self.openpi_repo_url,
            "openpi_root": str(self.openpi_root),
            "policy_config": self.policy_config,
            "checkpoint": self.checkpoint,
            "suite": self.suite,
            "task_id": self.task_id,
            "episodes": self.episodes,
            "seed": self.seed,
            "default_action_horizon": self.default_action_horizon,
            "server_command": list(self.server_command),
            "client_command": list(self.client_command),
            "status_path": str(self.status_path),
            "report_path": str(self.report_path),
        }


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config key {key!r} must be a mapping")
    return value


def load_smoke_config(path: str | Path) -> OpenPILiberoSmokeConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"Config {path} did not parse to a mapping")
    return OpenPILiberoSmokeConfig.from_mapping(data)
