from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class OpenPIBackendConfig:
    root: Path = Path("external/openpi")
    policy_config: str = "pi05_libero"
    checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero/"
    server_host: str = "127.0.0.1"
    server_port: int = 18000

    def validate(self) -> None:
        if not self.policy_config:
            raise ValueError("policy_config is required")
        if not self.checkpoint:
            raise ValueError("checkpoint is required")
        if self.server_port <= 0:
            raise ValueError("server_port must be positive")


@dataclass(frozen=True)
class LiberoRunConfig:
    suites: tuple[str, ...] = ("libero_spatial",)
    task_ids: tuple[int, ...] = (0,)
    episodes_per_task: int = 1
    seed: int = 7
    n_action_steps: int = 10
    stressors: tuple[str, ...] = ("none",)
    stressor_severity: float = 0.0

    def validate(self) -> None:
        if not self.suites:
            raise ValueError("At least one LIBERO suite is required")
        if not self.task_ids:
            raise ValueError("At least one task id is required")
        if self.episodes_per_task <= 0:
            raise ValueError("episodes_per_task must be positive")
        if self.n_action_steps <= 0:
            raise ValueError("n_action_steps must be positive")
        if not 0.0 <= self.stressor_severity <= 1.0:
            raise ValueError("stressor_severity must lie in [0, 1]")


@dataclass(frozen=True)
class OpenPIExperimentConfig:
    experiment_id: str
    openpi: OpenPIBackendConfig = field(default_factory=OpenPIBackendConfig)
    libero: LiberoRunConfig = field(default_factory=LiberoRunConfig)
    mode: str = "direct_openpi"
    risk_summary: Path | None = None
    output_dir: Path = Path("outputs/openpi_libero")
    rollout_dir: Path = Path("datasets/openpi_libero_rollouts")
    video_dir: Path = Path("videos")

    def validate(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        self.openpi.validate()
        self.libero.validate()
        if self.mode not in {
            "direct_openpi",
            "fixed_task_prior",
            "learned_risk_openpi",
            "selective_openpi",
            "adaptive_chunk_openpi",
            "no_progress_replan",
        }:
            raise ValueError(f"Unsupported OpenPI mode: {self.mode}")


def load_openpi_experiment_config(path: str | Path) -> OpenPIExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config {path} did not parse to a mapping")
    config = OpenPIExperimentConfig(
        experiment_id=str(raw.get("experiment_id", Path(path).stem)),
        openpi=_parse_openpi(raw.get("openpi", {})),
        libero=_parse_libero(raw.get("libero", {})),
        mode=str(raw.get("mode", raw.get("execution", {}).get("mode", "direct_openpi"))),
        risk_summary=Path(raw["risk_summary"]) if raw.get("risk_summary") else None,
        output_dir=Path(raw.get("outputs", {}).get("output_dir", "outputs/openpi_libero")),
        rollout_dir=Path(raw.get("outputs", {}).get("rollout_dir", "datasets/openpi_libero_rollouts")),
        video_dir=Path(raw.get("outputs", {}).get("video_dir", "videos")),
    )
    config.validate()
    return config


def _parse_openpi(raw: Any) -> OpenPIBackendConfig:
    data = raw if isinstance(raw, Mapping) else {}
    return OpenPIBackendConfig(
        root=Path(data.get("root", "external/openpi")),
        policy_config=str(data.get("policy_config", "pi05_libero")),
        checkpoint=str(data.get("checkpoint", "gs://openpi-assets/checkpoints/pi05_libero/")),
        server_host=str(data.get("server_host", "127.0.0.1")),
        server_port=int(data.get("server_port", 18000)),
    )


def _parse_libero(raw: Any) -> LiberoRunConfig:
    data = raw if isinstance(raw, Mapping) else {}
    return LiberoRunConfig(
        suites=tuple(str(item) for item in data.get("suites", ["libero_spatial"])),
        task_ids=tuple(int(item) for item in data.get("task_ids", data.get("tasks", [0]))),
        episodes_per_task=int(data.get("episodes_per_task", data.get("num_trials", 1))),
        seed=int(data.get("seed", data.get("seed_start", 7))),
        n_action_steps=int(data.get("n_action_steps", data.get("default_action_horizon", 10))),
        stressors=tuple(str(item) for item in data.get("stressors", ["none"])),
        stressor_severity=float(data.get("stressor_severity", 0.0)),
    )
