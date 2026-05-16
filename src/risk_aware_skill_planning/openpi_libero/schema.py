from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class OpenPILiberoStepLog:
    timestep: int
    observation_keys: tuple[str, ...]
    action_chunk_length: int
    action_norm: float
    action_smoothness: float | None = None
    predicted_risk: float | None = None
    action_horizon: int | None = None
    reward: float | None = None
    done: bool = False
    success: bool | None = None
    no_progress: bool | None = None

    def validate(self) -> None:
        if self.timestep < 0:
            raise ValueError("timestep must be non-negative")
        if self.action_chunk_length < 0:
            raise ValueError("action_chunk_length must be non-negative")
        if self.action_norm < 0.0:
            raise ValueError("action_norm must be non-negative")
        if self.predicted_risk is not None and not 0.0 <= self.predicted_risk <= 1.0:
            raise ValueError("predicted_risk must be in [0, 1]")
        if self.action_horizon is not None and self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self) | {"observation_keys": list(self.observation_keys)}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OpenPILiberoStepLog":
        return cls(
            timestep=int(payload["timestep"]),
            observation_keys=tuple(str(key) for key in payload.get("observation_keys", ())),
            action_chunk_length=int(payload["action_chunk_length"]),
            action_norm=float(payload["action_norm"]),
            action_smoothness=_optional_float(payload.get("action_smoothness")),
            predicted_risk=_optional_float(payload.get("predicted_risk")),
            action_horizon=_optional_int(payload.get("action_horizon")),
            reward=_optional_float(payload.get("reward")),
            done=bool(payload.get("done", False)),
            success=_optional_bool(payload.get("success")),
            no_progress=_optional_bool(payload.get("no_progress")),
        )


@dataclass(frozen=True)
class OpenPILiberoEpisodeLog:
    episode_id: str
    suite: str
    task_id: int
    task_language: str
    seed: int
    mode: str
    policy_config: str
    checkpoint: str
    action_horizon: int
    terminal_label: str
    success: bool
    timeout: bool
    steps: tuple[OpenPILiberoStepLog, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id is required")
        if self.task_id < 0:
            raise ValueError("task_id must be non-negative")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        for step in self.steps:
            step.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        data["steps"] = [step.to_dict() for step in self.steps]
        data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OpenPILiberoEpisodeLog":
        steps = tuple(OpenPILiberoStepLog.from_mapping(step) for step in payload.get("steps", ()))
        return cls(
            episode_id=str(payload["episode_id"]),
            suite=str(payload["suite"]),
            task_id=int(payload["task_id"]),
            task_language=str(payload.get("task_language", "")),
            seed=int(payload["seed"]),
            mode=str(payload["mode"]),
            policy_config=str(payload["policy_config"]),
            checkpoint=str(payload["checkpoint"]),
            action_horizon=int(payload["action_horizon"]),
            terminal_label=str(payload["terminal_label"]),
            success=bool(payload["success"]),
            timeout=bool(payload["timeout"]),
            steps=steps,
            metadata=dict(payload.get("metadata", {})),
        )


def write_episode_jsonl(episodes: Iterable[OpenPILiberoEpisodeLog], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_dict(), sort_keys=True) + "\n")


def read_episode_jsonl(path: str | Path) -> list[OpenPILiberoEpisodeLog]:
    episodes: list[OpenPILiberoEpisodeLog] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            episodes.append(OpenPILiberoEpisodeLog.from_mapping(payload))
    return episodes


def summarize_episode_logs(episodes: Iterable[OpenPILiberoEpisodeLog]) -> dict[str, Any]:
    items = list(episodes)
    by_mode: dict[str, dict[str, Any]] = {}
    for episode in items:
        stats = by_mode.setdefault(
            episode.mode,
            {"episodes": 0, "successes": 0, "timeouts": 0, "steps": [], "terminal_labels": {}},
        )
        stats["episodes"] += 1
        stats["successes"] += int(episode.success)
        stats["timeouts"] += int(episode.timeout)
        stats["steps"].append(len(episode.steps))
        labels = stats["terminal_labels"]
        labels[episode.terminal_label] = labels.get(episode.terminal_label, 0) + 1
    for stats in by_mode.values():
        steps = stats.pop("steps")
        stats["success_rate"] = stats["successes"] / stats["episodes"] if stats["episodes"] else 0.0
        stats["timeout_rate"] = stats["timeouts"] / stats["episodes"] if stats["episodes"] else 0.0
        stats["mean_episode_steps"] = mean(steps) if steps else 0.0
    return {"episodes": len(items), "by_mode": by_mode}


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)
