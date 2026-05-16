from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from risk_aware_skill_planning.openpi_libero.schema import OpenPILiberoEpisodeLog


@dataclass(frozen=True)
class OpenPIRiskFeatureRow:
    episode_id: str
    suite: str
    task_id: int
    mode: str
    timestep: int
    timestep_fraction: float
    action_chunk_length: int
    action_norm: float
    action_smoothness: float
    action_horizon: int
    no_progress: bool
    image_embedding_path: str | None
    language_embedding_path: str | None
    world_model_progress_delta: float | None
    terminal_failure_label: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_structured_risk_rows(
    episodes: Iterable[OpenPILiberoEpisodeLog],
    *,
    image_embedding_root: str | None = None,
    language_embedding_root: str | None = None,
    include_world_model_placeholder: bool = True,
) -> list[OpenPIRiskFeatureRow]:
    rows: list[OpenPIRiskFeatureRow] = []
    for episode in episodes:
        horizon = max(1, len(episode.steps) - 1)
        terminal_failure = not episode.success
        for step in episode.steps:
            image_embedding_path = (
                f"{image_embedding_root}/{episode.episode_id}/{step.timestep:04d}.npy"
                if image_embedding_root
                else None
            )
            language_embedding_path = (
                f"{language_embedding_root}/{episode.suite}_{episode.task_id}.npy"
                if language_embedding_root
                else None
            )
            rows.append(
                OpenPIRiskFeatureRow(
                    episode_id=episode.episode_id,
                    suite=episode.suite,
                    task_id=episode.task_id,
                    mode=episode.mode,
                    timestep=step.timestep,
                    timestep_fraction=step.timestep / horizon,
                    action_chunk_length=step.action_chunk_length,
                    action_norm=step.action_norm,
                    action_smoothness=step.action_smoothness if step.action_smoothness is not None else 0.0,
                    action_horizon=step.action_horizon if step.action_horizon is not None else episode.action_horizon,
                    no_progress=bool(step.no_progress),
                    image_embedding_path=image_embedding_path,
                    language_embedding_path=language_embedding_path,
                    world_model_progress_delta=_world_model_placeholder(step.action_norm)
                    if include_world_model_placeholder
                    else None,
                    terminal_failure_label=terminal_failure,
                )
            )
    return rows


def _world_model_placeholder(action_norm: float) -> float:
    """Temporary feature slot for learned predicted progress.

    Once rollout data exists, this value should be replaced by a learned latent
    transition/world-model prediction. It is deterministic here so downstream
    risk code can be built before that model is trained.
    """

    return max(0.0, min(1.0, action_norm / 5.0))
