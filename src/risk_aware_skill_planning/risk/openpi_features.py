from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from risk_aware_skill_planning.risk.openpi_dataset import OPENPI_FEATURE_NAMES, example_from_episode


@dataclass(frozen=True)
class OpenPIFeatureRow:
    feature_names: tuple[str, ...]
    features: tuple[float, ...]
    image_embedding_path: str | None = None
    language_embedding_path: str | None = None
    openpi_embedding_path: str | None = None


def extract_openpi_feature_row(
    episode: Mapping[str, Any],
    *,
    prefix_steps: int = 10,
    image_embedding_root: str | None = None,
    language_embedding_root: str | None = None,
    openpi_embedding_root: str | None = None,
) -> OpenPIFeatureRow:
    example = example_from_episode(episode, source_path="<memory>", prefix_steps=prefix_steps)
    episode_id = str(episode.get("episode_id", "episode"))
    suite = str(episode.get("libero_suite", episode.get("suite", "")))
    task_id = int(episode.get("libero_task_id", episode.get("task_id", 0)))
    return OpenPIFeatureRow(
        feature_names=tuple(example.feature_names),
        features=tuple(example.features),
        image_embedding_path=f"{image_embedding_root}/{episode_id}/0000.npy" if image_embedding_root else None,
        language_embedding_path=f"{language_embedding_root}/{suite}_{task_id}.npy" if language_embedding_root else None,
        openpi_embedding_path=f"{openpi_embedding_root}/{episode_id}.npy" if openpi_embedding_root else None,
    )


def feature_index(names: Sequence[str] = OPENPI_FEATURE_NAMES) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(names)}
