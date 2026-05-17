from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from risk_aware_skill_planning.risk.openpi_dataset import OpenPIRiskExample


VISION_FEATURE_PREFIX = "siglip_image_"


def vision_feature_names(dims: int) -> tuple[str, ...]:
    if dims <= 0:
        raise ValueError("Vision embedding dimension must be positive")
    return tuple(f"{VISION_FEATURE_PREFIX}{idx:03d}" for idx in range(dims))


def has_vision_features(feature_names: Sequence[str]) -> bool:
    return any(name.startswith(VISION_FEATURE_PREFIX) for name in feature_names)


def load_vision_embedding_map(path: str | Path, *, dims: int | None = None) -> dict[tuple[str, str], tuple[float, ...]]:
    embeddings: dict[tuple[str, str], tuple[float, ...]] = {}
    embedding_path = Path(path)
    with embedding_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            vector = row.get("embedding")
            if not isinstance(vector, list):
                raise ValueError(f"Embedding row {line_number} in {embedding_path} has no embedding list")
            values = tuple(float(value) for value in vector)
            if dims is not None:
                if len(values) < dims:
                    raise ValueError(
                        f"Embedding row {line_number} in {embedding_path} has {len(values)} dims, "
                        f"but {dims} were requested"
                    )
                values = values[:dims]
            key = (str(row.get("run_id", "")), str(row.get("episode_id", "")))
            if not all(key):
                raise ValueError(f"Embedding row {line_number} in {embedding_path} is missing run_id or episode_id")
            embeddings[key] = values
    return embeddings


def append_vision_embeddings(
    examples: Sequence[OpenPIRiskExample],
    embeddings: Mapping[tuple[str, str], Sequence[float]],
) -> tuple[list[OpenPIRiskExample], list[tuple[str, str]]]:
    output: list[OpenPIRiskExample] = []
    missing: list[tuple[str, str]] = []
    dims: int | None = None
    names: tuple[str, ...] | None = None
    for example in examples:
        key = (example.run_id, example.episode_id)
        vector = embeddings.get(key)
        if vector is None:
            missing.append(key)
            continue
        values = tuple(float(value) for value in vector)
        if dims is None:
            dims = len(values)
            names = vision_feature_names(dims)
        elif len(values) != dims:
            raise ValueError(f"Vision embedding dimension mismatch for {key}: {len(values)} != {dims}")
        metadata = dict(example.metadata)
        metadata["vision_embedding_available"] = True
        output.append(
            replace(
                example,
                feature_names=example.feature_names + tuple(names or ()),
                features=example.features + values,
                metadata=metadata,
            )
        )
    return output, missing

