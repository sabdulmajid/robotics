from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_OUTPUT_ROOTS = ("outputs/", "checkpoints/", "datasets/", "videos/", "reports/figures/")


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    validate_experiment_config(data)
    return data


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config key {key!r} must be a mapping")
    return value


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    required = ["experiment_id", "num_episodes", "scenario_ids", "planner_modes", "planner", "outputs"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    if not isinstance(config["num_episodes"], int) or config["num_episodes"] < 0:
        raise ValueError("num_episodes must be a non-negative integer")
    if not config["scenario_ids"]:
        raise ValueError("scenario_ids must be non-empty")
    if not config["planner_modes"]:
        raise ValueError("planner_modes must be non-empty")
    planner = _require_mapping(config, "planner")
    for key in ["lambda_risk", "next_skill_threshold", "plan_threshold", "max_replans"]:
        if key not in planner:
            raise ValueError(f"planner.{key} is required")
    outputs = _require_mapping(config, "outputs")
    summary_path = outputs.get("summary_path")
    if not isinstance(summary_path, str):
        raise ValueError("outputs.summary_path must be a string")
    if not summary_path.startswith(REPO_OUTPUT_ROOTS):
        raise ValueError(f"outputs.summary_path must be under one of {REPO_OUTPUT_ROOTS}")

