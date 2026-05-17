from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_EPISODE_FIELDS = {
    "run_id",
    "timestamp",
    "git_sha",
    "hostname",
    "gpu_id",
    "cuda_device_name",
    "openpi_repo_path",
    "openpi_commit",
    "checkpoint_path",
    "openpi_config_name",
    "libero_suite",
    "libero_task_id",
    "libero_task_name",
    "language_instruction",
    "seed",
    "episode_id",
    "stressor_name",
    "stressor_params",
    "n_action_steps",
    "policy_backend",
    "success",
    "timeout",
    "failure_label",
    "episode_length",
    "total_reward",
    "terminal_reason",
    "video_path",
    "metadata",
}

REQUIRED_STEP_FIELDS = {
    "run_id",
    "episode_id",
    "timestep",
    "observation_summary",
    "action_summary",
    "action_chunk_summary",
    "selected_action_index",
    "n_action_steps",
    "predicted_risk",
    "calibrated_risk",
    "supervisor_decision",
    "no_progress_score",
    "reward",
    "done",
    "info_summary",
}


def validate_episode_mapping(episode: Mapping[str, Any]) -> None:
    missing = REQUIRED_EPISODE_FIELDS.difference(episode)
    if missing:
        raise ValueError(f"Episode log missing required fields: {sorted(missing)}")
    if episode.get("policy_backend") != "openpi":
        raise ValueError("OpenPI rollout logs must set policy_backend='openpi'")
    if not isinstance(episode.get("steps", []), list):
        raise ValueError("Episode steps must be a list")
    for index, step in enumerate(episode.get("steps", [])):
        validate_step_mapping(step, index=index)


def validate_step_mapping(step: Mapping[str, Any], *, index: int = 0) -> None:
    missing = REQUIRED_STEP_FIELDS.difference(step)
    if missing:
        raise ValueError(f"Step {index} missing required fields: {sorted(missing)}")
    risk = step.get("predicted_risk")
    if risk is not None and not 0.0 <= float(risk) <= 1.0:
        raise ValueError(f"Step {index} predicted_risk must lie in [0, 1]")


def validate_openpi_jsonl(path: str | Path) -> dict[str, int]:
    episodes = 0
    steps = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                episode = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            validate_episode_mapping(episode)
            episodes += 1
            steps += len(episode.get("steps", []))
    return {"episodes": episodes, "steps": steps}
