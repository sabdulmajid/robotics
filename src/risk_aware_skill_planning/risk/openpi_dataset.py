from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


OPENPI_FEATURE_NAMES = (
    "bias",
    "task_id_scaled",
    "suite_hash",
    "language_hash",
    "n_action_steps_scaled",
    "stressor_severity",
    "stressor_none",
    "stressor_occlusion",
    "stressor_action_noise",
    "stressor_gaussian_noise",
    "stressor_brightness",
    "stressor_action_delay",
    "stressor_action_precision",
    "prefix_action_norm_mean",
    "prefix_action_norm_max",
    "prefix_action_smoothness_mean",
    "prefix_no_progress_mean",
    "prefix_reward_sum",
)


@dataclass(frozen=True)
class OpenPIRiskExample:
    episode_id: str
    run_id: str
    suite: str
    task_id: int
    stressor_name: str
    stressor_severity: float
    label_failure: int
    label_timeout: int
    feature_names: tuple[str, ...]
    features: tuple[float, ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "suite": self.suite,
            "task_id": self.task_id,
            "stressor_name": self.stressor_name,
            "stressor_severity": self.stressor_severity,
            "label_failure": self.label_failure,
            "label_timeout": self.label_timeout,
            "feature_names": list(self.feature_names),
            "features": list(self.features),
            "metadata": dict(self.metadata),
        }


def expand_input_paths(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted(dict.fromkeys(paths))


def load_openpi_risk_examples(paths: Sequence[str | Path], *, prefix_steps: int = 10) -> list[OpenPIRiskExample]:
    examples: list[OpenPIRiskExample] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    episode = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
                example = example_from_episode(episode, source_path=str(path), prefix_steps=prefix_steps)
                key = (example.run_id, example.episode_id)
                if key in seen:
                    continue
                seen.add(key)
                examples.append(example)
    return examples


def example_from_episode(episode: Mapping[str, Any], *, source_path: str, prefix_steps: int = 10) -> OpenPIRiskExample:
    suite = str(episode.get("libero_suite", episode.get("suite", "")))
    task_id = int(episode.get("libero_task_id", episode.get("task_id", 0)))
    language = str(episode.get("language_instruction", episode.get("task_language", "")))
    stressor_name = str(episode.get("stressor_name", episode.get("metadata", {}).get("stressor_name", "none")))
    stressor_params = episode.get("stressor_params", {})
    stressor_severity = float(stressor_params.get("severity", episode.get("metadata", {}).get("stressor_severity", 0.0)))
    steps = list(episode.get("steps", ()))[: max(1, prefix_steps)]
    action_norms = [_float(step.get("action_norm")) for step in steps]
    smoothness = [_float(step.get("action_smoothness")) for step in steps if step.get("action_smoothness") is not None]
    no_progress = [_float(step.get("no_progress_score")) for step in steps if step.get("no_progress_score") is not None]
    rewards = [_float(step.get("reward")) for step in steps]
    n_action_steps = int(episode.get("n_action_steps", episode.get("action_horizon", 0)))
    success = bool(episode.get("success", False))
    timeout = bool(episode.get("timeout", False)) or str(episode.get("failure_label", "")) == "timeout"
    feature_values = (
        1.0,
        task_id / 100.0,
        _hash_to_unit(suite),
        _hash_to_unit(language),
        n_action_steps / 20.0,
        stressor_severity,
        float(stressor_name in ("", "none", "direct")),
        float(stressor_name == "occlusion"),
        float(stressor_name == "action_noise"),
        float(stressor_name == "gaussian_noise"),
        float(stressor_name == "brightness"),
        float(stressor_name == "action_delay"),
        float(stressor_name == "action_precision"),
        _safe_mean(action_norms) / 5.0,
        (max(action_norms) if action_norms else 0.0) / 5.0,
        _safe_mean(smoothness) / 2.0,
        _safe_mean(no_progress),
        sum(rewards),
    )
    return OpenPIRiskExample(
        episode_id=str(episode.get("episode_id", "")),
        run_id=str(episode.get("run_id", episode.get("metadata", {}).get("run_id", ""))),
        suite=suite,
        task_id=task_id,
        stressor_name=stressor_name,
        stressor_severity=stressor_severity,
        label_failure=int(not success),
        label_timeout=int(timeout),
        feature_names=OPENPI_FEATURE_NAMES,
        features=tuple(float(value) for value in feature_values),
        metadata={
            "source_path": source_path,
            "terminal_label": episode.get("terminal_label", episode.get("terminal_reason")),
            "failure_label": episode.get("failure_label"),
            "episode_length": episode.get("episode_length", len(episode.get("steps", ()))) ,
            "mode": episode.get("mode"),
            "cuda_device_name": episode.get("cuda_device_name", episode.get("metadata", {}).get("cuda_device_name")),
            "video_path": episode.get("video_path", episode.get("metadata", {}).get("video_path")),
            "first_image_path": _first_image_path(episode.get("steps", ())),
        },
    )


def split_examples(
    examples: Sequence[OpenPIRiskExample],
    *,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
) -> dict[str, list[OpenPIRiskExample]]:
    splits = {"train": [], "calibration": [], "test": []}
    for label in (0, 1):
        items = sorted(
            [example for example in examples if example.label_failure == label],
            key=_stable_split_key,
        )
        n = len(items)
        if n == 0:
            continue
        if n == 1:
            splits["train"].extend(items)
            continue
        if n == 2:
            splits["train"].append(items[0])
            splits["test"].append(items[1])
            continue
        train_end = max(1, int(n * train_fraction))
        calibration_end = max(train_end + 1, train_end + int(n * calibration_fraction))
        calibration_end = min(calibration_end, n - 1)
        splits["train"].extend(items[:train_end])
        splits["calibration"].extend(items[train_end:calibration_end])
        splits["test"].extend(items[calibration_end:])
    return {name: sorted(items, key=_stable_split_key) for name, items in splits.items()}


def class_balance(examples: Sequence[OpenPIRiskExample]) -> dict[str, float | int]:
    total = len(examples)
    failures = sum(example.label_failure for example in examples)
    timeouts = sum(example.label_timeout for example in examples)
    return {
        "examples": total,
        "failures": failures,
        "failure_rate": failures / total if total else 0.0,
        "timeouts": timeouts,
        "timeout_rate": timeouts / total if total else 0.0,
    }


def _hash_to_unit(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    number = int(digest[:8], 16)
    return (number / 0xFFFFFFFF) * 2.0 - 1.0


def _stable_split_key(example: OpenPIRiskExample) -> str:
    key = f"{example.run_id}:{example.episode_id}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _safe_mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _float(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _first_image_path(steps: Iterable[Mapping[str, Any]]) -> str | None:
    for step in steps:
        image_path = step.get("image_path")
        if image_path:
            return str(image_path)
    return None
