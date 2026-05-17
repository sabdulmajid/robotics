from __future__ import annotations

import json
from pathlib import Path

from risk_aware_skill_planning.evaluation.openpi_risk import run_openpi_risk_training
from risk_aware_skill_planning.evaluation.openpi_metrics import auprc
from risk_aware_skill_planning.risk.openpi_dataset import load_openpi_risk_examples


def test_openpi_risk_dataset_and_training_on_synthetic_logs(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.jsonl"
    episodes = [
        _episode("ep_success_0", success=True, stressor="none", severity=0.0, action_norm=0.8),
        _episode("ep_success_1", success=True, stressor="none", severity=0.0, action_norm=1.0),
        _episode("ep_fail_0", success=False, stressor="occlusion", severity=0.8, action_norm=2.8),
        _episode("ep_fail_1", success=False, stressor="action_noise", severity=0.8, action_norm=3.1),
        _episode("ep_success_2", success=True, stressor="none", severity=0.0, action_norm=1.2),
        _episode("ep_fail_2", success=False, stressor="occlusion", severity=0.9, action_norm=3.0),
    ]
    path.write_text("\n".join(json.dumps(ep) for ep in episodes), encoding="utf-8")

    examples = load_openpi_risk_examples([path])
    assert len(examples) == 6
    assert examples[0].feature_names
    assert {example.label_failure for example in examples} == {0, 1}

    summary = run_openpi_risk_training([str(path)])
    assert summary["ok"]
    assert summary["metrics"]["test"]["examples"] >= 1
    assert "temperature" in summary["calibration"]
    assert "normalization" in summary
    assert set(summary["normalization"]) == {"mean", "std"}


def test_openpi_risk_loader_deduplicates_combined_and_per_task_logs(tmp_path: Path) -> None:
    combined = tmp_path / "combined.jsonl"
    task = tmp_path / "task.jsonl"
    episode = _episode("ep_success_0", success=True, stressor="none", severity=0.0, action_norm=0.8)
    combined.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    task.write_text(json.dumps(episode) + "\n", encoding="utf-8")

    examples = load_openpi_risk_examples([combined, task])

    assert len(examples) == 1
    assert examples[0].episode_id == "ep_success_0"


def test_auprc_handles_tied_scores_without_positive_first_bias() -> None:
    assert auprc([0, 1, 0], [0.5, 0.5, 0.5]) == 1 / 3


def _episode(episode_id: str, *, success: bool, stressor: str, severity: float, action_norm: float) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "run_id": "synthetic",
        "libero_suite": "libero_spatial",
        "libero_task_id": 0,
        "language_instruction": "pick up the bowl",
        "success": success,
        "timeout": not success,
        "failure_label": "success" if success else "timeout",
        "stressor_name": stressor,
        "stressor_params": {"severity": severity},
        "n_action_steps": 5,
        "steps": [
            {
                "timestep": idx,
                "action_norm": action_norm + 0.1 * idx,
                "action_smoothness": 0.2,
                "no_progress_score": 0.1 if success else 0.9,
                "reward": 1.0 if success and idx == 2 else 0.0,
            }
            for idx in range(3)
        ],
    }
