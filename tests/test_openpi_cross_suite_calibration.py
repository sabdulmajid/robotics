from __future__ import annotations

import importlib.util
from pathlib import Path


def test_calibration_selects_threshold_without_test_outcomes() -> None:
    module = load_script("scripts/calibrate_openpi_cross_suite_threshold.py", "cross_calibration")
    manifest = [
        {"job_id": 1, "label": "direct"},
        {"job_id": 2, "label": "siglip_score"},
    ]
    episodes = []
    for index, (success, risk) in enumerate([(True, 0.1), (True, 0.2), (False, 0.9), (False, 0.95)]):
        direct = episode("direct_openpi", index, success=success, risk=None)
        direct["_manifest"] = manifest[0]
        score = episode("vision_language_risk_selective", index, success=success, risk=risk)
        score["_manifest"] = manifest[1]
        episodes.extend([direct, score])

    summary = module.calibrate_threshold(
        manifest,
        episodes,
        [0.70, 0.75, 1.0],
        random_seeds=20,
    )

    assert summary["ok"]
    assert summary["paired_episodes"] == 4
    assert 0.9 < summary["selection"]["threshold"] < 0.95
    assert summary["selection"]["calibration_metrics"]["coverage"] == 0.75
    assert summary["risk_metrics"]["auroc"] == 1.0


def test_held_out_summary_checks_frozen_threshold_and_paired_deltas() -> None:
    module = load_script("scripts/summarize_openpi_calibrated_cross_suite.py", "cross_deployment")
    threshold = 0.91
    calibration = {"selection": {"threshold": threshold}, "job_ids": [1, 2]}
    manifest = [
        {"job_id": 3, "label": "direct", "threshold": None},
        {"job_id": 4, "label": "spatial_0986", "threshold": 0.9860334584902223},
        {"job_id": 5, "label": "cross_calibrated", "threshold": threshold},
    ]
    episodes = []
    outcomes = [True, True, False, False]
    for index, success in enumerate(outcomes):
        direct = episode("direct_openpi", index, success=success, risk=None)
        direct["_manifest"] = manifest[0]
        spatial = episode(
            "vision_language_risk_selective",
            index,
            success=success,
            risk=0.2 if success else 0.95,
            abstained=index == 3,
        )
        spatial["_manifest"] = manifest[1]
        calibrated = episode(
            "vision_language_risk_selective",
            index,
            success=success,
            risk=0.2 if success else 0.95,
            abstained=not success,
        )
        calibrated["_manifest"] = manifest[2]
        episodes.extend([direct, spatial, calibrated])

    summary = module.summarize_deployment(
        manifest,
        calibration,
        episodes,
        random_seeds=20,
    )

    assert summary["ok"]
    assert summary["episodes"] == 12
    assert summary["metrics"]["cross_calibrated"]["coverage"] == 0.5
    comparison = summary["comparisons"]["cross_calibrated_vs_direct"]
    assert comparison["attempted_failure_delta_vs_direct"] == -0.5
    assert comparison["utility_delta_ci95"]["samples"] == 1000
    assert summary["analysis_questions"]["beats_random_abstention_on_attempted_failure"]


def test_retrospective_analysis_keeps_task_splits_disjoint() -> None:
    module = load_script("scripts/analyze_openpi_cross_suite_retrospective.py", "cross_retrospective")
    manifest = [
        {"job_id": 1, "label": "direct"},
        {"job_id": 2, "label": "siglip_0986"},
    ]
    direct_manifest = {"job_id": 1, "label": "direct"}
    score_manifest = {"job_id": 2, "label": "siglip_score"}
    episodes = []
    for task_id in range(5, 10):
        for episode_index, (success, risk) in enumerate([(True, 0.1), (False, 0.95)]):
            direct = episode("direct_openpi", task_id, success=success, risk=None)
            direct["metadata"]["episode_index"] = episode_index
            direct["_manifest"] = direct_manifest
            score = episode("vision_language_risk_selective", task_id, success=success, risk=risk)
            score["metadata"]["episode_index"] = episode_index
            score["_manifest"] = score_manifest
            episodes.extend([direct, score])
    module.load_manifest_episodes = lambda _: episodes

    summary = module.analyze_retrospective(
        manifest,
        calibration_task_ids={5, 6},
        test_task_ids={7, 8, 9},
        target_coverages=[0.5, 1.0],
        random_seeds=20,
    )

    assert summary["ok"]
    assert summary["split"]["calibration_pairs"] == 4
    assert summary["split"]["test_pairs"] == 6
    assert not summary["fresh_online_deployment"]
    assert summary["held_out_test"]["selected_threshold"]["coverage"] == 0.5
    assert set(summary["held_out_test"]["per_suite"]) == {"libero_goal", "libero_object"}
    assert set(summary["risk_metrics"]["test_by_suite"]) == {"libero_goal", "libero_object"}
    assert summary["risk_metrics"]["test_by_stressor"]["none:0.00"]["examples"] == 6


def test_rollout_job_supports_node_local_staging() -> None:
    script = Path("slurm/openpi_libero_rollouts.sbatch").read_text(encoding="utf-8")
    assert "ROBOTICS_REPO_ROOT" in script
    assert script.count("python3 - \"${PORT}\"") == 2


def load_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode(
    mode: str,
    index: int,
    *,
    success: bool,
    risk: float | None,
    abstained: bool = False,
) -> dict:
    decision = None
    if risk is not None:
        decision = {
            "predicted_risk": risk,
            "threshold": 0.91,
            "prefix_steps_observed": 10,
            "risk_compute_seconds": 0.01,
        }
    terminal = "abstained" if abstained else "success" if success else "timeout"
    return {
        "mode": mode,
        "libero_suite": "libero_object" if index % 2 == 0 else "libero_goal",
        "libero_task_id": index,
        "seed": 8000,
        "stressor_name": "none",
        "stressor_params": {"severity": 0.0},
        "success": success and not abstained,
        "timeout": not success and not abstained,
        "terminal_label": terminal,
        "failure_label": terminal,
        "episode_length": 10 if abstained else 50,
        "n_action_steps": 5,
        "runtime_supervisor_decision": decision,
        "metadata": {
            "episode_index": index,
            "init_state_index": index,
            "action_queries": 2 if abstained else 10,
            "runtime_supervisor_decision": decision,
        },
    }
