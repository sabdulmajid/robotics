from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from risk_aware_skill_planning.backends.openpi import StressorConfig, load_openpi_experiment_config, validate_openpi_jsonl
from risk_aware_skill_planning.backends.openpi.action_horizon import ActionHorizonPolicy
from risk_aware_skill_planning.backends.openpi.libero_runner import sbatch_environment
from risk_aware_skill_planning.evaluation.bootstrap import bootstrap_ci
from risk_aware_skill_planning.evaluation.risk_coverage import expected_utility, risk_coverage_curve
from risk_aware_skill_planning.supervision.no_progress import NoProgressWindow
from risk_aware_skill_planning.supervision.openpi_supervisor import decide_openpi_supervision


def test_openpi_named_configs_parse() -> None:
    baseline = load_openpi_experiment_config("configs/openpi/libero_collect_baseline.yaml")
    supervisor = load_openpi_experiment_config("configs/openpi/eval_supervisor.yaml")

    assert baseline.mode == "direct_openpi"
    assert baseline.libero.task_ids == (0, 1, 2)
    assert supervisor.mode == "adaptive_chunk_openpi"
    assert sbatch_environment(supervisor)["RISK_SUMMARY"] == "reports/openpi_libero_risk_summary.json"


def test_action_horizon_stressor_and_supervisor_decisions() -> None:
    policy = ActionHorizonPolicy()
    assert policy.horizon_for_risk(0.1) == 10
    assert policy.horizon_for_risk(0.5) == 5
    assert policy.horizon_for_risk(0.8) == 2
    assert policy.horizon_for_risk(0.9) == 1
    assert policy.should_abstain(0.96)

    stressor = StressorConfig("occlusion", 0.7)
    assert stressor.to_cli_args() == ["--stressor-name", "occlusion", "--stressor-severity", "0.7"]

    selective = decide_openpi_supervision(0.96, config=None)
    assert selective["should_abstain"]


def test_no_progress_window_and_risk_coverage_metrics() -> None:
    window = NoProgressWindow(patience=3, threshold=0.8)
    assert not window.update(0.7)
    assert not window.update(0.9)
    assert not window.update(0.8)
    assert window.update(0.9)

    curve = risk_coverage_curve([0, 1, 0], [0.2, 0.9, 0.4])
    assert any(row["threshold"] == 0.8 and row["coverage"] == 2 / 3 for row in curve)
    assert expected_utility(success=True, timeout=False, abstained=False, episode_length=100) > 0.0
    ci = bootstrap_ci([0, 1, 1], lambda values: sum(values) / len(values), samples=20)
    assert ci["samples"] == 20


def test_validate_openpi_jsonl_schema(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    episode = {
        "run_id": "run",
        "timestamp": "2026-05-17T00:00:00Z",
        "git_sha": "abc",
        "hostname": "host",
        "gpu_id": "0",
        "cuda_device_name": "NVIDIA RTX A4500",
        "openpi_repo_path": "external/openpi",
        "openpi_commit": "commit",
        "checkpoint_path": "gs://openpi-assets/checkpoints/pi05_libero/",
        "openpi_config_name": "pi05_libero",
        "libero_suite": "libero_spatial",
        "libero_task_id": 0,
        "libero_task_name": "task",
        "language_instruction": "pick up the bowl",
        "seed": 7,
        "episode_id": "ep0",
        "stressor_name": "none",
        "stressor_params": {"severity": 0.0},
        "n_action_steps": 5,
        "policy_backend": "openpi",
        "success": True,
        "timeout": False,
        "failure_label": "success",
        "episode_length": 1,
        "total_reward": 1.0,
        "terminal_reason": "success",
        "video_path": None,
        "metadata": {},
        "steps": [
            {
                "run_id": "run",
                "episode_id": "ep0",
                "timestep": 0,
                "observation_summary": {},
                "action_summary": {},
                "action_chunk_summary": {},
                "selected_action_index": 0,
                "n_action_steps": 5,
                "predicted_risk": 0.2,
                "calibrated_risk": 0.2,
                "supervisor_decision": None,
                "no_progress_score": 0.0,
                "reward": 1.0,
                "done": True,
                "info_summary": {},
            }
        ],
    }
    path.write_text(json.dumps(episode), encoding="utf-8")

    status = validate_openpi_jsonl(path)
    assert status["episodes"] == 1
    assert status["steps"] == 1


def test_summarize_script_writes_aggregate(tmp_path: Path) -> None:
    output = tmp_path / "aggregate.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_openpi_results.py",
            "--run-dir",
            "reports",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"]
    assert "10096" in payload["rollout_summaries"]
