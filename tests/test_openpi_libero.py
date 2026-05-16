from __future__ import annotations

from pathlib import Path

from risk_aware_skill_planning.openpi_libero.config import OpenPILiberoSmokeConfig, load_smoke_config
from risk_aware_skill_planning.openpi_libero.features import extract_structured_risk_rows
from risk_aware_skill_planning.openpi_libero.schema import (
    OpenPILiberoEpisodeLog,
    OpenPILiberoStepLog,
    read_episode_jsonl,
    summarize_episode_logs,
    write_episode_jsonl,
)
from risk_aware_skill_planning.openpi_libero.smoke import run_openpi_libero_smoke
from risk_aware_skill_planning.openpi_libero.supervisor import RiskSupervisorConfig, decide_supervisor_action


def test_rollout_schema_roundtrip_and_summary(tmp_path: Path) -> None:
    episode = OpenPILiberoEpisodeLog(
        episode_id="libero_spatial_000",
        suite="libero_spatial",
        task_id=0,
        task_language="put the object on the target",
        seed=7,
        mode="direct_openpi",
        policy_config="pi05_libero",
        checkpoint="gs://openpi-assets/checkpoints/pi05_libero/",
        action_horizon=10,
        terminal_label="success",
        success=True,
        timeout=False,
        steps=(
            OpenPILiberoStepLog(
                timestep=0,
                observation_keys=("agentview_image", "robot0_eef_pos"),
                action_chunk_length=10,
                action_norm=1.25,
                predicted_risk=0.12,
                action_horizon=10,
                reward=0.0,
            ),
            OpenPILiberoStepLog(
                timestep=1,
                observation_keys=("agentview_image",),
                action_chunk_length=10,
                action_norm=0.75,
                predicted_risk=0.08,
                action_horizon=10,
                reward=1.0,
                done=True,
                success=True,
            ),
        ),
    )
    path = tmp_path / "episodes.jsonl"
    write_episode_jsonl([episode], path)
    loaded = read_episode_jsonl(path)
    assert loaded == [episode]
    summary = summarize_episode_logs(loaded)
    assert summary["episodes"] == 1
    assert summary["by_mode"]["direct_openpi"]["success_rate"] == 1.0
    assert summary["by_mode"]["direct_openpi"]["mean_episode_steps"] == 2


def test_supervisor_decision_thresholds() -> None:
    config = RiskSupervisorConfig()
    assert decide_supervisor_action(0.10, config=config).action == "direct_openpi"
    assert decide_supervisor_action(0.40, config=config).action_horizon == 5
    assert decide_supervisor_action(0.70, config=config).action_horizon == 2
    assert decide_supervisor_action(0.90, config=config).action_horizon == 1
    abstain = decide_supervisor_action(0.97, config=config)
    assert abstain.should_abstain
    replan = decide_supervisor_action(0.80, no_progress_steps=8, config=config)
    assert replan.should_replan
    assert replan.action == "no_progress_replan"


def test_structured_risk_features_include_vlm_and_world_model_slots() -> None:
    episode = OpenPILiberoEpisodeLog(
        episode_id="ep0",
        suite="libero_goal",
        task_id=2,
        task_language="open the drawer",
        seed=3,
        mode="adaptive_chunk_openpi",
        policy_config="pi05_libero",
        checkpoint="gs://openpi-assets/checkpoints/pi05_libero/",
        action_horizon=10,
        terminal_label="timeout",
        success=False,
        timeout=True,
        steps=(
            OpenPILiberoStepLog(
                timestep=0,
                observation_keys=("agentview_image",),
                action_chunk_length=10,
                action_norm=2.5,
                action_horizon=5,
                no_progress=True,
            ),
        ),
    )
    rows = extract_structured_risk_rows(
        [episode],
        image_embedding_root="datasets/openpi_libero_embeddings/images",
        language_embedding_root="datasets/openpi_libero_embeddings/language",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.terminal_failure_label
    assert row.image_embedding_path == "datasets/openpi_libero_embeddings/images/ep0/0000.npy"
    assert row.language_embedding_path == "datasets/openpi_libero_embeddings/language/libero_goal_2.npy"
    assert row.world_model_progress_delta == 0.5


def test_openpi_smoke_reports_missing_external_root(tmp_path: Path) -> None:
    config = OpenPILiberoSmokeConfig(
        experiment_id="test",
        openpi_repo_url="https://github.com/Physical-Intelligence/openpi.git",
        openpi_root=tmp_path / "missing-openpi",
        policy_config="pi05_libero",
        checkpoint="gs://openpi-assets/checkpoints/pi05_libero/",
        suite="libero_spatial",
        task_id=0,
        episodes=1,
        seed=0,
        default_action_horizon=10,
        server_command=("uv", "run", "scripts/serve_policy.py", "--env", "LIBERO"),
        client_command=("python", "examples/libero/main.py"),
        status_path=tmp_path / "status.json",
        report_path=tmp_path / "report.md",
    )
    status = run_openpi_libero_smoke(config)
    assert not status["ok"]
    assert any("OpenPI root does not exist" in blocker for blocker in status["blockers"])
    assert any("git clone --recurse-submodules" in command for command in status["resume_commands"])


def test_load_openpi_smoke_config() -> None:
    config = load_smoke_config("configs/openpi_libero_smoke.yaml")
    assert config.policy_config == "pi05_libero"
    assert config.checkpoint == "gs://openpi-assets/checkpoints/pi05_libero/"
    assert config.suite == "libero_spatial"
