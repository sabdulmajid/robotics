from __future__ import annotations

from pathlib import Path

from risk_aware_skill_planning.backends.openpi.config import OpenPIExperimentConfig


def single_task_eval_argv(
    config: OpenPIExperimentConfig,
    *,
    suite: str,
    task_id: int,
    stressor: str,
    jsonl_out: Path,
    video_out: Path,
) -> list[str]:
    argv = [
        "scripts/openpi_libero_single_task_eval.py",
        "--openpi-root",
        str(config.openpi.root),
        "--mode",
        config.mode,
        "--policy-config",
        config.openpi.policy_config,
        "--checkpoint",
        config.openpi.checkpoint,
        "--host",
        config.openpi.server_host,
        "--port",
        str(config.openpi.server_port),
        "--task-suite-name",
        suite,
        "--task-id",
        str(task_id),
        "--num-trials",
        str(config.libero.episodes_per_task),
        "--seed",
        str(config.libero.seed),
        "--replan-steps",
        str(config.libero.n_action_steps),
        "--stressor-name",
        stressor,
        "--stressor-severity",
        str(config.libero.stressor_severity),
        "--jsonl-out",
        str(jsonl_out),
        "--video-out-path",
        str(video_out),
    ]
    if config.risk_summary is not None:
        argv.extend(["--risk-summary", str(config.risk_summary)])
    return argv


def sbatch_environment(config: OpenPIExperimentConfig) -> dict[str, str]:
    env = {
        "MODE": config.mode,
        "SUITES": " ".join(config.libero.suites),
        "TASK_IDS": " ".join(str(task_id) for task_id in config.libero.task_ids),
        "NUM_TRIALS": str(config.libero.episodes_per_task),
        "SEED": str(config.libero.seed),
        "REPLAN_STEPS": str(config.libero.n_action_steps),
        "STRESSORS": " ".join(config.libero.stressors),
        "STRESSOR_SEVERITY": str(config.libero.stressor_severity),
    }
    if config.risk_summary is not None:
        env["RISK_SUMMARY"] = str(config.risk_summary)
    return env


def sbatch_command(config: OpenPIExperimentConfig, script: str = "slurm/openpi_libero_rollouts.sbatch") -> list[str]:
    env = sbatch_environment(config)
    return [f"{key}={value}" for key, value in env.items()] + ["sbatch", script]
