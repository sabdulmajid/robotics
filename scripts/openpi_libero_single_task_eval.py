#!/usr/bin/env python
"""Run a small OpenPI/LIBERO evaluation and emit repository JSONL logs.

This script is intentionally Python 3.8-compatible because OpenPI's LIBERO
client environment is separate from this repository's Python package runtime.
It mirrors OpenPI's upstream LIBERO evaluator, but adds task filtering and a
stable JSONL contract for risk-model training.
"""

from __future__ import annotations

import argparse
import collections
import datetime as _datetime
import json
import logging
import math
import os
import pathlib
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
DEFAULT_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
RUNTIME_FEATURE_NAMES = (
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
VISION_SELECTIVE_MODE = "vision_language_risk_selective"
FIXED_PRIOR_SELECTIVE_MODE = "fixed_task_prior_selective"
SIGLIP_FEATURE_PREFIX = "siglip_image_"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    openpi_root = pathlib.Path(args.openpi_root).expanduser().resolve()
    libero_config_path = pathlib.Path(args.libero_config_path).expanduser().resolve()
    prepare_libero_runtime(openpi_root, libero_config_path)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "openpi_root": str(openpi_root),
                    "libero_config_path": str(libero_config_path),
                    "task_suite_name": args.task_suite_name,
                    "task_id": args.task_id,
                    "num_trials": args.num_trials,
                    "jsonl_out": args.jsonl_out,
                    "video_out_path": args.video_out_path,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    return run_eval(args, openpi_root)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one OpenPI/LIBERO task and write risk-rollout JSONL")
    parser.add_argument("--openpi-root", default="external/openpi")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--run-id")
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video-out-path", default="videos/openpi_libero_smoke")
    parser.add_argument("--jsonl-out", default="datasets/openpi_libero_rollouts/openpi_libero_smoke.jsonl")
    parser.add_argument("--mode", default="direct_openpi")
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero/")
    parser.add_argument("--risk-summary")
    parser.add_argument("--runtime-risk-prefix-steps", type=int, default=10)
    parser.add_argument("--siglip-model", default="google/siglip-base-patch16-224")
    parser.add_argument("--siglip-dims", type=int, default=64)
    parser.add_argument("--runtime-vision-device", default="cpu", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--low-risk-threshold", type=float, default=0.35)
    parser.add_argument("--medium-risk-threshold", type=float, default=0.65)
    parser.add_argument("--high-risk-threshold", type=float, default=0.85)
    parser.add_argument("--abstain-threshold", type=float, default=0.95)
    parser.add_argument("--stressor-name", default="none")
    parser.add_argument("--stressor-severity", type=float, default=0.0)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--libero-config-path", default="outputs/openpi_libero/libero_config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args(argv)


def prepare_libero_runtime(openpi_root: pathlib.Path, libero_config_path: pathlib.Path) -> None:
    libero_source = openpi_root / "third_party" / "libero"
    if str(libero_source) not in sys.path:
        sys.path.insert(0, str(libero_source))
    os.environ["LIBERO_CONFIG_PATH"] = str(libero_config_path)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    write_libero_config(openpi_root, libero_config_path)


def write_libero_config(openpi_root: pathlib.Path, libero_config_path: pathlib.Path) -> pathlib.Path:
    benchmark_root = openpi_root / "third_party" / "libero" / "libero" / "libero"
    config_file = libero_config_path / "config.yaml"
    libero_config_path.mkdir(parents=True, exist_ok=True)
    datasets = openpi_root / "third_party" / "libero" / "libero" / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    payload = {
        "assets": benchmark_root / "assets",
        "bddl_files": benchmark_root / "bddl_files",
        "benchmark_root": benchmark_root,
        "datasets": datasets,
        "init_states": benchmark_root / "init_files",
    }
    with config_file.open("w", encoding="utf-8") as handle:
        for key in sorted(payload):
            handle.write("{0}: {1}\n".format(key, payload[key]))
    return config_file


def run_eval(args: argparse.Namespace, openpi_root: pathlib.Path) -> int:
    # Optional imports are kept inside the runtime path so repository tests do
    # not require OpenPI, robosuite, MuJoCo, or LIBERO.
    import imageio  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from libero.libero import benchmark  # type: ignore
    from libero.libero import get_libero_path  # type: ignore
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore
    from openpi_client import image_tools  # type: ignore
    from openpi_client import websocket_client_policy  # type: ignore

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise ValueError("Unknown LIBERO suite: {0}".format(args.task_suite_name))
    np.random.seed(args.seed)
    max_steps = args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS[args.task_suite_name]
    task_suite = benchmark_dict[args.task_suite_name]()
    if args.task_id < 0 or args.task_id >= task_suite.n_tasks:
        raise ValueError("task_id {0} is outside suite range 0..{1}".format(args.task_id, task_suite.n_tasks - 1))

    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    if len(initial_states) == 0:
        raise RuntimeError("LIBERO returned no initial states for task {0}".format(args.task_id))

    video_dir = pathlib.Path(args.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = pathlib.Path(args.jsonl_out)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    task_description = str(task.language)
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(args.seed)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    run_metadata = build_run_metadata(args, openpi_root, task, torch)
    runtime_vision_encoder = load_runtime_vision_encoder(args) if args.mode == VISION_SELECTIVE_MODE else None

    episodes: List[Dict[str, Any]] = []
    successes = 0
    start_time = time.time()
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        try:
            for episode_idx in range(args.num_trials):
                init_state_index = episode_idx % len(initial_states)
                episode = run_episode(
                    args=args,
                    env=env,
                    client=client,
                    image_tools=image_tools,
                    np_module=np,
                    imageio_module=imageio,
                    task_description=task_description,
                    initial_state=initial_states[init_state_index],
                    episode_idx=episode_idx,
                    init_state_index=init_state_index,
                    max_steps=max_steps,
                    video_dir=video_dir,
                    run_metadata=run_metadata,
                    runtime_vision_encoder=runtime_vision_encoder,
                )
                if episode["success"]:
                    successes += 1
                jsonl.write(json.dumps(episode, sort_keys=True) + "\n")
                jsonl.flush()
                episodes.append(episode)
        finally:
            if hasattr(env, "close"):
                env.close()

    summary = {
        "ok": True,
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": float(successes) / float(len(episodes)) if episodes else 0.0,
        "jsonl_out": str(jsonl_path),
        "video_out_path": str(video_dir),
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "task_language": task_description,
        "elapsed_seconds": time.time() - start_time,
        "openpi_root": str(openpi_root),
        "run_id": run_metadata["run_id"],
        "stressor_name": args.stressor_name,
        "stressor_severity": args.stressor_severity,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_episode(
    args: argparse.Namespace,
    env: Any,
    client: Any,
    image_tools: Any,
    np_module: Any,
    imageio_module: Any,
    task_description: str,
    initial_state: Any,
    episode_idx: int,
    init_state_index: int,
    max_steps: int,
    video_dir: pathlib.Path,
    run_metadata: Dict[str, Any],
    runtime_vision_encoder: Any = None,
) -> Dict[str, Any]:
    obs = env.reset()
    action_plan: collections.deque = collections.deque()
    obs = env.set_init_state(initial_state)

    t = 0
    done = False
    reward = 0.0
    exception_text = None
    replay_images = []
    step_logs: List[Dict[str, Any]] = []
    previous_action = None
    action_chunk_length = 0
    action_queries = 0
    total_reward = 0.0
    previous_obs = None
    delayed_action = None
    risk_summary = load_risk_summary(args.risk_summary)
    current_action_horizon = int(args.replan_steps)
    latest_predicted_risk = None
    latest_supervisor_decision = None
    runtime_supervisor_decision = None
    runtime_vision_embedding = None
    runtime_vision_frame_id = None
    runtime_vision_frame_path = None
    terminal_label = None
    failure_label = None
    episode_id = "{0}_task{1:02d}_seed{2}_ep{3:03d}".format(
        args.task_suite_name, args.task_id, args.seed, episode_idx
    )
    frame_dir = video_dir / "frames" / episode_id
    if args.save_images:
        frame_dir.mkdir(parents=True, exist_ok=True)

    while t < max_steps + args.num_steps_wait:
        try:
            if t < args.num_steps_wait:
                obs, reward, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np_module.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np_module.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )
            img = apply_image_stressor(img, args.stressor_name, args.stressor_severity, np_module, t)
            wrist_img = apply_image_stressor(wrist_img, args.stressor_name, args.stressor_severity, np_module, t + 17)
            replay_images.append(img)
            rollout_timestep = int(t - args.num_steps_wait)

            if args.mode == VISION_SELECTIVE_MODE and runtime_vision_embedding is None:
                if runtime_vision_encoder is None:
                    raise RuntimeError("Runtime SigLIP encoder was not initialized")
                runtime_vision_embedding = runtime_vision_encoder.embed(img)
                runtime_vision_frame_id = "{0:04d}".format(rollout_timestep)
                if args.save_images:
                    frame_dir.mkdir(parents=True, exist_ok=True)
                    runtime_vision_frame_path = frame_dir / "risk_initial_{0}.png".format(runtime_vision_frame_id)
                    imageio_module.imwrite(str(runtime_vision_frame_path), img)

            if not action_plan:
                should_delay_vision_decision = (
                    args.mode == VISION_SELECTIVE_MODE
                    and runtime_supervisor_decision is None
                    and len(step_logs) < max(0, int(args.runtime_risk_prefix_steps))
                )
                if should_delay_vision_decision:
                    latest_predicted_risk = None
                    latest_supervisor_decision = {
                        "action": VISION_SELECTIVE_MODE,
                        "n_action_steps": int(args.replan_steps),
                        "reason": "collecting_early_prefix_before_runtime_risk",
                        "predicted_risk": None,
                        "threshold": runtime_abstain_threshold(args, risk_summary, None),
                        "frame_id": runtime_vision_frame_id,
                    }
                elif runtime_supervisor_decision is not None and runtime_supervisor_decision.get("action") != "abstain":
                    latest_predicted_risk = runtime_supervisor_decision.get("predicted_risk")
                    latest_supervisor_decision = {
                        "action": args.mode,
                        "n_action_steps": int(args.replan_steps),
                        "reason": "accepted_after_initial_runtime_risk_check",
                        "predicted_risk": latest_predicted_risk,
                        "threshold": runtime_supervisor_decision.get("threshold"),
                        "frame_id": runtime_supervisor_decision.get("frame_id"),
                    }
                else:
                    risk_compute_start = time.time()
                    latest_predicted_risk = predict_runtime_risk(
                        risk_summary,
                        args=args,
                        task_description=task_description,
                        step_logs=step_logs,
                        vision_embedding=runtime_vision_embedding,
                    )
                    risk_compute_seconds = time.time() - risk_compute_start
                    latest_supervisor_decision = decide_runtime_supervisor(
                        args,
                        latest_predicted_risk,
                        risk_summary=risk_summary,
                    )
                    if args.mode in (VISION_SELECTIVE_MODE, FIXED_PRIOR_SELECTIVE_MODE):
                        latest_supervisor_decision["frame_id"] = runtime_vision_frame_id
                        latest_supervisor_decision["frame_path"] = (
                            str(runtime_vision_frame_path) if runtime_vision_frame_path is not None else None
                        )
                        latest_supervisor_decision["prefix_steps_observed"] = len(step_logs)
                        latest_supervisor_decision["risk_compute_seconds"] = risk_compute_seconds
                        runtime_supervisor_decision = dict(latest_supervisor_decision)
                if latest_supervisor_decision["action"] == "abstain":
                    terminal_label = "abstained"
                    failure_label = "abstained"
                    break
                current_action_horizon = int(latest_supervisor_decision["n_action_steps"])
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np_module.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"], np_module),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }
                action_chunk = client.infer(element)["actions"]
                action_chunk_length = len(action_chunk)
                if action_chunk_length < current_action_horizon:
                    raise RuntimeError(
                        "Policy predicted {0} actions, but requested horizon={1}".format(
                            action_chunk_length, current_action_horizon
                        )
                    )
                action_plan.extend(action_chunk[:current_action_horizon])
                action_queries += 1

            selected_action_index = int(current_action_horizon - len(action_plan))
            action = np_module.asarray(action_plan.popleft())
            action, delayed_action = apply_action_stressor(
                action,
                delayed_action,
                args.stressor_name,
                args.stressor_severity,
                np_module,
                t,
            )
            obs, reward, done, _ = env.step(action.tolist())
            total_reward += float(reward)
            smoothness = None
            if previous_action is not None:
                smoothness = float(np_module.linalg.norm(action - previous_action))
            no_progress_score = compute_no_progress_score(previous_obs, obs, np_module)
            previous_obs = obs
            previous_action = action
            image_path = None
            if args.save_images:
                image_path = frame_dir / "{0:04d}.png".format(rollout_timestep)
                imageio_module.imwrite(str(image_path), img)

            step_logs.append(
                {
                    "run_id": run_metadata["run_id"],
                    "episode_id": episode_id,
                    "timestep": rollout_timestep,
                    "observation_keys": sorted(str(key) for key in obs.keys()),
                    "observation_summary": summarize_observation(obs, np_module),
                    "image_path": str(image_path) if image_path is not None else None,
                    "image_id": "{0:04d}".format(rollout_timestep),
                    "action_chunk_length": int(action_chunk_length),
                    "action_norm": float(np_module.linalg.norm(action)),
                    "action_smoothness": smoothness,
                    "action_summary": summarize_action(action, np_module),
                    "action_chunk_summary": {
                        "length": int(action_chunk_length),
                        "policy_queries": int(action_queries),
                    },
                    "selected_action_index": selected_action_index,
                    "n_action_steps": int(current_action_horizon),
                    "predicted_risk": latest_predicted_risk,
                    "calibrated_risk": latest_predicted_risk,
                    "action_horizon": int(current_action_horizon),
                    "supervisor_decision": latest_supervisor_decision,
                    "no_progress_score": no_progress_score,
                    "reward": float(reward),
                    "done": bool(done),
                    "success": bool(done),
                    "no_progress": no_progress_score is not None and no_progress_score > 0.8,
                    "info_summary": {},
                }
            )
            t += 1
            if done:
                break
        except Exception as exc:  # pragma: no cover - external simulator boundary
            exception_text = repr(exc)
            logging.exception("Episode %s failed at simulator timestep %s", episode_idx, t)
            break

    if terminal_label is None:
        terminal_label = "success" if done else "runtime_error" if exception_text else "timeout"
    if failure_label is None:
        failure_label = "success" if done else "policy_invalid_action" if exception_text else "timeout"
    video_path = None
    if replay_images:
        suffix = "success" if done else "failure"
        video_path = video_dir / "rollout_{0}_task{1:02d}_ep{2:03d}_{3}.mp4".format(
            args.task_suite_name, args.task_id, episode_idx, suffix
        )
        imageio_module.mimwrite(str(video_path), [np_module.asarray(x) for x in replay_images], fps=10)

    metadata = dict(run_metadata)
    metadata.update(
        {
            "host": args.host,
            "port": int(args.port),
            "resize_size": int(args.resize_size),
            "num_steps_wait": int(args.num_steps_wait),
            "max_steps": int(max_steps),
            "episode_index": int(episode_idx),
            "init_state_index": int(init_state_index),
            "action_queries": int(action_queries),
            "risk_summary": args.risk_summary,
            "exception": exception_text,
            "video_path": str(video_path) if video_path is not None else None,
            "runtime_supervisor_decision": runtime_supervisor_decision,
            "runtime_vision_frame_id": runtime_vision_frame_id,
            "runtime_vision_frame_path": str(runtime_vision_frame_path) if runtime_vision_frame_path is not None else None,
        }
    )
    return {
        "run_id": run_metadata["run_id"],
        "timestamp": run_metadata["timestamp"],
        "git_sha": run_metadata["git_sha"],
        "hostname": run_metadata["hostname"],
        "gpu_id": run_metadata["gpu_id"],
        "cuda_device_name": run_metadata["cuda_device_name"],
        "openpi_repo_path": run_metadata["openpi_repo_path"],
        "openpi_commit": run_metadata["openpi_commit"],
        "checkpoint_path": args.checkpoint,
        "openpi_config_name": args.policy_config,
        "libero_suite": args.task_suite_name,
        "libero_task_id": int(args.task_id),
        "libero_task_name": run_metadata["libero_task_name"],
        "language_instruction": task_description,
        "stressor_name": args.stressor_name,
        "stressor_params": {
            "name": args.stressor_name,
            "severity": float(args.stressor_severity),
        },
        "n_action_steps": int(args.replan_steps),
        "policy_backend": "openpi",
        "failure_label": failure_label,
        "episode_length": len(step_logs),
        "total_reward": total_reward,
        "terminal_reason": terminal_label,
        "video_path": str(video_path) if video_path is not None else None,
        "episode_id": episode_id,
        "suite": args.task_suite_name,
        "task_id": int(args.task_id),
        "task_language": task_description,
        "seed": int(args.seed),
        "mode": args.mode,
        "policy_config": args.policy_config,
        "checkpoint": args.checkpoint,
        "action_horizon": int(args.replan_steps),
        "terminal_label": terminal_label,
        "success": bool(done),
        "timeout": bool(terminal_label == "timeout"),
        "runtime_supervisor_decision": runtime_supervisor_decision,
        "steps": step_logs,
        "metadata": metadata,
    }


def quat2axisangle(quat: Any, np_module: Any) -> Any:
    quat = np_module.asarray(quat).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np_module.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np_module.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def build_run_metadata(args: argparse.Namespace, openpi_root: pathlib.Path, task: Any, torch_module: Any) -> Dict[str, Any]:
    run_id = args.run_id or "{0}_{1}_task{2:02d}_{3}".format(
        args.mode,
        args.task_suite_name,
        args.task_id,
        _datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
    )
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cuda_device_name = None
    if torch_module.cuda.is_available():
        cuda_device_name = torch_module.cuda.get_device_name(0)
    return {
        "run_id": run_id,
        "timestamp": _datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_sha": git_rev_parse(pathlib.Path.cwd()),
        "hostname": socket.gethostname(),
        "gpu_id": gpu_id,
        "cuda_device_name": cuda_device_name,
        "openpi_repo_path": str(openpi_root),
        "openpi_commit": git_rev_parse(openpi_root),
        "libero_task_name": str(getattr(task, "bddl_file", "")),
    }


def git_rev_parse(path: pathlib.Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def apply_image_stressor(image: Any, name: str, severity: float, np_module: Any, salt: int) -> Any:
    if severity <= 0.0 or name in ("", "none", "direct"):
        return image
    sev = max(0.0, min(1.0, float(severity)))
    if name == "brightness":
        return np_module.clip(image.astype(np_module.float32) * (1.0 - 0.7 * sev), 0, 255).astype(np_module.uint8)
    if name == "gaussian_noise":
        rng = np_module.random.default_rng(int(7919 + salt))
        noise = rng.normal(0.0, 55.0 * sev, size=image.shape)
        return np_module.clip(image.astype(np_module.float32) + noise, 0, 255).astype(np_module.uint8)
    if name == "occlusion":
        out = image.copy()
        height, width = out.shape[:2]
        box_h = max(1, int(height * (0.15 + 0.45 * sev)))
        box_w = max(1, int(width * (0.15 + 0.45 * sev)))
        y0 = max(0, (height - box_h) // 2)
        x0 = max(0, (width - box_w) // 2)
        out[y0 : y0 + box_h, x0 : x0 + box_w] = 0
        return out
    return image


def apply_action_stressor(
    action: Any,
    delayed_action: Any,
    name: str,
    severity: float,
    np_module: Any,
    timestep: int,
) -> Any:
    if severity <= 0.0 or name in ("", "none", "direct"):
        return action, action.copy()
    sev = max(0.0, min(1.0, float(severity)))
    if name == "action_noise":
        rng = np_module.random.default_rng(int(104729 + timestep))
        noise = rng.normal(0.0, 0.2 * sev, size=action.shape)
        return action + noise, action.copy()
    if name == "action_delay" and delayed_action is not None:
        use_previous = (timestep % max(2, int(6 - 4 * sev))) == 0
        return (delayed_action.copy() if use_previous else action), action.copy()
    if name == "action_precision":
        decimals = 1 if sev > 0.5 else 2
        return np_module.round(action, decimals=decimals), action.copy()
    return action, action.copy()


def summarize_observation(obs: Dict[str, Any], np_module: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"keys": sorted(str(key) for key in obs.keys())}
    for key in ("robot0_eef_pos", "robot0_gripper_qpos", "object-state"):
        if key in obs:
            value = np_module.asarray(obs[key])
            summary[key] = {
                "shape": list(value.shape),
                "mean": float(value.mean()) if value.size else 0.0,
                "std": float(value.std()) if value.size else 0.0,
            }
    return summary


def summarize_action(action: Any, np_module: Any) -> Dict[str, Any]:
    value = np_module.asarray(action)
    return {
        "shape": list(value.shape),
        "mean": float(value.mean()) if value.size else 0.0,
        "std": float(value.std()) if value.size else 0.0,
        "norm": float(np_module.linalg.norm(value)),
        "min": float(value.min()) if value.size else 0.0,
        "max": float(value.max()) if value.size else 0.0,
    }


def compute_no_progress_score(previous_obs: Any, obs: Dict[str, Any], np_module: Any) -> Optional[float]:
    if previous_obs is None or "robot0_eef_pos" not in previous_obs or "robot0_eef_pos" not in obs:
        return None
    delta = float(np_module.linalg.norm(np_module.asarray(obs["robot0_eef_pos"]) - np_module.asarray(previous_obs["robot0_eef_pos"])))
    return max(0.0, min(1.0, 1.0 - delta / 0.01))


def load_risk_summary(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with pathlib.Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def predict_runtime_risk(
    risk_summary: Optional[Dict[str, Any]],
    *,
    args: argparse.Namespace,
    task_description: str,
    step_logs: List[Dict[str, Any]],
    vision_embedding: Optional[Sequence[float]] = None,
) -> Optional[float]:
    if not risk_summary or not risk_summary.get("ok") or "normalization" not in risk_summary:
        return None
    if args.mode == FIXED_PRIOR_SELECTIVE_MODE:
        return predict_fixed_task_prior(risk_summary, args)
    risk_payload = runtime_risk_payload(risk_summary, args)
    if risk_payload is None:
        return None
    features = dict(zip(RUNTIME_FEATURE_NAMES, runtime_feature_vector(args, task_description, step_logs)))
    if args.mode == VISION_SELECTIVE_MODE:
        if vision_embedding is None:
            return None
        for idx, value in enumerate(vision_embedding):
            features["{0}{1:03d}".format(SIGLIP_FEATURE_PREFIX, idx)] = float(value)
    feature_names = risk_payload["feature_names"]
    weights = risk_payload["weights"]
    mean = risk_payload["normalization"]["mean"]
    std = risk_payload["normalization"]["std"]
    temperature = max(float(risk_payload["calibration"]["temperature"]), 1e-6)
    logit = 0.0
    for name in feature_names:
        if name not in features:
            return None
        value = features[name]
        sigma = max(float(std[name]), 1e-6)
        logit += float(weights[name]) * ((float(value) - float(mean[name])) / sigma)
    return sigmoid(logit / temperature)


def runtime_risk_payload(risk_summary: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if args.mode == VISION_SELECTIVE_MODE:
        payload = risk_summary.get("model_variants", {}).get("vision_language_risk")
        return payload if isinstance(payload, dict) and payload.get("ok") else None
    return risk_summary


def predict_fixed_task_prior(risk_summary: Dict[str, Any], args: argparse.Namespace) -> Optional[float]:
    payload = risk_summary.get("model_variants", {}).get("structured_progress_risk", {})
    priors = payload.get("fixed_task_priors", [])
    fallback = payload.get("global_prior")
    for row in priors:
        if row.get("suite") == args.task_suite_name and int(row.get("task_id", -1)) == int(args.task_id):
            return float(row["failure_prior"])
    return float(fallback) if fallback is not None else None


def runtime_feature_vector(args: argparse.Namespace, task_description: str, step_logs: List[Dict[str, Any]]) -> List[float]:
    prefix = step_logs[-10:] if step_logs else []
    action_norms = [float(step.get("action_norm", 0.0) or 0.0) for step in prefix]
    smoothness = [float(step.get("action_smoothness", 0.0) or 0.0) for step in prefix]
    no_progress = [float(step.get("no_progress_score", 0.0) or 0.0) for step in prefix]
    rewards = [float(step.get("reward", 0.0) or 0.0) for step in prefix]
    stressor_name = args.stressor_name
    sev = max(0.0, min(1.0, float(args.stressor_severity)))
    return [
        1.0,
        int(args.task_id) / 100.0,
        hash_to_unit(args.task_suite_name),
        hash_to_unit(task_description),
        int(args.replan_steps) / 20.0,
        sev,
        float(stressor_name in ("", "none", "direct")),
        float(stressor_name == "occlusion"),
        float(stressor_name == "action_noise"),
        float(stressor_name == "gaussian_noise"),
        float(stressor_name == "brightness"),
        float(stressor_name == "action_delay"),
        float(stressor_name == "action_precision"),
        (sum(action_norms) / len(action_norms) if action_norms else 0.0) / 5.0,
        (max(action_norms) if action_norms else 0.0) / 5.0,
        (sum(smoothness) / len(smoothness) if smoothness else 0.0) / 2.0,
        sum(no_progress) / len(no_progress) if no_progress else 0.0,
        sum(rewards),
    ]


def decide_runtime_supervisor(
    args: argparse.Namespace,
    predicted_risk: Optional[float],
    *,
    risk_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    risk_threshold = runtime_abstain_threshold(args, risk_summary, None)
    if predicted_risk is None or args.mode not in (
        "selective_openpi",
        "adaptive_chunk_openpi",
        "no_progress_replan",
        VISION_SELECTIVE_MODE,
        FIXED_PRIOR_SELECTIVE_MODE,
    ):
        return {
            "action": args.mode,
            "n_action_steps": int(args.replan_steps),
            "reason": "no_risk_intervention",
            "predicted_risk": predicted_risk,
            "threshold": risk_threshold,
        }
    if args.mode in ("selective_openpi", VISION_SELECTIVE_MODE, FIXED_PRIOR_SELECTIVE_MODE) and predicted_risk >= risk_threshold:
        return {
            "action": "abstain",
            "n_action_steps": 1,
            "reason": "risk_exceeds_abstain_threshold",
            "predicted_risk": predicted_risk,
            "threshold": risk_threshold,
        }
    if args.mode in (VISION_SELECTIVE_MODE, FIXED_PRIOR_SELECTIVE_MODE):
        return {
            "action": args.mode,
            "n_action_steps": int(args.replan_steps),
            "reason": "risk_below_abstain_threshold",
            "predicted_risk": predicted_risk,
            "threshold": risk_threshold,
        }
    if args.mode == "adaptive_chunk_openpi":
        if predicted_risk < args.low_risk_threshold:
            horizon, reason = 10, "low_risk_default_horizon"
        elif predicted_risk < args.medium_risk_threshold:
            horizon, reason = 5, "medium_risk_medium_horizon"
        elif predicted_risk < args.high_risk_threshold:
            horizon, reason = 2, "high_risk_short_horizon"
        else:
            horizon, reason = 1, "extreme_risk_single_step"
        return {
            "action": "adaptive_chunk_openpi",
            "n_action_steps": horizon,
            "reason": reason,
            "predicted_risk": predicted_risk,
            "threshold": risk_threshold,
        }
    return {
        "action": args.mode,
        "n_action_steps": int(args.replan_steps),
        "reason": "risk_scored_no_horizon_change",
        "predicted_risk": predicted_risk,
        "threshold": risk_threshold,
    }


def runtime_abstain_threshold(
    args: argparse.Namespace,
    risk_summary: Optional[Dict[str, Any]],
    risk_payload: Optional[Dict[str, Any]],
) -> float:
    if not risk_summary:
        return float(args.abstain_threshold)
    if args.mode == VISION_SELECTIVE_MODE:
        payload = risk_payload or risk_summary.get("model_variants", {}).get("vision_language_risk", {})
        return float(payload.get("calibration", {}).get("threshold", args.abstain_threshold))
    if args.mode == FIXED_PRIOR_SELECTIVE_MODE:
        payload = risk_summary.get("model_variants", {}).get("structured_progress_risk", {})
        return float(payload.get("baseline_thresholds", {}).get("fixed_task_prior", args.abstain_threshold))
    return float(args.abstain_threshold)


def load_runtime_vision_encoder(args: argparse.Namespace) -> Any:
    return RuntimeSigLIPEncoder(args.siglip_model, int(args.siglip_dims), args.runtime_vision_device)


class RuntimeSigLIPEncoder:
    def __init__(self, model_name: str, dims: int, device_name: str) -> None:
        from transformers import AutoModel, AutoProcessor  # type: ignore
        import torch  # type: ignore

        self.torch = torch
        self.device = self._choose_device(device_name, torch)
        self.dims = int(dims)
        patch_siglip_tanh_gelu(torch)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, image: Any) -> List[float]:
        from PIL import Image  # type: ignore

        pil_image = Image.fromarray(image).convert("RGB")
        inputs = self.processor(images=[pil_image], return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model.get_image_features(**inputs)
            features = pooled_tensor(output).float()
            features = self.torch.nn.functional.normalize(features, dim=-1).detach().cpu()[0]
        return compact_embedding([float(value) for value in features.tolist()], self.dims)

    @staticmethod
    def _choose_device(device_name: str, torch_module: Any) -> str:
        if device_name == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested for SigLIP but torch.cuda.is_available() is false")
        return device_name


def pooled_tensor(output: Any) -> Any:
    if hasattr(output, "float"):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state.mean(dim=1)
    raise TypeError("Unsupported image feature output type: {0!r}".format(type(output)))


def patch_siglip_tanh_gelu(torch_module: Any) -> None:
    try:
        from transformers.activations import ACT2FN  # type: ignore
    except Exception:
        return

    class TanhGELU(torch_module.nn.Module):  # type: ignore[misc]
        def forward(self, value: Any) -> Any:
            return 0.5 * value * (
                1.0
                + torch_module.tanh(
                    math.sqrt(2.0 / math.pi) * (value + 0.044715 * torch_module.pow(value, 3))
                )
            )

    ACT2FN["gelu_pytorch_tanh"] = TanhGELU


def compact_embedding(values: Sequence[float], dims: int) -> List[float]:
    if dims <= 0:
        raise ValueError("dims must be positive")
    if dims >= len(values):
        return [float(value) for value in values]
    compact: List[float] = []
    width = len(values) / float(dims)
    for idx in range(dims):
        start = int(math.floor(idx * width))
        stop = int(math.floor((idx + 1) * width))
        stop = max(stop, start + 1)
        bucket = values[start:stop]
        compact.append(float(sum(bucket) / len(bucket)))
    return compact


def hash_to_unit(value: str) -> float:
    import hashlib

    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    number = int(digest[:8], 16)
    return (number / 0xFFFFFFFF) * 2.0 - 1.0


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


if __name__ == "__main__":
    raise SystemExit(main())
