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
import json
import logging
import math
import os
import pathlib
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
            replay_images.append(img)

            if not action_plan:
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
                if action_chunk_length < args.replan_steps:
                    raise RuntimeError(
                        "Policy predicted {0} actions, but replan_steps={1}".format(
                            action_chunk_length, args.replan_steps
                        )
                    )
                action_plan.extend(action_chunk[: args.replan_steps])
                action_queries += 1

            action = np_module.asarray(action_plan.popleft())
            obs, reward, done, _ = env.step(action.tolist())
            smoothness = None
            if previous_action is not None:
                smoothness = float(np_module.linalg.norm(action - previous_action))
            previous_action = action

            step_logs.append(
                {
                    "timestep": int(t - args.num_steps_wait),
                    "observation_keys": sorted(str(key) for key in obs.keys()),
                    "action_chunk_length": int(action_chunk_length),
                    "action_norm": float(np_module.linalg.norm(action)),
                    "action_smoothness": smoothness,
                    "predicted_risk": None,
                    "action_horizon": int(args.replan_steps),
                    "reward": float(reward),
                    "done": bool(done),
                    "success": bool(done),
                    "no_progress": None,
                }
            )
            t += 1
            if done:
                break
        except Exception as exc:  # pragma: no cover - external simulator boundary
            exception_text = repr(exc)
            logging.exception("Episode %s failed at simulator timestep %s", episode_idx, t)
            break

    terminal_label = "success" if done else "runtime_error" if exception_text else "timeout"
    video_path = None
    if replay_images:
        suffix = "success" if done else "failure"
        video_path = video_dir / "rollout_{0}_task{1:02d}_ep{2:03d}_{3}.mp4".format(
            args.task_suite_name, args.task_id, episode_idx, suffix
        )
        imageio_module.mimwrite(str(video_path), [np_module.asarray(x) for x in replay_images], fps=10)

    return {
        "episode_id": "{0}_task{1:02d}_seed{2}_ep{3:03d}".format(
            args.task_suite_name, args.task_id, args.seed, episode_idx
        ),
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
        "steps": step_logs,
        "metadata": {
            "host": args.host,
            "port": int(args.port),
            "resize_size": int(args.resize_size),
            "num_steps_wait": int(args.num_steps_wait),
            "max_steps": int(max_steps),
            "episode_index": int(episode_idx),
            "init_state_index": int(init_state_index),
            "action_queries": int(action_queries),
            "exception": exception_text,
            "video_path": str(video_path) if video_path is not None else None,
        },
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


if __name__ == "__main__":
    raise SystemExit(main())
