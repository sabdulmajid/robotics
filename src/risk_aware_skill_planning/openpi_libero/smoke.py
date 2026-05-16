from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from risk_aware_skill_planning.openpi_libero.config import OpenPILiberoSmokeConfig


def run_openpi_libero_smoke(config: OpenPILiberoSmokeConfig) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    openpi_root = config.openpi_root

    if not openpi_root.exists():
        blockers.append(f"OpenPI root does not exist: {openpi_root}")
    else:
        _require_path(openpi_root / "examples/libero/main.py", blockers)
        _require_path(openpi_root / "scripts/serve_policy.py", blockers)
        _require_path(openpi_root / "third_party/libero", blockers)

    command_status = {name: shutil.which(name) is not None for name in ("git", "uv", "docker", "nvidia-smi")}
    if not command_status["git"]:
        blockers.append("git is not available")
    if not command_status["uv"]:
        blockers.append("uv is not available; OpenPI's LIBERO docs use uv for the non-Docker path")
    if not command_status["nvidia-smi"]:
        warnings.append("nvidia-smi is not available on this node; run GPU smoke through SLURM")

    current_module_status = {
        "openpi_client": importlib.util.find_spec("openpi_client") is not None,
        "libero": importlib.util.find_spec("libero") is not None,
        "mujoco": importlib.util.find_spec("mujoco") is not None,
    }
    venv_python = openpi_root / "examples/libero/.venv/bin/python"
    libero_pythonpath = openpi_root / "third_party/libero"
    venv_module_status = _venv_module_status(venv_python, libero_pythonpath) if venv_python.exists() else {}
    if openpi_root.exists() and not venv_python.exists():
        blockers.append(f"OpenPI LIBERO venv does not exist: {venv_python}")
    if venv_python.exists() and not venv_module_status.get("openpi_client", False):
        blockers.append("openpi_client is not importable in the OpenPI LIBERO venv")
    if venv_python.exists() and not venv_module_status.get("libero", False):
        blockers.append("libero is not importable in the OpenPI LIBERO venv")

    gpu_status = _capture_command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if gpu_status["available"] and gpu_status["returncode"] not in (0, None):
        warnings.append("nvidia-smi is available but failed; GPU driver/NVML may be unusable on this node")
    git_status = _capture_command(["git", "rev-parse", "HEAD"], cwd=openpi_root) if openpi_root.exists() else None

    resume_commands = _resume_commands(config)
    status = {
        "ok": not blockers,
        "config": config.to_dict(),
        "python": sys.version,
        "platform": platform.platform(),
        "command_status": command_status,
        "module_status": {
            "current_python": current_module_status,
            "openpi_libero_venv": venv_module_status,
            "venv_python": str(venv_python),
            "libero_pythonpath": str(libero_pythonpath),
        },
        "gpu_status": gpu_status,
        "openpi_git_commit": git_status,
        "blockers": blockers,
        "warnings": warnings,
        "resume_commands": resume_commands,
    }
    return status


def write_smoke_report(status: dict[str, Any], path: str | Path) -> None:
    config = status["config"]
    blockers = status["blockers"]
    warnings = status["warnings"]
    lines = [
        "# OpenPI/LIBERO Smoke Status",
        "",
        f"Experiment: `{config['experiment_id']}`",
        f"OpenPI root: `{config['openpi_root']}`",
        f"Policy config: `{config['policy_config']}`",
        f"Checkpoint: `{config['checkpoint']}`",
        f"LIBERO suite/task: `{config['suite']}` / `{config['task_id']}`",
        "",
        f"Status: `{'PASS' if status['ok'] else 'BLOCKED'}`",
        "",
    ]
    if blockers:
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in blockers)
        lines.append("")
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines.extend(
        [
            "## Resume Commands",
            "",
            "```bash",
            *status["resume_commands"],
            "```",
            "",
            "## Raw Status",
            "",
            "```json",
            json.dumps(status, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _require_path(path: Path, blockers: list[str]) -> None:
    if not path.exists():
        blockers.append(f"Expected OpenPI path is missing: {path}")


def _capture_command(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    if shutil.which(command[0]) is None:
        return {"available": False, "command": command, "returncode": None, "stdout": "", "stderr": ""}
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except Exception as exc:  # pragma: no cover - defensive external command boundary
        return {"available": True, "command": command, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "available": True,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _venv_module_status(venv_python: Path, libero_pythonpath: Path) -> dict[str, bool]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{libero_pythonpath}:{env.get('PYTHONPATH', '')}"
    command = [
        str(venv_python),
        "-c",
        (
            "import importlib.util, json; "
            "mods=['openpi_client','libero','libero.lifelong','mujoco','torch']; "
            "print(json.dumps({m: importlib.util.find_spec(m) is not None for m in mods}))"
        ),
    ]
    result = _capture_command(command, env=env)
    if result["returncode"] != 0:
        return {"_probe_failed": True}
    try:
        parsed = json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"_probe_failed": True}
    return {str(key): bool(value) for key, value in parsed.items()}


def _resume_commands(config: OpenPILiberoSmokeConfig) -> list[str]:
    root = config.openpi_root
    return [
        f"git clone --recurse-submodules {config.openpi_repo_url} {root}",
        f"cd {root}",
        "git submodule update --init --recursive",
        "uv venv --python 3.8 examples/libero/.venv",
        "source examples/libero/.venv/bin/activate",
        "uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match",
        "uv pip install -e packages/openpi-client",
        "uv pip install -e third_party/libero",
        "export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero",
        "uv run scripts/serve_policy.py --env LIBERO",
        "MUJOCO_GL=egl python examples/libero/main.py --task-suite-name " + config.suite,
    ]
