from __future__ import annotations

import importlib.util
import json
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

    module_status = {
        "openpi_client": importlib.util.find_spec("openpi_client") is not None,
        "libero": importlib.util.find_spec("libero") is not None,
        "mujoco": importlib.util.find_spec("mujoco") is not None,
    }
    if openpi_root.exists() and not module_status["openpi_client"]:
        blockers.append("openpi_client is not importable in the current Python environment")
    if openpi_root.exists() and not module_status["libero"]:
        blockers.append("libero is not importable in the current Python environment")

    gpu_status = _capture_command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    git_status = _capture_command(["git", "rev-parse", "HEAD"], cwd=openpi_root) if openpi_root.exists() else None

    resume_commands = _resume_commands(config)
    status = {
        "ok": not blockers,
        "config": config.to_dict(),
        "python": sys.version,
        "platform": platform.platform(),
        "command_status": command_status,
        "module_status": module_status,
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


def _capture_command(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    if shutil.which(command[0]) is None:
        return {"available": False, "command": command, "returncode": None, "stdout": "", "stderr": ""}
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
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
