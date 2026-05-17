from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from risk_aware_skill_planning.backends.openpi.config import OpenPIBackendConfig


@dataclass(frozen=True)
class OpenPIPolicyServerCommand:
    config: OpenPIBackendConfig

    def argv(self) -> list[str]:
        return [
            "uv",
            "run",
            "scripts/serve_policy.py",
            "--env",
            "LIBERO",
            "--port",
            str(self.config.server_port),
        ]

    def cwd(self) -> Path:
        return self.config.root


def openpi_cache_env(repo_root: Path) -> dict[str, str]:
    return {
        "OPENPI_DATA_HOME": str(repo_root / "checkpoints/openpi_cache"),
        "XDG_CACHE_HOME": str(repo_root / ".cache"),
        "HF_HOME": str(repo_root / ".cache/huggingface"),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.80",
    }
