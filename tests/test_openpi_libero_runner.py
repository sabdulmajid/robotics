from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_single_task_eval_dry_run_writes_libero_config(tmp_path: Path) -> None:
    openpi_root = tmp_path / "openpi"
    benchmark_root = openpi_root / "third_party/libero/libero/libero"
    for subdir in ("assets", "bddl_files", "init_files"):
        (benchmark_root / subdir).mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "libero_config"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/openpi_libero_single_task_eval.py",
            "--openpi-root",
            str(openpi_root),
            "--libero-config-path",
            str(config_path),
            "--task-suite-name",
            "libero_spatial",
            "--task-id",
            "0",
            "--num-trials",
            "1",
            "--dry-run",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = json.loads(completed.stdout)
    assert payload["ok"]
    config_file = config_path / "config.yaml"
    assert config_file.exists()
    config_text = config_file.read_text(encoding="utf-8")
    assert "benchmark_root:" in config_text
    assert "bddl_files:" in config_text
    assert "init_states:" in config_text
