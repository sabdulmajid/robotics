from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from argparse import Namespace
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


def test_runtime_fixed_and_vision_risk_modes_score_from_summary() -> None:
    module = _load_single_task_eval_module()
    fixed_args = Namespace(
        mode="fixed_task_prior_selective",
        task_suite_name="libero_spatial",
        task_id=0,
        abstain_threshold=0.95,
        replan_steps=5,
    )
    fixed_summary = {
        "ok": True,
        "normalization": {},
        "model_variants": {
            "structured_progress_risk": {
                "global_prior": 0.2,
                "baseline_thresholds": {"fixed_task_prior": 0.3},
                "fixed_task_priors": [{"suite": "libero_spatial", "task_id": 0, "failure_prior": 0.4}],
            }
        },
    }

    fixed_risk = module.predict_runtime_risk(
        fixed_summary,
        args=fixed_args,
        task_description="pick up the bowl",
        step_logs=[],
    )
    fixed_decision = module.decide_runtime_supervisor(fixed_args, fixed_risk, risk_summary=fixed_summary)

    assert fixed_risk == 0.4
    assert fixed_decision["action"] == "abstain"
    assert fixed_decision["threshold"] == 0.3

    vision_args = Namespace(
        mode="vision_language_risk_selective",
        task_suite_name="libero_spatial",
        task_id=0,
        replan_steps=5,
        stressor_name="none",
        stressor_severity=0.0,
        abstain_threshold=0.95,
    )
    vision_summary = {
        "ok": True,
        "normalization": {},
        "model_variants": {
            "vision_language_risk": {
                "ok": True,
                "feature_names": ["bias", "siglip_image_000"],
                "weights": {"bias": 0.0, "siglip_image_000": 1.0},
                "normalization": {
                    "mean": {"bias": 1.0, "siglip_image_000": 0.0},
                    "std": {"bias": 1.0, "siglip_image_000": 1.0},
                },
                "calibration": {"temperature": 1.0, "threshold": 0.5},
            }
        },
    }

    vision_risk = module.predict_runtime_risk(
        vision_summary,
        args=vision_args,
        task_description="pick up the bowl",
        step_logs=[],
        vision_embedding=[1.0],
    )
    vision_decision = module.decide_runtime_supervisor(vision_args, vision_risk, risk_summary=vision_summary)

    assert vision_risk is not None and vision_risk > 0.7
    assert vision_decision["action"] == "abstain"
    assert vision_decision["threshold"] == 0.5


def _load_single_task_eval_module():
    spec = importlib.util.spec_from_file_location("openpi_single_eval", "scripts/openpi_libero_single_task_eval.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
