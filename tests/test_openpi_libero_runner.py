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
        runtime_risk_threshold_override=None,
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
        runtime_risk_threshold_override=None,
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

    vision_args.runtime_risk_threshold_override = 0.95
    tuned_decision = module.decide_runtime_supervisor(vision_args, vision_risk, risk_summary=vision_summary)
    assert tuned_decision["action"] == "vision_language_risk_selective"
    assert tuned_decision["threshold"] == 0.95


def test_runtime_threshold_sweep_uses_task_disjoint_calibration() -> None:
    module = _load_script_module("scripts/sweep_openpi_runtime_thresholds.py", "openpi_threshold_sweep")
    episodes = []
    risks = {
        0: 0.1,
        1: 0.2,
        2: 0.9,
        3: 0.1,
        4: 0.8,
        5: 0.9,
    }
    direct_success = {
        0: True,
        1: True,
        2: False,
        3: True,
        4: False,
        5: False,
    }
    for task_id, risk in risks.items():
        episodes.append(_episode("direct_openpi", task_id, success=direct_success[task_id]))
        episodes.append(_episode("fixed_task_prior_selective", task_id, success=direct_success[task_id]))
        episodes.append(_episode("vision_language_risk_selective", task_id, success=False, risk=risk, abstained=risk > 0.5))

    sweep = module.build_threshold_sweep(episodes, target_coverages=[0.50], calibration_task_max=2)

    assert sweep["ok"]
    assert sweep["split"]["calibration_task_ids"] == [0, 1, 2]
    assert sweep["split"]["test_task_ids"] == [3, 4, 5]
    row = sweep["threshold_rows"][0]
    assert row["threshold_source"] == "runtime_calibration_split"
    assert row["test"]["coverage"] == 1 / 3
    assert row["test"]["failure_rate_attempted"] == 0.0


def test_controlled_deployment_summary_adds_matched_baselines() -> None:
    module = _load_script_module("scripts/summarize_openpi_controlled_deployment.py", "openpi_controlled")
    direct = [
        _episode("direct_openpi", 0, success=True),
        _episode("direct_openpi", 1, success=False),
        _episode("direct_openpi", 2, success=True),
        _episode("direct_openpi", 3, success=False),
    ]
    fixed = [_episode("fixed_task_prior_selective", idx, success=episode["success"]) for idx, episode in enumerate(direct)]
    siglip_0933 = [
        _episode("vision_language_risk_selective", 0, success=True, risk=0.1),
        _episode("vision_language_risk_selective", 1, success=False, risk=0.9, abstained=True),
        _episode("vision_language_risk_selective", 2, success=True, risk=0.1),
        _episode("vision_language_risk_selective", 3, success=False, risk=0.9, abstained=True),
    ]
    siglip_0986 = [
        _episode("vision_language_risk_selective", idx, success=episode["success"], risk=0.1)
        for idx, episode in enumerate(direct)
    ]
    manifest = [
        {"job_id": 1, "label": "direct"},
        {"job_id": 2, "label": "fixed"},
        {"job_id": 3, "label": "siglip_0933"},
        {"job_id": 4, "label": "siglip_0986"},
    ]

    summary = module.summarize_controlled_deployment(
        manifest,
        {
            "direct": direct,
            "fixed": fixed,
            "siglip_0933": siglip_0933,
            "siglip_0986": siglip_0986,
        },
        random_seeds=100,
    )

    assert summary["ok"]
    assert summary["grid_checks"]["same_grid"]
    assert summary["metrics"]["siglip_0933"]["coverage"] == 0.5
    assert summary["matched_baselines"]["siglip_0933"]["random_abstain_matched_coverage"]["samples"] == 100
    assert summary["matched_baselines"]["siglip_0933"]["oracle_abstain_upper_bound"]["source"].startswith(
        "diagnostic upper bound"
    )
    assert summary["analysis_questions"]["siglip_0933"]["reduces_attempted_failure_vs_direct"]


def test_multiseed_summary_reports_grouped_delta_cis() -> None:
    module = _load_script_module("scripts/summarize_openpi_multiseed_deployment.py", "openpi_multiseed")
    manifest = []
    episodes = []
    job_id = 1
    for experiment, suite, seed in [
        ("spatial_multiseed", "libero_spatial", 5000),
        ("spatial_multiseed", "libero_spatial", 6000),
        ("cross_suite", "libero_object", 5000),
    ]:
        for label, mode, threshold in [
            ("direct", "direct_openpi", None),
            ("siglip_0986", "vision_language_risk_selective", 0.9860334584902223),
        ]:
            row = {
                "experiment": experiment,
                "job_id": job_id,
                "label": label,
                "mode": mode,
                "suites": [suite],
                "seed": seed,
                "threshold": threshold,
            }
            manifest.append(row)
            for task_id, success in [(0, True), (1, False), (2, False)]:
                abstained = label == "siglip_0986" and not success
                episode = _episode(mode, task_id, success=success, risk=0.8 if abstained else 0.1, abstained=abstained)
                episode["libero_suite"] = suite
                episode["seed"] = seed
                episode["_manifest"] = row
                episodes.append(episode)
            job_id += 1

    summary = module.summarize_multiseed_deployment(manifest, episodes, random_seeds=50)

    assert summary["ok"]
    spatial = summary["groups"]["spatial_multiseed"]
    assert spatial["metrics"]["siglip_0986"]["coverage"] == 1 / 3
    assert spatial["comparisons_vs_direct"]["siglip_0986"]["utility_delta_ci95"]["samples"] == 1000
    assert spatial["matched_baselines"]["siglip_0986"]["random_abstain_matched_coverage"]["samples"] == 50
    assert summary["per_seed"]["spatial_multiseed:seed5000"]["episodes"] == 6
    assert summary["per_suite"]["cross_suite:libero_object"]["metrics"]["siglip_0986"]["coverage"] == 1 / 3
    assert summary["analysis_questions"]["cross_suite_generalization_holds"]


def _load_single_task_eval_module():
    return _load_script_module("scripts/openpi_libero_single_task_eval.py", "openpi_single_eval")


def _load_script_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episode(mode: str, task_id: int, *, success: bool, risk: float | None = None, abstained: bool = False) -> dict:
    decision = None
    if risk is not None:
        decision = {
            "predicted_risk": risk,
            "threshold": 0.5,
            "prefix_steps_observed": 10,
        }
    terminal_label = "abstained" if abstained else "success" if success else "timeout"
    return {
        "mode": mode,
        "libero_suite": "libero_spatial",
        "libero_task_id": task_id,
        "stressor_name": "none",
        "stressor_params": {"severity": 0.0},
        "success": success and not abstained,
        "timeout": (not success) and not abstained,
        "terminal_label": terminal_label,
        "failure_label": terminal_label,
        "episode_length": 10 if abstained else 50,
        "n_action_steps": 5,
        "runtime_supervisor_decision": decision,
        "metadata": {
            "episode_index": 0,
            "init_state_index": 0,
            "action_queries": 2 if abstained else 10,
            "runtime_supervisor_decision": decision,
        },
    }
