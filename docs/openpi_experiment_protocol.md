# OpenPI/LIBERO Experiment Protocol

## Goal

Evaluate whether observed-state risk improves OpenPI `pi05_libero` execution under controlled LIBERO distribution shift.

Measure coverage, task completion, attempted failure, and utility.

Do not infer utility improvement from failure reduction alone.

## Modes

| Mode | Meaning |
| --- | --- |
| `direct_openpi` | Run OpenPI normally with a fixed action horizon. |
| `fixed_task_prior` | Use train-split per-task failure priors for risk reporting or rejection. |
| `learned_risk_openpi` | Score risk but do not intervene. |
| `selective_openpi` | Abstain if calibrated risk exceeds the chosen calibration threshold. |
| `adaptive_chunk_openpi` | Shorten action horizon as risk increases; query OpenPI more frequently. |
| `no_progress_replan` | Use no-progress windows to force short-horizon re-query/recovery when feasible. |
| `vision_language_risk_selective` | Use frozen SigLIP and progress features for runtime rejection. |

## Current Reproduction Commands

```bash
PYTHONPATH=src python scripts/openpi_libero_smoke.py --config configs/openpi_libero_smoke.yaml --strict
python scripts/collect_openpi_libero.py --config configs/openpi/libero_collect_baseline.yaml
python scripts/collect_openpi_libero.py --config configs/openpi/libero_collect_stress.yaml
PYTHONPATH=src python scripts/train_openpi_risk.py --config configs/openpi/train_risk.yaml
python scripts/eval_openpi_supervisor.py --config configs/openpi/eval_supervisor.yaml
python scripts/summarize_openpi_results.py --run-dir reports
```

Add `--submit` to the collection/evaluation scripts to submit their printed `sbatch` commands.

Run the current retrospective cross-suite diagnostic with:

```bash
PYTHONPATH=src python scripts/analyze_openpi_cross_suite_retrospective.py
python scripts/plot_openpi_cross_suite_retrospective.py
```

## Fresh Cross-Suite Protocol

Use `libero_object` and `libero_goal`.

Use tasks `0..4` and seed `7500` for threshold calibration.

Use tasks `5..9` and seed `8000` for held-out deployment.

Use `none:0.0`, `occlusion:0.8`, and `action_noise:0.6`.

Compare direct OpenPI, spatial threshold `0.9860`, and the frozen cross-suite threshold.

Select the threshold before the deployment jobs start.

The prepared tools are:

- `scripts/calibrate_openpi_cross_suite_threshold.py`
- `scripts/summarize_openpi_calibrated_cross_suite.py`

The 2026-09-22 attempt produced zero episodes because cluster infrastructure blocked execution.

See `reports/openpi_cross_suite_online_followup_status.json` for the exact state.

## Stress Suite

The current stress suite uses observation occlusion and action noise. It is a robustness/stress evaluation, not a standard LIBERO leaderboard result. Stress labels are allowed for analysis and controlled experiments, but the next professional risk model should replace them with observed image/language/progress features.

## Data Splits

Risk training currently uses direct-policy runs only:

- `datasets/openpi_libero_rollouts/openpi_rollouts_10094.jsonl`
- `datasets/openpi_libero_rollouts/openpi_rollouts_10095.jsonl`
- `datasets/openpi_libero_rollouts/openpi_rollouts_10096.jsonl`

Supervisor runs `10097` and `10098` are evaluation artifacts and are intentionally excluded from risk training.

## Metrics

Report:

- success, timeout, abstention, coverage;
- failure-at-coverage and success-at-coverage;
- Brier, NLL, ECE, AUROC, AUPRC;
- runtime overhead through episode length and policy-query counts;
- exact OpenPI commit, checkpoint, LIBERO suite/task IDs, and SLURM hardware.

Do not claim formal safety or real-world deployment.
