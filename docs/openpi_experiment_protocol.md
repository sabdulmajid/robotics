# OpenPI/LIBERO Experiment Protocol

## Goal

Evaluate whether calibrated risk prediction improves the reliability/coverage tradeoff of OpenPI `pi05_libero` execution on LIBERO through selective rejection and adaptive action-horizon supervision.

## Modes

| Mode | Meaning |
| --- | --- |
| `direct_openpi` | Run OpenPI normally with a fixed action horizon. |
| `fixed_task_prior` | Use train-split per-task failure priors for risk reporting or rejection. |
| `learned_risk_openpi` | Score risk but do not intervene. |
| `selective_openpi` | Abstain if calibrated risk exceeds the chosen calibration threshold. |
| `adaptive_chunk_openpi` | Shorten action horizon as risk increases; query OpenPI more frequently. |
| `no_progress_replan` | Use no-progress windows to force short-horizon re-query/recovery when feasible. |

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
