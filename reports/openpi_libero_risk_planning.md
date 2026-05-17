# OpenPI/LIBERO Risk Training

Status: `PASS`

This report is the current robot-foundation-policy checkpoint for the project. OpenPI `pi05_libero` is used as the vision-language-action policy, LIBERO supplies the manipulation tasks, and this layer learns a rollout-level failure-risk model used for selective execution and adaptive action chunking.

## Dataset

The risk dataset is built from direct OpenPI/LIBERO rollouts. Each row is one episode converted into initial/task/stressor features plus early rollout progress statistics; labels mark any terminal failure or timeout.

```json
{
  "all": {
    "examples": 30,
    "failure_rate": 0.3333333333333333,
    "failures": 10,
    "timeout_rate": 0.3333333333333333,
    "timeouts": 10
  },
  "calibration": {
    "examples": 6,
    "failure_rate": 0.3333333333333333,
    "failures": 2,
    "timeout_rate": 0.3333333333333333,
    "timeouts": 2
  },
  "test": {
    "examples": 6,
    "failure_rate": 0.3333333333333333,
    "failures": 2,
    "timeout_rate": 0.3333333333333333,
    "timeouts": 2
  },
  "train": {
    "examples": 18,
    "failure_rate": 0.3333333333333333,
    "failures": 6,
    "timeout_rate": 0.3333333333333333,
    "timeouts": 6
  }
}
```

## Calibration

Temperature scaling selected `T=5.0` and planner threshold `0.8` on the calibration split.

```json
{
  "method": "temperature_scaling_grid",
  "temperature": 5.0,
  "threshold": 0.8
}
```

## Test Metrics

| Model | AUROC | AUPRC | Brier | NLL | ECE | Coverage @ threshold | Failure rate attempted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| global prior | 0.500 | 1.000 | 0.222 | 0.637 | 0.000 | 1.000 | 0.333 |
| fixed task prior | 1.000 | 1.000 | 0.121 | 0.401 | 0.312 | 1.000 | 0.333 |
| logistic state/progress risk | 0.750 | 0.750 | 0.432 | 1.122 | 0.465 | 0.667 | 0.250 |

The test split is still small, so these numbers should be treated as an engineering checkpoint rather than a final benchmark claim. The fixed task prior is a strong baseline because the current stress suite makes task identity informative; the next step is to add richer image/language and transition features so the learned critic can beat fixed priors under held-out perturbations.

## Offline Supervisor

```json
{
  "interpretation": "Episodes with calibrated p_failure >= threshold are abstained in this offline coverage analysis.",
  "mode": "selective_openpi",
  "test_coverage": 0.6666666666666666,
  "test_failure_rate_attempted": 0.25,
  "threshold": 0.8,
  "threshold_source": "calibration_split"
}
```

## Reproduce

```bash
SUITES="libero_spatial" TASK_IDS="0 1 2" NUM_TRIALS=3 STRESSORS="none" sbatch slurm/openpi_libero_rollouts.sbatch
SUITES="libero_spatial" TASK_IDS="0 1 2" NUM_TRIALS=3 STRESSORS="occlusion" STRESSOR_SEVERITY=1.0 sbatch slurm/openpi_libero_rollouts.sbatch
PYTHONPATH=src python scripts/train_openpi_risk.py --config configs/openpi/train_risk.yaml
MODE=adaptive_chunk_openpi RISK_SUMMARY=reports/openpi_libero_risk_summary.json SUITES="libero_spatial" TASK_IDS="0 1 2" NUM_TRIALS=2 STRESSORS="occlusion" STRESSOR_SEVERITY=1.0 sbatch slurm/openpi_libero_rollouts.sbatch
```

## Limitations

- The current OpenPI result is a small rollout study, not a benchmark-scale LIBERO evaluation.
- The present risk features include stressor metadata for controlled stress testing; the production path should replace this with directly observed image/language/progress features.
- The learned risk critic is a transparent logistic baseline. Neural VLM/world-model features are planned after this executable risk-supervision loop is stable.
