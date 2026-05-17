# OpenPI/LIBERO Risk-Aware Execution Report

Date: 2026-05-17

## Summary

This checkpoint evaluates **risk-aware execution for OpenPI robot foundation policies** on LIBERO. OpenPI `pi05_libero` is used as the vision-language-action policy, LIBERO supplies the manipulation tasks, and this repository adds rollout logging, stress testing, calibrated failure-risk prediction, selective rejection, and adaptive action-horizon supervision.

The result is meaningful but not final: OpenPI runs for real, severe occlusion creates a measurable failure regime, a calibrated rollout-risk baseline can reject high-risk attempts, and the first adaptive action-horizon intervention did **not** recover successes under severe occlusion. This is a robotics systems result, not a formal safety result or LIBERO leaderboard claim.

## Hardware

- Cluster: ECE Nebula SLURM, repo under `/mnt/slurm_nfs/a6abdulm/robotics`.
- Main jobs: `dualcard`, 1 GPU, 6 CPUs, 32G RAM.
- GPU recorded by setup smoke: 2 x NVIDIA RTX A4500, 20470 MiB each, driver `575.57.08`.
- Headless rendering: `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`.

## OpenPI And LIBERO

- OpenPI repo: `external/openpi`.
- OpenPI commit: `c23745b5ad24e98f66967ea795a07b2588ed6c79`.
- Policy config: `pi05_libero`.
- Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero/`, cached under `checkpoints/openpi_cache`.
- Benchmark: primary risk experiments use LIBERO `libero_spatial`, tasks `0`, `1`, `2`; cross-suite smoke also covers `libero_object`, `libero_goal`, and `libero_10` task `0`.
- Official smoke: job `10092`, 1 episode, 1 success, 101 steps.

## Stress Protocol

The nominal baseline uses no stressor. The robustness suite uses controlled stressors, clearly separate from standard LIBERO benchmark reporting:

- moderate stress: occlusion and action noise, severity `0.7`;
- severe stress: occlusion, severity `1.0`.

Stress metadata is currently used by the transparent logistic risk baseline. The next research step is to replace stress metadata with directly observed image/language/progress features.

## Rollout Results

| Job | Mode | Stressor | Episodes | Success | Timeout | Abstain | Mean steps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10094 | direct OpenPI | none | 9 | 9 | 0 | 0 | 99.3 |
| 10095 | direct OpenPI | occlusion/action_noise, severity 0.7 | 12 | 10 | 2 | 0 | 123.9 |
| 10096 | direct OpenPI | occlusion, severity 1.0 | 9 | 1 | 8 | 0 | 213.6 |
| 10097 | selective OpenPI | occlusion, severity 1.0 | 9 | 0 | 0 | 9 | 0.0 |
| 10098 | adaptive chunk OpenPI | occlusion, severity 1.0 | 6 | 0 | 6 | 0 | 220.0 |
| 10099 | direct OpenPI cross-suite smoke | none | 3 | 3 | 0 | 0 | 176.0 |

Interpretation:

- Nominal selected spatial tasks are easy for OpenPI: `9/9` direct success.
- Severe occlusion is failure-rich: direct OpenPI drops to `1/9`.
- Selective risk supervision rejects every severe-occlusion attempt at the current threshold. Coverage is zero, but timeouting executions are avoided.
- Adaptive chunking executes the runtime intervention and shortens the horizon, but still times out on `6/6` severe-occlusion episodes.

## Risk Dataset

Risk training uses direct OpenPI rollouts only: jobs `10094`, `10095`, and `10096`. Supervisor evaluation jobs `10097` and `10098` are excluded from training.

| Split | Episodes | Failures | Failure rate | Timeouts |
| --- | ---: | ---: | ---: | ---: |
| train | 18 | 6 | 0.333 | 6 |
| calibration | 6 | 2 | 0.333 | 2 |
| test | 6 | 2 | 0.333 | 2 |
| all | 30 | 10 | 0.333 | 10 |

## Risk Model Features

Current structured/log features:

- suite hash, task id, language hash;
- action horizon;
- stressor one-hot and severity;
- prefix action norm mean/max;
- prefix action smoothness;
- prefix no-progress score;
- prefix reward sum.

VLM/world-model hooks are planned and documented: per-step RGB image paths, task language, no-progress, action smoothness, and reward are already logged so frozen image-language embeddings and learned progress predictors can be added without changing the rollout schema.

## Calibration And Risk Metrics

Model: calibrated logistic state/progress/stress risk baseline. Calibration uses temperature scaling on the calibration split. Selected temperature: `5.0`. Selected threshold: `0.8`.

| Test model | AUROC | AUPRC | Brier | NLL | ECE | Coverage @ threshold | Failure rate attempted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| global prior | 0.500 | 1.000 | 0.222 | 0.637 | 0.000 | 1.000 | 0.333 |
| fixed task prior | 1.000 | 1.000 | 0.121 | 0.401 | 0.312 | 1.000 | 0.333 |
| logistic state/progress risk | 0.750 | 0.750 | 0.432 | 1.122 | 0.465 | 0.667 | 0.250 |

This is not yet a learned-risk win over fixed priors. The fixed task prior is strong because the current small stress dataset makes task identity highly informative. The correct next experiment is to scale data and add observed VLM/progress features.

## Supervisor Policy

Selective mode:

- if calibrated `p_failure >= 0.8`, abstain before execution.

Adaptive action-horizon mode:

- low risk: use normal action horizon;
- medium risk: shorten horizon;
- high/extreme risk: query OpenPI every 1-2 simulator steps;
- log predicted risk and the decision at every step.

Under severe occlusion, adaptive runs predicted initial risks around `0.992-0.994` and used horizons `[1, 2]`.

## Direct Versus Risk-Aware

| Comparison on severe occlusion | Coverage | Success over all episodes | Timeout over all episodes | Abstention |
| --- | ---: | ---: | ---: | ---: |
| direct OpenPI | 1.000 | 0.111 | 0.889 | 0.000 |
| selective OpenPI | 0.000 | 0.000 | 0.000 | 1.000 |
| adaptive chunk OpenPI | 1.000 | 0.000 | 1.000 | 0.000 |

Coverage/failure tradeoff:

- Selective supervision reduces timeouts at the cost of full rejection under this severe setting.
- Adaptive action-horizon supervision increases compute overhead and does not improve success in this first severe-occlusion test.
- Cross-suite task-0 smoke passed for `libero_object`, `libero_goal`, and `libero_10`; this validates that the runner is not Spatial-only, while the risk results remain Spatial-only.

## Runtime Overhead

Severe direct OpenPI averaged `42.8` policy queries and `213.6` simulator steps per episode. Adaptive chunking averaged `161.3` policy queries and `220.0` simulator steps per episode. The intervention is therefore substantially more expensive in the current form.

## Failure Examples

- Direct severe occlusion job `10096`: 8 of 9 episodes timed out; videos are under `videos/openpi_libero_spatial_task*_occlusion_10096`.
- Adaptive severe occlusion job `10098`: 6 of 6 episodes timed out despite high-risk short-horizon querying; videos are under `videos/openpi_libero_spatial_task*_occlusion_10098`.
- Selective job `10097`: 9 of 9 episodes abstained before motion, so no failure video is generated beyond immediate rejection logs.

## Reproduction Commands

```bash
PYTHONPATH=src python scripts/openpi_libero_smoke.py --config configs/openpi_libero_smoke.yaml --strict
sbatch slurm/openpi_libero_official_smoke.sbatch

python scripts/collect_openpi_libero.py --config configs/openpi/libero_collect_baseline.yaml
python scripts/collect_openpi_libero.py --config configs/openpi/libero_collect_stress.yaml
PYTHONPATH=src python scripts/train_openpi_risk.py --config configs/openpi/train_risk.yaml
python scripts/eval_openpi_supervisor.py --config configs/openpi/eval_supervisor.yaml
python scripts/summarize_openpi_results.py --run-dir reports
```

Add `--submit` to `collect_openpi_libero.py` or `eval_openpi_supervisor.py` to launch SLURM jobs.

The exact historical SLURM commands used for the tracked runs were:

```bash
SUITES="libero_spatial" TASK_IDS="0 1 2" NUM_TRIALS=3 STRESSORS="none" sbatch slurm/openpi_libero_rollouts.sbatch
SUITES="libero_spatial" TASK_IDS="0 1 2" NUM_TRIALS=2 STRESSORS="occlusion action_noise" STRESSOR_SEVERITY=0.7 sbatch slurm/openpi_libero_rollouts.sbatch
SUITES="libero_spatial" TASK_IDS="0 1 2" NUM_TRIALS=3 STRESSORS="occlusion" STRESSOR_SEVERITY=1.0 sbatch slurm/openpi_libero_rollouts.sbatch
MODE=selective_openpi RISK_SUMMARY=reports/openpi_libero_risk_summary.json SUITES="libero_spatial" TASK_IDS="0 1 2" NUM_TRIALS=3 STRESSORS="occlusion" STRESSOR_SEVERITY=1.0 sbatch slurm/openpi_libero_rollouts.sbatch
MODE=adaptive_chunk_openpi RISK_SUMMARY=reports/openpi_libero_risk_summary.json SUITES="libero_spatial" TASK_IDS="0 1 2" NUM_TRIALS=2 STRESSORS="occlusion" STRESSOR_SEVERITY=1.0 sbatch slurm/openpi_libero_rollouts.sbatch
```

## Limitations

- Only 30 direct OpenPI episodes are used for risk training; this is below the desired 300-500+ episode dataset.
- The current learned risk baseline is logistic and uses controlled stress metadata.
- The fixed task prior is stronger than the learned critic on this small held-out split.
- Adaptive action chunking alone is not enough to recover severe occlusion failures.
- No formal safety or real-world deployment claim is made.
- Current results cover selected `libero_spatial` tasks only, not the full LIBERO suite family.

## Next Step

Scale direct rollouts across `libero_object` and `libero_goal`, then train a risk critic with frozen VLM image-language embeddings plus a world-model/progress feature head. The main scientific question should then be evaluated at matched coverage against direct OpenPI and fixed task priors.
