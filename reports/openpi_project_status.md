# OpenPI Risk-Aware Execution Project Status

Date: 2026-05-17

## Current Claim

The repository now demonstrates a real OpenPI/LIBERO risk-aware execution loop:

- OpenPI `pi05_libero` runs through a policy server on SLURM.
- LIBERO rollouts are logged as risk-training JSONL with action, reward, no-progress, predicted-risk, supervisor, video, and optional frame-image fields.
- The direct-policy dataset now has `993` audited OpenPI/LIBERO episodes for risk training and evaluation.
- The scaled expansion added `400` nominal episodes across `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`, plus `560` stress episodes over occlusion/action-noise severities.
- The risk report trains two transparent logistic ablations: `metadata_oracle_risk` and `structured_progress_risk`.
- The saved risk summary can be loaded back into the evaluator for selective rejection and adaptive action chunking.

The strongest honest statement is:

> Built an audited risk-supervision layer for OpenPI robot foundation policies on LIBERO, with scaled SLURM rollout collection, stress-test generation, calibrated rollout-failure prediction, selective-execution analysis, and runtime supervision hooks.

This is not a formal safety guarantee and not an OpenPI leaderboard claim.

## Open-Source Stack

| Project | Current role | Status |
| --- | --- | --- |
| OpenPI | Robot foundation policy / VLA backbone, using `pi05_libero` | active, smoke and scaled rollout jobs passing |
| LIBERO | Manipulation benchmark tasks and initial states | active, four suites logged |
| robosuite / MuJoCo | Headless simulation/rendering backend through LIBERO | active on `dualcard` with EGL; `midcard` had EGL-device initialization failures |
| VLM embeddings | Frozen image-language features for risk prediction | data path added through `SAVE_IMAGES=1`; embedding extraction/training not claimed yet |
| World model / progress model | Transition/progress signals for timeout/no-progress risk | prefix action/no-progress/reward statistics active; learned predictive dynamics planned |
| LeRobot | Dataset/export format and future policy baseline | planned, not used in current metrics |

## Scaled Rollout Results

| Job | Mode | Suite/stressor | Episodes | Success | Timeout | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 10120 | direct OpenPI | `libero_spatial`, `libero_object`, nominal | 200 | 199 | 1 | Nominal OpenPI is strong on these two suites. |
| 10121 | direct OpenPI | `libero_goal`, `libero_10`, nominal | 200 | 192 | 8 | Nominal failures exist but are sparse. |
| 10127 | direct OpenPI | occlusion/action-noise severity 0.2 | 140 | 135 | 5 | Mild stress mostly succeeds. |
| 10128 | direct OpenPI | occlusion/action-noise severity 0.4 | 140 | 126 | 14 | Moderate stress introduces a useful failure signal. |
| 10129 | direct OpenPI | occlusion/action-noise severity 0.6 | 140 | 118 | 22 | Failure rate increases but remains task/stressor dependent. |
| 10130 | direct OpenPI | occlusion severity 0.8 | 70 | 21 | 49 | High occlusion creates a strong timeout regime. |
| 10131 | direct OpenPI | occlusion severity 1.0 | 70 | 5 | 65 | Severe occlusion is mostly unsuccessful. |

Including earlier direct smoke/stress files, the training config uses `993` direct episodes and excludes supervisor runs `10097` and `10098`.

## Risk Critic Checkpoint

Final audited dataset split:

| Split | Episodes | Failures | Failure rate |
| --- | ---: | ---: | ---: |
| train | 595 | 104 | 0.175 |
| calibration | 197 | 34 | 0.173 |
| test | 201 | 36 | 0.179 |
| all | 993 | 174 | 0.175 |

Primary deployable model: `structured_progress_risk`, which excludes injected stressor metadata.

| Test model | AUROC | AUPRC | Brier | ECE | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| global prior | 0.500 | 0.179 | 0.147 | 0.004 | Constant risk; AUPRC equals positive rate after tie audit. |
| fixed task prior | 0.695 | 0.347 | 0.139 | 0.062 | Strong baseline because task identity is informative. |
| structured progress risk | 0.702 | 0.297 | 0.238 | 0.315 | Slightly higher AUROC than fixed prior, weaker calibration/AUPRC. |
| metadata oracle risk | 0.930 | 0.840 | 0.146 | 0.264 | Diagnostic upper bound; sees stressor metadata. |

At about 90% coverage on the test split:

| Policy | Coverage | Task completion | Failure among attempts |
| --- | ---: | ---: | ---: |
| direct OpenPI | 1.000 | 0.821 | 0.179 |
| fixed task prior selective | 0.900 | 0.771 | 0.144 |
| structured progress selective | 0.900 | 0.756 | 0.160 |
| metadata oracle selective | 0.900 | 0.821 | 0.088 |

Interpretation: there is exploitable risk structure, but the deployable structured model is not yet strong enough to beat fixed priors across all metrics. The metadata oracle shows the stress suite contains a strong signal; the next research step is replacing stressor metadata with observed VLM/world-model features.

## VLM And World-Model Integration

OpenPI is already the active VLA policy: it consumes RGB observations and language prompts and outputs robot actions. The risk layer currently wraps that policy using structured rollout features. The VLM/world-model path is now concrete:

1. `SAVE_IMAGES=1` in `slurm/openpi_libero_rollouts.sbatch` passes `--save-images` into the evaluator.
2. `scripts/openpi_libero_single_task_eval.py` writes per-step RGB frame paths into JSONL when image saving is enabled.
3. `vision_language_risk` is represented in the risk summary, but remains `skipped` until frozen image/language embeddings are extracted.
4. Prefix action norms, no-progress scores, action smoothness, and reward are already used as lightweight progress/world-model proxy features.
5. The next model should compare fixed priors, structured progress, frozen VLM embeddings, and learned predictive dynamics at matched coverage.

## Verification

- `python scripts/audit_openpi_metrics.py --risk-summary reports/openpi_libero_risk_summary.json --report reports/openpi_libero_risk_planning.md` passes with no failures.
- The audit recomputes split metrics and thresholds from raw JSONL.
- Supervisor/non-direct runs are excluded from training inputs.
- The global-prior AUPRC tie-handling bug is fixed and audited.
- `python -m pytest -q` is the final repo regression check.

## Resume Framing

Use this phrasing:

> Built a risk-aware execution layer for OpenPI robot foundation policies on LIBERO, including SLURM rollout infrastructure, stress-test generation, calibrated failure-risk prediction, selective rejection, adaptive replanning hooks, and coverage-aware evaluation over 993 audited policy rollouts.

Do not say the project has solved robot safety. The professional framing is calibrated risk-aware supervision for brittle robot foundation policy execution.
