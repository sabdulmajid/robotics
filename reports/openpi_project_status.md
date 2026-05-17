# OpenPI Risk-Aware Execution Project Status

Date: 2026-05-17

## Current Claim

The repository now demonstrates a real OpenPI/LIBERO risk-aware execution loop:

- `pi05_libero` runs through an OpenPI policy server on SLURM.
- LIBERO rollouts are logged as risk-training JSONL with per-step action, image, no-progress, reward, predicted-risk, and supervisor fields.
- Direct-policy baselines expose a real stress failure regime.
- A calibrated rollout-risk baseline trains on direct OpenPI episodes only.
- The saved risk summary can be loaded back into the LIBERO evaluator for runtime selective rejection and adaptive action chunking.

This is not yet a final research result. The strongest honest statement is: **the infrastructure is real, OpenPI is active, stress failures are measurable, risk prediction is wired into runtime supervision, and the first simple adaptive intervention did not recover severe occlusion failures.**

## Open-Source Stack

| Project | Current role | Status |
| --- | --- | --- |
| OpenPI | Robot foundation policy / vision-language-action backbone, using `pi05_libero` | active, smoke and rollout jobs passing |
| LIBERO | Manipulation benchmark tasks and initial states | active, `libero_spatial` tasks 0, 1, 2 evaluated |
| robosuite / MuJoCo | Headless simulation/rendering backend through LIBERO | active on SLURM with EGL |
| LeRobot | Dataset/export format and future policy baseline | planned, not yet an active baseline |
| VLM embeddings | Future frozen image-language features for risk prediction | planned; current risk critic is structured/log-only |
| World model / progress model | Future transition/no-progress predictor for failure risk | planned; logs now contain the fields needed to train it |

## Rollout Results

| Job | Mode | Stressor | Episodes | Success | Timeout | Abstain | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 10092 | official OpenPI smoke | none | 1 | 1 | 0 | 0 | Proves the OpenPI policy server and LIBERO client path work. |
| 10094 | direct OpenPI | none | 9 | 9 | 0 | 0 | Nominal baseline succeeds on the selected spatial tasks. |
| 10095 | direct OpenPI | occlusion/action_noise, severity 0.7 | 12 | 10 | 2 | 0 | Moderate stress creates some failures. |
| 10096 | direct OpenPI | occlusion, severity 1.0 | 9 | 1 | 8 | 0 | Severe occlusion gives a meaningful failure regime. |
| 10097 | selective OpenPI | occlusion, severity 1.0 | 9 | 0 | 0 | 9 | Risk model rejects all severe-occlusion attempts. Coverage is zero but unsafe/timeouting executions are avoided. |
| 10098 | adaptive chunk OpenPI | occlusion, severity 1.0 | 6 | 0 | 6 | 0 | Runtime risk supervision executed, but simple action-horizon shortening did not recover successes. |
| 10099 | direct OpenPI cross-suite smoke | none | 3 | 3 | 0 | 0 | One task-0 episode each for `libero_object`, `libero_goal`, and `libero_10`; validates cross-suite execution path. |

## Risk Critic Checkpoint

Training data is intentionally restricted to direct-policy rollouts: jobs `10094`, `10095`, and `10096`.

| Split | Episodes | Failures | Failure rate |
| --- | ---: | ---: | ---: |
| train | 18 | 6 | 0.333 |
| calibration | 6 | 2 | 0.333 |
| test | 6 | 2 | 0.333 |
| all | 30 | 10 | 0.333 |

Calibration selected temperature `5.0` and risk threshold `0.8`.

| Test model | AUROC | AUPRC | Brier | NLL | ECE | Coverage @ threshold | Failure rate attempted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| global prior | 0.500 | 1.000 | 0.222 | 0.637 | 0.000 | 1.000 | 0.333 |
| fixed task prior | 1.000 | 1.000 | 0.121 | 0.401 | 0.312 | 1.000 | 0.333 |
| logistic state/progress risk | 0.750 | 0.750 | 0.432 | 1.122 | 0.465 | 0.667 | 0.250 |

The fixed task prior is currently strong because the dataset is small and task/stressor identity carries a lot of information. The project should not overclaim this as a learned-risk win yet.

## VLM And World-Model Integration Path

OpenPI is the active VLA backbone: it consumes camera observations and language instructions and outputs robot actions. The risk layer currently wraps that policy using structured rollout features. To make the project stronger and resume-worthy, the next technical step is to replace stressor metadata with observed features:

1. **Frozen VLM risk features:** encode the initial RGB frame, wrist frame, and language instruction with a frozen vision-language encoder, then train the same calibrated risk heads on those embeddings plus task/action summaries.
2. **World-model/progress features:** train a lightweight transition or progress predictor from early rollout prefixes to estimate no-progress, likely timeout, or future reward stagnation.
3. **Policy-stack comparison:** keep OpenPI as the primary policy, add a LeRobot-format dataset export, and optionally compare with a LeRobot-compatible baseline policy only after data volume is large enough.
4. **Supervisor redesign:** use risk to choose between attempt, abstain, recover, or alternate strategy. The first adaptive chunking experiment shows that re-querying OpenPI more often is not enough under severe occlusion.

## Best Next Step

The professional path is to present the current work as an executable robot-foundation-policy risk harness, then add one stronger learned signal:

1. Collect more direct OpenPI rollouts across `libero_spatial`, `libero_object`, and `libero_goal`, with seed-disjoint stress settings.
2. Add frozen image/language embeddings to the risk dataset and compare them against global and fixed-task priors.
3. Add a progress/world-model feature head trained to predict timeout/no-progress from the first `N` steps.
4. Evaluate at matched coverage: direct OpenPI, fixed prior rejection, VLM risk rejection, VLM plus progress/world-model risk, and oracle/stress labels only as an upper-bound diagnostic.

The resume framing should be:

> Built a risk-aware execution layer for OpenPI robot foundation policies on LIBERO, including SLURM rollout infrastructure, stress-test generation, calibrated failure-risk prediction, selective rejection, adaptive replanning hooks, and coverage-aware evaluation.
