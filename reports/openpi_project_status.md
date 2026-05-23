# OpenPI Risk-Aware Execution Project Status

Date: 2026-05-23

## Current Claim

The repository now demonstrates a real OpenPI/LIBERO risk-aware execution loop:

- OpenPI `pi05_libero` runs through a policy server on SLURM.
- LIBERO rollouts are logged as risk-training JSONL with action, reward, no-progress, predicted-risk, supervisor, video, and optional frame-image fields.
- The direct-policy dataset now has `993` audited OpenPI/LIBERO episodes for risk training and evaluation.
- The scaled expansion added `400` nominal episodes across `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`, plus `560` stress episodes over occlusion/action-noise severities.
- The risk report trains three transparent logistic ablations: `metadata_oracle_risk`, `structured_progress_risk`, and frozen-SigLIP `vision_language_risk`.
- The saved risk summary can be loaded back into the evaluator for selective rejection and adaptive action chunking.
- The frozen-SigLIP risk model now runs online inside OpenPI/LIBERO rollouts as `vision_language_risk_selective`, using a runtime RGB frame and 10-step progress prefix before either executing or abstaining.
- A held-out runtime grid has `630` real OpenPI/LIBERO episodes across direct OpenPI, fixed task priors, and runtime SigLIP supervision.
- A fresh same-seed controlled deployment adds `500` online episodes across direct OpenPI, fixed task priors, and two tuned runtime SigLIP thresholds on identical tasks, stressors, and seeds.
- A multiseed/cross-suite controlled deployment adds `1,950` online episodes across `69` SLURM jobs: three `libero_spatial` seeds plus `libero_object`/`libero_goal` cross-suite runs.

The strongest honest statement is:

> Built an audited risk-supervision layer for OpenPI robot foundation policies on LIBERO, with scaled SLURM rollout collection, stress-test generation, calibrated rollout-failure prediction, frozen SigLIP image-risk models, runtime selective execution, task-disjoint threshold tuning, and coverage-aware evaluation over 3,080 held-out online comparison episodes.

This is not a formal safety guarantee and not an OpenPI leaderboard claim.

## Open-Source Stack

| Project | Current role | Status |
| --- | --- | --- |
| OpenPI | Robot foundation policy / VLA backbone, using `pi05_libero` | active, smoke and scaled rollout jobs passing |
| LIBERO | Manipulation benchmark tasks and initial states | active, four suites logged |
| robosuite / MuJoCo | Headless simulation/rendering backend through LIBERO | active on `dualcard` with EGL; `midcard` had EGL-device initialization failures |
| VLM embeddings | Frozen image features for risk prediction | active offline and runtime ablation using `google/siglip-base-patch16-224`; runtime supervisor embeds the post-stressor initial RGB frame |
| World model / progress model | Transition/progress signals for timeout/no-progress risk | 10-step prefix action/no-progress/reward statistics active; learned predictive dynamics planned |
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

## Runtime SigLIP Supervisor Results

Runtime validation is now executed in the real OpenPI/LIBERO loop, not just replayed offline. Jobs `10133` through `10147` evaluated `libero_spatial` tasks `0..9`, seed `2000`, three trials per condition, and stressors `none:0.0`, `occlusion:0.4/0.6/0.8/1.0`, and `action_noise:0.4/0.6`.

| Runtime mode | Episodes | Coverage | Completion | Attempted completion | Failure among attempts | Timeout | Abstain | Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct OpenPI | 210 | 1.000 | 0.695 | 0.695 | 0.305 | 0.305 | 0.000 | 0.528 |
| fixed task prior selective | 210 | 1.000 | 0.686 | 0.686 | 0.314 | 0.314 | 0.000 | 0.514 |
| runtime SigLIP selective | 210 | 0.681 | 0.595 | 0.874 | 0.126 | 0.086 | 0.319 | 0.480 |

Interpretation: the runtime SigLIP risk signal transfers enough to cut attempted failures by more than half versus direct OpenPI. The original offline threshold is over-conservative, abstaining on `31.9%` of held-out episodes and lowering total completion/utility. The project now includes a task-disjoint runtime threshold sweep that chooses thresholds only on runtime calibration tasks `0..4` and evaluates once on test tasks `5..9`.

Runtime threshold sweep, test split:

| Calibration target | Test coverage | Completion | Attempted failure | Utility | Utility delta vs direct |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.70 | 0.686 | 0.629 | 0.083 | 0.528 | -0.043 |
| 0.75 | 0.781 | 0.714 | 0.085 | 0.627 | 0.056 |
| 0.80 | 0.810 | 0.714 | 0.118 | 0.618 | 0.047 |
| 0.85 | 0.857 | 0.724 | 0.156 | 0.617 | 0.046 |
| 0.90 | 0.933 | 0.724 | 0.224 | 0.592 | 0.021 |
| 0.95 | 0.962 | 0.724 | 0.248 | 0.583 | 0.012 |
| 1.00 | 1.000 | 0.724 | 0.276 | 0.571 | 0.000 |

Best utility operating point: target `0.75`, test utility `0.627`, coverage `0.781`, and attempted failure `0.085`. Best safety operating point: target `0.70`, attempted failure `0.083`. At at least `0.85` test coverage, target `0.85` keeps attempted failure at `0.156`, still well below direct OpenPI's `0.276` on the same test split.

The rollout script now supports `RUNTIME_RISK_THRESHOLD_OVERRIDE`, so the tuned threshold can be deployed without editing the risk summary. The best-utility threshold from the sweep is `0.9333276460818999`; the best >=85% coverage threshold is `0.9860334584902223`.

Fresh tuned-threshold deployment:

| Job | Threshold | Seed | Conditions | Episodes | Success | Timeout | Abstain | Utility |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10148 | 0.9333276460818999 | 3000 | tasks `5..9`, occlusion/action-noise severity `0.6` | 30 | 28 | 2 | 0 | 0.888 |

Job `10148` is a real deployment sanity check for the tuned threshold. It attempted all moderate-stress episodes and achieved `93.3%` completion. It should not be presented as a same-seed controlled comparison against direct OpenPI; it is evidence that the tuned threshold can run online without the over-abstention seen from the original offline threshold.

## Same-Seed Controlled Deployment

The controlled deployment closes the main evaluation gap from job `10148`: all modes are now run on the same task IDs, stress conditions, trial count, and seed. Jobs `10149..10168` use `libero_spatial` tasks `5..9`, seed `4000`, `5` trials per condition, and the stress grid `none`, `occlusion:0.60`, `occlusion:0.80`, `occlusion:1.00`, and `action_noise:0.60`.

| Runtime mode | Episodes | Coverage | Completion | Attempted completion | Attempted failure | Timeout | Abstain | Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct OpenPI | 125 | 1.000 | 0.656 | 0.656 | 0.344 | 0.344 | 0.000 | 0.469 |
| fixed task prior selective | 125 | 1.000 | 0.648 | 0.648 | 0.352 | 0.352 | 0.000 | 0.457 |
| SigLIP selective, threshold `0.9333276460818999` | 125 | 0.680 | 0.584 | 0.859 | 0.141 | 0.096 | 0.320 | 0.463 |
| SigLIP selective, threshold `0.9860334584902223` | 125 | 0.800 | 0.616 | 0.770 | 0.230 | 0.184 | 0.200 | 0.473 |

Controlled result:

- The higher-coverage SigLIP threshold `0.9860334584902223` is the best deployable tradeoff in this run: utility improves by `0.004` over direct OpenPI while attempted failure drops by `0.114` absolute.
- The lower threshold `0.9333276460818999` is safer but more selective: attempted failure drops by `0.203` absolute, with `0.680` coverage and utility `0.006` below direct OpenPI.
- Both SigLIP thresholds beat random abstention at matched coverage on utility and attempted-failure rate, so the result is not explained by rejecting arbitrary episodes.
- The supervisor behaves as intended under severe occlusion: at `occlusion:1.00`, threshold `0.9333` abstains on all episodes and threshold `0.9860` abstains on `96%`.

This is now a meaningful online runtime-supervision result: the VLM risk model rejects distribution-shifted states in the real OpenPI/LIBERO loop and improves reliability at a tunable coverage/utility point. It is still a simulation result, not a formal safety guarantee.

## Multiseed And Cross-Suite Deployment

The follow-on deployment keeps the same runtime model and thresholds, adds more seeds for `libero_spatial`, and tests `libero_object` plus `libero_goal`. It uses jobs `10180..10248`, for `1,950` new online episodes.

Spatial pooled result over seeds `5000`, `6000`, and `7000`:

| Runtime mode | Episodes | Coverage | Completion | Attempted failure | Abstain | Utility | Utility delta CI | Failure delta CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct OpenPI | 375 | 1.000 | 0.661 | 0.339 | 0.000 | 0.477 | - | - |
| fixed task prior selective | 375 | 1.000 | 0.656 | 0.344 | 0.000 | 0.469 | -0.008 [-0.048, 0.032] | 0.005 [-0.021, 0.032] |
| SigLIP `0.9333` | 375 | 0.667 | 0.597 | 0.104 | 0.333 | 0.487 | 0.010 [-0.041, 0.056] | -0.235 [-0.286, -0.183] |
| SigLIP `0.9860` | 375 | 0.803 | 0.637 | 0.206 | 0.197 | 0.504 | 0.027 [-0.029, 0.076] | -0.133 [-0.181, -0.083] |

Cross-suite result on `libero_object` and `libero_goal`:

| Runtime mode | Episodes | Coverage | Completion | Attempted failure | Abstain | Utility | Utility delta CI | Failure delta CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct OpenPI | 150 | 1.000 | 0.733 | 0.267 | 0.000 | 0.583 | - | - |
| SigLIP `0.9333` | 150 | 0.693 | 0.627 | 0.096 | 0.307 | 0.522 | -0.061 [-0.130, 0.009] | -0.171 [-0.249, -0.093] |
| SigLIP `0.9860` | 150 | 0.833 | 0.660 | 0.208 | 0.167 | 0.526 | -0.056 [-0.120, 0.006] | -0.059 [-0.110, -0.008] |

Interpretation:

- `0.9860` remains the best deployable threshold on `libero_spatial`: it has the best utility point estimate and robust attempted-failure reduction.
- The utility gain is fragile, not proven: the bootstrap CI for utility delta still crosses zero.
- `0.9333` should be framed as safety mode: it gives the largest attempted-failure reduction but rejects about one third of spatial episodes.
- Cross-suite generalization holds for risk filtering, not utility. Both SigLIP thresholds reduce attempted failure on `libero_object`/`libero_goal`, but direct OpenPI retains higher utility because rejection costs outweigh the saved failures in this grid.

## Risk Critic Checkpoint

Final audited dataset split:

| Split | Episodes | Failures | Failure rate |
| --- | ---: | ---: | ---: |
| train | 595 | 104 | 0.175 |
| calibration | 197 | 34 | 0.173 |
| test | 201 | 36 | 0.179 |
| all | 993 | 174 | 0.175 |

Runtime-loadable structured model: `structured_progress_risk`, which excludes injected stressor metadata.
Strongest observed-image ablation: `vision_language_risk`, which also excludes injected stressor metadata and uses frozen SigLIP first-frame embeddings extracted from rollout videos.

| Test model | AUROC | AUPRC | Brier | ECE | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| global prior | 0.500 | 0.179 | 0.147 | 0.004 | Constant risk; AUPRC equals positive rate after tie audit. |
| fixed task prior | 0.695 | 0.347 | 0.139 | 0.062 | Strong baseline because task identity is informative. |
| structured progress risk | 0.702 | 0.297 | 0.238 | 0.315 | Slightly higher AUROC than fixed prior, weaker calibration/AUPRC. |
| vision language risk | 0.905 | 0.811 | 0.136 | 0.235 | Frozen SigLIP first-frame embedding ablation; observed images replace hidden stressor metadata. |
| metadata oracle risk | 0.930 | 0.840 | 0.146 | 0.264 | Diagnostic upper bound; sees stressor metadata. |

At about 90% coverage on the test split:

| Policy | Coverage | Task completion | Failure among attempts | Utility | Query overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct OpenPI | 1.000 | 0.821 | 0.179 | 0.716 | 1.000 |
| fixed task prior selective | 0.900 | 0.771 | 0.144 | 0.673 | 0.883 |
| structured progress selective | 0.900 | 0.756 | 0.160 | 0.651 | 0.898 |
| vision language selective | 0.900 | 0.821 | 0.088 | 0.748 | 0.856 |
| metadata oracle selective | 0.900 | 0.821 | 0.088 | 0.748 | 0.856 |
| adaptive chunk offline | 1.000 | 0.821 | 0.179 | 0.716 | 0.997 |
| early abort on no-progress offline | 0.970 | 0.796 | 0.179 | 0.689 | 0.944 |
| adaptive chunk plus abort offline | 0.970 | 0.796 | 0.179 | 0.689 | 0.941 |

The adaptive/abort rows above are offline counterfactuals from logged episodes, not resimulated robot executions. Interpretation: there is exploitable risk structure. The structured-only model is not strong enough to beat fixed priors across all metrics, but the frozen SigLIP observed-image ablation recovers most of the metadata-oracle signal without using hidden stressor labels. The VLM signal has now been moved into the runtime supervisor; the remaining research step is improving calibration/coverage and adding a true learned predictive dynamics/world-model head.

## VLM And World-Model Integration

OpenPI is already the active VLA policy: it consumes RGB observations and language prompts and outputs robot actions. The risk layer currently wraps that policy using structured rollout features, and now includes an offline frozen-VLM image-risk ablation. The VLM/world-model path is concrete:

1. `scripts/extract_openpi_siglip_embeddings.py` decodes the first frame from each logged rollout video and embeds it with frozen `google/siglip-base-patch16-224`.
2. `vision_language_risk` trains on those frozen image embeddings plus observable structured/progress features, with stressor metadata removed.
3. `vision_language_risk_selective` computes the same SigLIP feature at runtime, logs the frame id/path, predicts risk from the runtime observation plus prefix statistics, and rejects episodes above the calibration threshold.
4. `SAVE_IMAGES=1` in `slurm/openpi_libero_rollouts.sbatch` supports per-step RGB frame logging for future temporal VLM features.
5. Prefix action norms, no-progress scores, action smoothness, and reward are already used as lightweight progress/world-model proxy features.
6. The next model should compare fixed priors, structured progress, frozen VLM embeddings, and learned predictive dynamics at matched coverage under actual runtime execution.

Feasibility check from this run: Hugging Face access is available, SigLIP loads through `transformers`, and the existing MP4 rollout videos are decodable through the OpenPI LIBERO venv. The generated embedding artifact is intentionally ignored under `outputs/openpi_libero/`; reproduce it with:

```bash
PYTHONPATH=src python scripts/extract_openpi_siglip_embeddings.py --config configs/openpi/train_risk.yaml --output outputs/openpi_libero/siglip_episode_embeddings.jsonl --dims 64
PYTHONPATH=src python scripts/train_openpi_risk.py --config configs/openpi/train_risk.yaml
```

## Verification

- `python scripts/audit_openpi_metrics.py --risk-summary reports/openpi_libero_risk_summary.json --report reports/openpi_libero_risk_planning.md` passes with no failures.
- The audit recomputes split metrics and thresholds from raw JSONL.
- The audit also recomputes the trained `vision_language_risk` metrics from the local SigLIP embedding artifact.
- Supervisor/non-direct runs are excluded from training inputs.
- Runtime supervisor evaluation and the task-disjoint runtime threshold sweep are summarized in `reports/openpi_runtime_siglip_eval_summary.json`.
- Same-seed controlled deployment is summarized in `reports/openpi_runtime_controlled_deployment_summary.json`.
- Multiseed and cross-suite deployment is summarized in `reports/openpi_runtime_multiseed_summary.json`.
- The global-prior AUPRC tie-handling bug is fixed and audited.
- `python -m pytest -q` is the final repo regression check.

## Resume Framing

Use this phrasing:

> Built a risk-aware execution layer for OpenPI robot foundation policies on LIBERO, including SLURM rollout infrastructure, stress-test generation, calibrated failure-risk prediction, frozen SigLIP image-risk models, runtime selective rejection, task-disjoint threshold tuning, adaptive replanning hooks, and coverage-aware evaluation over 993 audited offline training/evaluation rollouts plus 3,080 held-out online comparison episodes.

Do not say the project has solved robot safety or improves utility robustly across suites. The professional framing is calibrated risk-aware supervision for brittle robot foundation policy execution, with robust attempted-failure reduction under controlled online stress tests.
