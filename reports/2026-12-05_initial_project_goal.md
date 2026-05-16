/goal

You are in repo `sabdulmajid/robotics`.

This is no longer a toy-polishing project.

Mission:

  Produce a serious OpenPI/LIBERO experimental result using the available GPUs.

The project must become:

  Risk-Aware Execution for OpenPI Robot Foundation Policies

Core research question:

  Can calibrated risk prediction improve the reliability / coverage tradeoff of
  OpenPI pi0.5 execution on LIBERO by deciding when to execute normally, shorten
  the action horizon, retry/replan, or abstain?

Main deliverable:

  A reproducible OpenPI + LIBERO benchmark report showing direct OpenPI versus
  risk-aware supervised OpenPI execution, with real rollout data, trained risk
  critics, calibration metrics, ablations, and honest limitations.

Do not spend the main effort on mocks.
Do not stop after writing wrappers.
Do not stop after writing docs.
Get experiments.

===============================================================================
NON-NEGOTIABLES
===============================================================================

1. Use OpenPI as the main policy source.
2. Use LIBERO as the main benchmark.
3. Use the official OpenPI LIBERO-compatible checkpoint/config where possible:
   `pi05_libero`, checkpoint path such as `gs://openpi-assets/checkpoints/pi05_libero/`.
4. All GPUs available (I refer to /mnt/slurm_nfs/a6abdulm/cluster_instructions.md for more details and always check before submitting) aggressively for rollout collection, embedding extraction, risk training, calibration, and ablations.
5. Preserve existing toy tests, but do not make toy work the focus.
6. No formal safety claims.
7. No real-world deployment claims.
8. No "we integrated OpenPI" claim unless OpenPI actually runs.
9. If an external dependency blocks execution, implement up to the runnable boundary,
   document the exact blocker, and provide the exact command to resume.
10. The final report must contain numbers, not just architecture.

===============================================================================
USE SUBAGENTS / PARALLEL WORKSTREAMS
===============================================================================

If the coding environment supports subagents, launch them immediately.

Use these subagents:

SUBAGENT A - OpenPI/LIBERO Integration
- Install / validate OpenPI.
- Validate LIBERO setup.
- Run official OpenPI LIBERO smoke eval.
- Identify exact OpenPI policy-server or direct-inference path.
- Own configs and environment docs.

SUBAGENT B - Rollout Instrumentation
- Modify/wrap OpenPI LIBERO eval to log episodes and steps.
- Capture observations, actions/action chunks, task language, success, timeout,
  no-progress, and metadata.
- Ensure logs are valid JSONL.
- Own dataset generation.

SUBAGENT C - Risk Critic + Features
- Build rollout dataset loader.
- Extract features:
  - task id / suite / language embedding if available
  - timestep / progress features
  - action chunk statistics
  - action norm / smoothness / disagreement across replans
  - image embeddings if feasible
  - OpenPI internal embeddings if accessible without huge surgery
- Train risk models.
- Calibrate probabilities.
- Own metrics: Brier, NLL, ECE, AUROC, AUPRC, risk-coverage.

SUBAGENT D - Risk-Aware Supervisor
- Implement execution variants:
  - direct OpenPI
  - fixed prior
  - learned risk
  - selective abstention
  - adaptive action-horizon supervisor
  - no-progress recovery/replan if feasible
- Own evaluation of interventions.

SUBAGENT E - Experiments + Report
- Run full experiments across GPUs.
- Aggregate results.
- Generate figures/tables.
- Write `reports/openpi_libero_risk_planning.md`.
- Keep claims honest and impressive.

If no subagent tool exists, emulate subagents as parallel workstreams with a concise
status file:

  outputs/openpi_libero/status.md

Do not wait for perfect architecture. Work toward results.

===============================================================================
CORE IDEA TO IMPLEMENT
===============================================================================

OpenPI pi0.5 commonly executes action chunks. A direct baseline uses a fixed action
horizon, often `n_action_steps=10`.

The risk-aware supervisor should test whether changing execution based on predicted
risk improves reliability.

Implement these execution modes:

1. `direct_openpi`
   - Run OpenPI normally.
   - Fixed `n_action_steps=10` unless official config says otherwise.

2. `fixed_task_prior`
   - Estimate failure risk from train split per task/suite.
   - Use that risk for abstention or reporting.

3. `learned_risk_openpi`
   - Predict failure probability from rollout state/task/action/image features.
   - Does not intervene, only evaluates prediction quality.

4. `selective_openpi`
   - Predict risk before or during execution.
   - If calibrated risk > threshold, abstain / stop early.
   - Report coverage and failure-at-coverage.

5. `adaptive_chunk_openpi`
   - If predicted risk is low: execute normal action horizon, e.g. 10.
   - If predicted risk is medium: shorten action horizon, e.g. 5.
   - If predicted risk is high: query OpenPI every 1-2 steps.
   - If risk remains extreme or no-progress persists: abstain / stop.
   - This is the key practical intervention.

6. `no_progress_replan`
   - Detect no progress from object/eef/state/task features when available.
   - Reset policy hidden/episode state if applicable.
   - Re-query OpenPI with shorter action horizon.
   - Count recovery attempts and recovery success.

Primary claim to test:

  "Calibrated risk prediction enables better failure/coverage tradeoffs and
  adaptive action-horizon execution around OpenPI on LIBERO."

===============================================================================
GPU UTILIZATION PLAN
===============================================================================

Use all GPUs.

Required behavior:

- Detect GPUs at runtime using SLURM (shared cluster)
- Log CUDA device names, memory, driver/CUDA/PyTorch/JAX versions.

Then run rollout collectors in parallel IF POSSIBLE.

===============================================================================
EXPERIMENTAL PLAN
===============================================================================

Do experiments in this order.

PHASE 1 - Official OpenPI/LIBERO smoke
Goal:
  Prove OpenPI pi0.5 LIBERO runs.

Run:
- one suite
- one task
- 1-3 episodes
- log observations/actions/outcomes

Acceptance:
- actual OpenPI actions are produced.
- LIBERO environment steps.
- episode outcome is logged.

PHASE 2 - Direct OpenPI baseline
Goal:
  Establish baseline performance.

Run initially:
- LIBERO Spatial
- LIBERO Object
- LIBERO Goal
- LIBERO-10 / Long if available

Recommended:
- 10 episodes per task if time permits.
- If compute/time is tight, start with 3 episodes per task, then scale to 10.

Log:
- success rate
- timeout rate
- average episode length
- failure cases
- action horizon
- checkpoint
- seed/task split

Acceptance:
- direct OpenPI baseline report exists.

PHASE 3 - Failure-rich stress suite
Goal:
  Get enough failures to train and evaluate risk critics.

Do not rely only on near-perfect official LIBERO performance.

Create controlled stress conditions that are scientifically honest:

Stress axes:
1. observation corruption
   - brightness shift
   - Gaussian noise
   - blur
   - random crop
   - partial occlusion rectangle
2. control perturbation
   - action noise
   - action delay
   - reduced action precision
3. initial-state / object pose perturbation if LIBERO permits it
4. action horizon variation
   - n_action_steps = 1, 2, 5, 10, 15
5. camera/view perturbation if available

Important:
- Do not present stress results as standard LIBERO leaderboard results.
- Clearly label them as robustness/stress evaluations.
- Use them to create meaningful risk-learning data.

Target dataset:
- minimum useful: 300-500 episodes
- better: 1,000-2,000 episodes
- stretch: 3,000+ episodes across suites/stressors/horizons

Use all GPUs to collect these.

PHASE 4 - Risk critic training
Goal:
  Predict failure / timeout / no-progress before or during execution.

Targets:
- `p_failure`
- `p_timeout`
- `p_no_progress`
- optionally `p_success`

Models:
1. logistic regression baseline
2. small MLP
3. calibrated MLP
4. optional frozen image encoder + MLP
5. optional OpenPI embedding/action-feature risk head if accessible

Features:
- suite id
- task id
- language embedding or hashed task/language features
- timestep
- episode progress
- end-effector/object state if available
- action chunk mean/std/norm
- action chunk smoothness
- difference between current and previous action chunks
- no-progress indicators
- image embedding if feasible
- corruption/stress metadata

Calibration:
- train / calibration / test split
- scenario/task/stress-disjoint split where possible
- thresholds selected only on calibration split

Metrics:
- Brier
- NLL
- ECE
- AUROC
- AUPRC
- reliability diagram
- risk-coverage curve
- failure-at-coverage

PHASE 5 - Risk-aware execution
Goal:
  Actually intervene, not just predict.

Evaluate:
- direct OpenPI
- fixed prior
- learned risk, no intervention
- selective OpenPI
- adaptive action-horizon OpenPI
- no-progress replan if feasible

Primary metrics:
- success rate
- failure rate
- timeout rate
- coverage
- rejection/abstention rate
- success at matched coverage
- failure at matched coverage
- expected utility
- mean episode length
- compute/runtime overhead
- bootstrap confidence intervals

Key plot:
- failure rate vs coverage
- success rate vs coverage
- direct OpenPI vs adaptive horizon
- calibration reliability
- stressor-specific breakdown

PHASE 6 - Optional high-impact extension if time remains
Only after phases 1-5 have working results:

A. compare OpenPI vs LeRobot pi0.5 LIBERO wrapper/checkpoint if easy
B. train an image-feature risk critic using frozen DINO/SigLIP/CLIP features
C. evaluate held-out stressors for OOD generalization
D. create short videos/traces of high-risk failures and risk-aware interventions

Do not derail the main OpenPI/LIBERO result.

===============================================================================
IMPLEMENTATION DELIVERABLES
===============================================================================

Add or modify:

  risk_aware_skill_planning/backends/openpi/
    __init__.py
    config.py
    policy.py
    libero_runner.py
    rollout_logger.py
    stressors.py
    action_horizon.py

  risk_aware_skill_planning/risk/
    openpi_features.py
    openpi_dataset.py
    openpi_models.py
    openpi_calibration.py

  risk_aware_skill_planning/evaluation/
    openpi_metrics.py
    risk_coverage.py
    bootstrap.py

  risk_aware_skill_planning/supervision/
    openpi_supervisor.py
    adaptive_chunking.py
    no_progress.py

  configs/openpi/
    libero_smoke.yaml
    libero_collect_baseline.yaml
    libero_collect_stress.yaml
    train_risk.yaml
    eval_risk.yaml
    eval_supervisor.yaml

  scripts/
    openpi_libero_smoke.py
    collect_openpi_libero.py
    train_openpi_risk.py
    eval_openpi_supervisor.py
    summarize_openpi_results.py

  docs/
    openpi_libero_setup.md
    openpi_experiment_protocol.md

  reports/
    openpi_libero_risk_planning.md
    openpi_libero_risk_summary.json

Use the existing CLI if clean. Otherwise scripts are acceptable. Results matter.

===============================================================================
ROLLING LOG SCHEMA
===============================================================================

Every episode log must include:

- run_id
- timestamp
- git_sha
- hostname
- gpu_id
- cuda_device_name
- openpi_repo_path
- openpi_commit if available
- checkpoint_path
- openpi_config_name
- libero_suite
- libero_task_id
- libero_task_name
- language_instruction
- seed
- episode_id
- stressor_name
- stressor_params
- n_action_steps
- policy_backend = openpi
- success
- timeout
- failure_label
- episode_length
- total_reward if available
- terminal_reason
- video_path if saved
- metadata

Every step log must include:

- run_id
- episode_id
- timestep
- observation_summary
- image_path or image_id if saved
- action_summary
- action_chunk_summary
- selected_action_index
- n_action_steps
- predicted_risk if available
- calibrated_risk if available
- supervisor_decision
- no_progress_score if available
- reward
- done
- info_summary

Use JSONL.

===============================================================================
FAILURE LABELS
===============================================================================

Minimum labels:

- success
- task_failure
- timeout
- no_progress
- possible_wrong_goal
- possible_collision_or_unsafe_contact
- policy_invalid_action
- abstained
- unknown_failure

If LIBERO does not expose enough signals for precise collision/wrong-goal labels,
mark them as approximate and keep `unknown_failure`.

Do not fake labels.

===============================================================================
RISK SUPERVISOR DETAILS
===============================================================================

Adaptive action horizon policy:

Inputs:
- calibrated p_failure
- calibrated p_timeout
- no_progress_score
- timestep
- task/suite id
- recent action chunk disagreement

Decision:

if p_failure < low_threshold:
    n_action_steps = 10
elif p_failure < medium_threshold:
    n_action_steps = 5
elif p_failure < high_threshold:
    n_action_steps = 2
else:
    n_action_steps = 1 or abstain

If no_progress_score stays high for K consecutive windows:
    reset/requery policy if possible
    reduce n_action_steps
    count recovery attempt

Thresholds:
- choose on calibration split only
- report chosen thresholds
- do not tune on test

Expected utility example:

  utility =
      +1.0 * success
      -1.0 * task_failure
      -0.5 * timeout
      -0.2 * abstained
      -0.01 * episode_length_penalty
      -0.02 * extra_policy_queries

Report both raw success and utility. Do not hide tradeoffs.

===============================================================================
ACCEPTANCE CRITERIA
===============================================================================

The goal is successful only if all are true:

1. Existing repo tests still pass.
2. OpenPI/LIBERO setup path is documented.
3. OpenPI policy is actually invoked.
4. At least one real OpenPI/LIBERO episode is collected.
5. Direct OpenPI baseline is run and logged.
6. Risk dataset is built from OpenPI/LIBERO logs.
7. At least one risk critic is trained and calibrated.
8. Risk metrics are reported on held-out data.
9. At least one risk-aware supervisor mode is evaluated.
10. Final report contains numeric results and exact reproduction commands.

Stretch acceptance:

11. Baseline collected across all standard LIBERO suites.
12. Stress suite collected with 500+ episodes.
13. Risk critic achieves nontrivial AUROC/AUPRC.
14. Selective/adaptive supervisor improves failure-at-coverage or utility.
15. Results include confidence intervals.

===============================================================================
TESTING
===============================================================================

Unit tests must not require OpenPI or LIBERO:

- config parsing
- JSONL log validation
- stressor config validation
- risk dataset loading from tiny synthetic logs
- calibration threshold selection
- supervisor decision logic
- clear error when OpenPI/LIBERO/checkpoint missing

Integration tests may require OpenPI/LIBERO and should be marked:

  pytest -m openpi
  pytest -m libero
  pytest -m gpu
  pytest -m integration

Run at minimum:

  python -m pytest

Then, in the real compute environment:

  nvidia-smi

  python scripts/openpi_libero_smoke.py \
    --config configs/openpi/libero_smoke.yaml

  python scripts/collect_openpi_libero.py \
    --config configs/openpi/libero_collect_baseline.yaml

  python scripts/collect_openpi_libero.py \
    --config configs/openpi/libero_collect_stress.yaml

  python scripts/train_openpi_risk.py \
    --config configs/openpi/train_risk.yaml

  python scripts/eval_openpi_supervisor.py \
    --config configs/openpi/eval_supervisor.yaml

  python scripts/summarize_openpi_results.py \
    --run-dir outputs/openpi_libero/<RUN_ID>

If commands differ because the repo CLI is better, use the better commands and
document exact invocations.

===============================================================================
REPORT REQUIREMENTS
===============================================================================

Create:

  reports/openpi_libero_risk_planning.md
  reports/openpi_libero_risk_summary.json

The report must include:

1. Hardware used.
2. OpenPI version/commit.
3. Checkpoint/config used.
4. LIBERO suites/tasks.
5. Episode counts.
6. Direct OpenPI baseline results.
7. Stress protocol.
8. Risk model features.
9. Calibration method.
10. Risk prediction metrics.
11. Supervisor intervention policy.
12. Direct vs risk-aware results.
13. Coverage/failure tradeoff.
14. Runtime overhead.
15. Failure examples.
16. Limitations.
17. Exact reproduction commands.

Interview-ready summary should be included at the top:

  "I integrated OpenPI pi0.5 with LIBERO, collected N real VLA simulation rollouts
  on X GPUs, trained a calibrated risk critic over policy/action/vision/task
  features, and evaluated selective/adaptive action-horizon supervision against
  direct OpenPI execution."

Only fill this sentence with actual N and actual results.

===============================================================================
WHAT NOT TO DO
===============================================================================

Do not:
- spend days on mock backends
- make OpenPI optional in this milestone
- stop at wrappers
- stop at docs
- claim safety
- hide low coverage
- tune thresholds on test
- fabricate collision/wrong-goal labels
- claim standard LIBERO improvements if using stress conditions
- vendor huge model weights
- commit datasets/checkpoints/videos unless tiny and intentional

===============================================================================
FALLBACKS
===============================================================================

If OpenPI Docker works:
- use Docker path first.

If Docker fails but local Python works:
- use local path.

If official OpenPI checkpoint download fails:
- document exact failure.
- support local checkpoint path.
- continue building runnable scripts.

If full all-suite evaluation is too slow:
- run one suite first.
- then stress subset.
- then scale.

If official OpenPI success is too high and failures are rare:
- use stress protocol.
- clearly label as robustness/stress evaluation.

If image embeddings are too slow:
- start with action/task/state features.
- add image embeddings later.

If adaptive horizon does not improve success:
- still report risk-coverage and abstention tradeoff honestly.
- analyze failure cases.

===============================================================================
FINAL RESPONSE FORMAT
===============================================================================

When finished, report only:

1. Actual OpenPI/LIBERO result achieved.
2. Number of episodes collected.
3. GPU usage.
4. Files changed.
5. Commands run.
6. Tests passed/failed.
7. Main metrics:
   - direct OpenPI success/failure
   - risk AUROC/AUPRC/ECE/Brier
   - selective/adaptive supervisor result
8. Exact blockers, if any.
9. Next highest-impact experiment.

No long essay.