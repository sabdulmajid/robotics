# Toy Learned Risk Validation

This note reports the first learned-risk sanity check in the toy domain.

Scope: these models are trained on stochastic toy rollouts. They are not robosuite policies or manipulation risk critics.

Risk threshold for selective metrics: `0.35`.

## Skill-Level Risk Metrics

| Model | Brier | NLL | ECE | AUROC | AUPRC | Coverage @ threshold | Selective success | FNR @ threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `global_prior` | 0.249 | 0.690 | 0.005 | 0.500 | 0.471 | 0.000 | 0.000 | 0.000 |
| `per_skill_prior` | 0.185 | 0.534 | 0.027 | 0.733 | 0.620 | 0.260 | 0.965 | 0.019 |
| `logistic_state_risk` | 0.125 | 0.415 | 0.161 | 0.933 | 0.907 | 0.338 | 0.944 | 0.041 |
| `calibrated_logistic_state_risk` | 0.100 | 0.342 | 0.064 | 0.933 | 0.907 | 0.449 | 0.938 | 0.060 |
| `oracle_true_risk` | 0.074 | 0.265 | 0.010 | 0.947 | 0.941 | 0.513 | 0.930 | 0.078 |

## Planner Impact Check

| Scenario | Planner | Task completion | Catastrophic failure | Coverage | Rejection |
| --- | --- | ---: | ---: | ---: | ---: |
| `direct_pick_blocked_by_distractor` | `naive_no_risk` | 0.222 | 0.480 | 1.000 | 0.000 |
| `direct_pick_blocked_by_distractor` | `per_skill_prior` | 0.726 | 0.078 | 1.000 | 0.000 |
| `direct_pick_blocked_by_distractor` | `calibrated_logistic_state_risk` | 0.726 | 0.078 | 1.000 | 0.000 |
| `direct_pick_blocked_by_distractor` | `oracle_true_risk` | 0.726 | 0.078 | 1.000 | 0.000 |
| `far_bin_high_drop_risk` | `naive_no_risk` | 0.336 | 0.332 | 1.000 | 0.000 |
| `far_bin_high_drop_risk` | `per_skill_prior` | 0.336 | 0.332 | 1.000 | 0.000 |
| `far_bin_high_drop_risk` | `calibrated_logistic_state_risk` | 0.728 | 0.094 | 1.000 | 0.000 |
| `far_bin_high_drop_risk` | `oracle_true_risk` | 0.728 | 0.094 | 1.000 | 0.000 |

Interpretation:

- The calibrated state-conditioned logistic model improves Brier score, NLL, AUROC, and AUPRC over the state-independent per-skill prior.
- Temperature scaling improves Brier, NLL, and ECE over the uncalibrated logistic model in this seeded run.
- In planning, the per-skill prior helps in the blocked-pick scenario because the average direct-pick risk is high, but it fails in the far-target scenario because fast placement is only risky in the far/holding state. The calibrated state-risk model chooses the safer behavior in both frozen scenarios.
