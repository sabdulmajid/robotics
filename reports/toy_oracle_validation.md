# Toy Oracle Validation

This note summarizes the current Milestone 0 gate result.

Command:

```bash
python -m risk_aware_skill_planning.cli toy-eval --config configs/toy_oracle_validation.yaml
```

Configuration:

- `500` episodes per scenario and planner mode.
- Seeds `0..499`.
- Planner modes: `naive_no_risk`, `fixed_per_skill_risk`, `oracle_risk`.
- Default planner thresholds: `next_skill_threshold=0.85`, `plan_threshold=0.85`.
- Rejection is enabled, but all evaluated default runs are attempted so comparisons are at equal coverage.
- Regression acceptance threshold: `oracle_risk` must improve task completion by more than `0.25` and reduce catastrophic failure by more than `0.15` relative to both baselines on each frozen scenario.

| Scenario | Planner | Task completion | Catastrophic failure | Coverage | Rejection |
| --- | --- | ---: | ---: | ---: | ---: |
| `direct_pick_blocked_by_distractor` | `naive_no_risk` | 0.224 (0.184-0.262) | 0.482 (0.444-0.530) | 1.000 | 0.000 |
| `direct_pick_blocked_by_distractor` | `fixed_per_skill_risk` | 0.224 (0.184-0.262) | 0.482 (0.444-0.530) | 1.000 | 0.000 |
| `direct_pick_blocked_by_distractor` | `oracle_risk` | 0.724 (0.680-0.772) | 0.076 (0.056-0.102) | 1.000 | 0.000 |
| `far_bin_high_drop_risk` | `naive_no_risk` | 0.338 (0.292-0.382) | 0.332 (0.294-0.382) | 1.000 | 0.000 |
| `far_bin_high_drop_risk` | `fixed_per_skill_risk` | 0.338 (0.292-0.382) | 0.332 (0.294-0.382) | 1.000 | 0.000 |
| `far_bin_high_drop_risk` | `oracle_risk` | 0.730 (0.692-0.766) | 0.092 (0.068-0.114) | 1.000 | 0.000 |

Gate status:

```text
PASS: oracle-risk planning beats naive and fixed-risk planning on both frozen toy scenarios.
```

Scope note:

This validates exploitable state-conditioned risk structure in the toy harness. It does not claim learned risk calibration, manipulation transfer, or formal safety.

Per-episode candidate-plan logs can be regenerated with:

```bash
python -m risk_aware_skill_planning.cli toy-trace \
  --scenario direct_pick_blocked_by_distractor \
  --planner-mode oracle_risk \
  --seed 0 \
  --output outputs/toy_trace_direct_pick_oracle.json
```
