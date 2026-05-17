# OpenPI/LIBERO Workstream Status

Updated: 2026-05-17

| Workstream | Owner equivalent | Status | Evidence |
| --- | --- | --- | --- |
| OpenPI/LIBERO integration | A | Complete for current scope | `pi05_libero` official smoke passed; OpenPI commit `c23745b5ad24e98f66967ea795a07b2588ed6c79`; `reports/openpi_libero_official_eval_smoke.md`. |
| Rollout instrumentation | B | Complete for current scope | `scripts/openpi_libero_single_task_eval.py` logs episode/step JSONL, stressors, videos, risk fields, and supervisor decisions. |
| Risk critic + features | C | Preliminary complete | `reports/openpi_libero_risk_summary.json`; 30 direct rollouts, calibrated logistic risk baseline, held-out metrics. |
| Risk-aware supervisor | D | Preliminary complete | selective run `10097`; adaptive chunking run `10098`; both logged through real OpenPI/LIBERO loop. |
| Experiments + report | E | Current checkpoint complete, not final benchmark | `reports/openpi_project_status.md` and `reports/openpi_libero_risk_planning.md` contain numeric results, limitations, and reproduction commands. Cross-suite direct smoke `10099` also passed on `libero_object`, `libero_goal`, and `libero_10` task 0. |

Next high-value step: collect more direct rollouts across LIBERO Object and Goal, then add frozen VLM/image-language embeddings and a progress/world-model feature head.
