# OpenPI/LIBERO Official Eval Smoke

Status: `READY_TO_RUN`

This is the first real-policy milestone after the setup smoke. It starts OpenPI's
official `pi05_libero` policy server, runs one filtered LIBERO-Spatial task, and
writes the same JSONL rollout schema used by the risk supervisor and future
VLM/world-model feature extractors.

## Command

```bash
sbatch slurm/openpi_libero_official_smoke.sbatch
```

## Expected Artifacts

- `datasets/openpi_libero_rollouts/openpi_libero_official_smoke_<job>.jsonl`
- `videos/openpi_libero_official_smoke_<job>/`
- `reports/openpi_libero_official_smoke_summary.json`
- `outputs/openpi_libero/policy_server_<job>.log`

## Scope

This smoke is intentionally small: `libero_spatial`, task `0`, one trial. It is
not a final benchmark result. Passing this smoke means the project is executing
real OpenPI actions in LIBERO and collecting risk-ready episode logs.
