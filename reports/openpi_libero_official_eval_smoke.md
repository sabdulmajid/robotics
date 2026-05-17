# OpenPI/LIBERO Official Eval Smoke

Status: `PASS`

This is the first real-policy milestone after the setup smoke. It starts
OpenPI's official `pi05_libero` policy server, runs one filtered LIBERO-Spatial
task, and writes the same JSONL rollout schema used by the risk supervisor and
future VLM/world-model feature extractors.

## Result

SLURM job: `10092`

Node: `ece-nebula10` (`dualcard`, `gpu:1`)

Policy: `pi05_libero`

Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`

Task: `libero_spatial`, task `0`

Language: `pick up the black bowl between the plate and the ramekin and place it on the plate`

Outcome: `success`

Summary:

```json
{
  "episodes": 1,
  "successes": 1,
  "success_rate": 1.0,
  "mean_episode_steps": 101,
  "timeouts": 0
}
```

## Command

```bash
sbatch slurm/openpi_libero_official_smoke.sbatch
```

## Expected Artifacts

- `datasets/openpi_libero_rollouts/openpi_libero_official_smoke_10092.jsonl`
- `videos/openpi_libero_official_smoke_10092/rollout_libero_spatial_task00_ep000_success.mp4`
- `reports/openpi_libero_official_smoke_summary.json`
- `outputs/openpi_libero/policy_server_10092.log`

Tracked GitHub review artifacts:

- `reports/artifacts/openpi_libero_official_smoke_10092.jsonl`
- `reports/artifacts/openpi_libero_official_smoke_10092_success.mp4`

## Scope

This smoke is intentionally small: `libero_spatial`, task `0`, one trial. It is
not a final benchmark result. Passing this smoke means the project is executing
real OpenPI actions in LIBERO and collecting risk-ready episode logs.

## Notes

- The first run installed OpenPI's Python 3.11 server environment, downloaded
  the 11.6 GiB `pi05_libero` checkpoint, and cached it under
  `checkpoints/openpi_cache`.
- The OpenPI server ran through the actual websocket policy path.
- The LIBERO client ran under OpenPI's Python 3.8 LIBERO environment.
- Websocket handshake errors in the server log came from the readiness probe
  opening a raw socket before the client connected; the actual OpenPI client
  connection opened and closed normally.
