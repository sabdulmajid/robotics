# OpenPI/LIBERO Smoke Status

Experiment: `openpi_libero_smoke`
OpenPI root: `external/openpi`
Policy config: `pi05_libero`
Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero/`
LIBERO suite/task: `libero_spatial` / `0`

Status: `PASS`

## Resume Commands

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git external/openpi
cd external/openpi
git submodule update --init --recursive
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
uv run scripts/serve_policy.py --env LIBERO
MUJOCO_GL=egl python examples/libero/main.py --task-suite-name libero_spatial
```

## Raw Status

```json
{
  "blockers": [],
  "command_status": {
    "docker": false,
    "git": true,
    "nvidia-smi": true,
    "uv": true
  },
  "config": {
    "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero/",
    "client_command": [
      "python",
      "examples/libero/main.py"
    ],
    "default_action_horizon": 10,
    "episodes": 1,
    "experiment_id": "openpi_libero_smoke",
    "openpi_repo_url": "https://github.com/Physical-Intelligence/openpi.git",
    "openpi_root": "external/openpi",
    "policy_config": "pi05_libero",
    "report_path": "reports/openpi_libero_smoke.md",
    "seed": 0,
    "server_command": [
      "uv",
      "run",
      "scripts/serve_policy.py",
      "--env",
      "LIBERO"
    ],
    "status_path": "outputs/openpi_libero/smoke_status.json",
    "suite": "libero_spatial",
    "task_id": 0
  },
  "gpu_status": {
    "available": true,
    "command": [
      "nvidia-smi",
      "--query-gpu=name,memory.total,driver_version",
      "--format=csv,noheader"
    ],
    "returncode": 0,
    "stderr": "",
    "stdout": "NVIDIA RTX A4500, 20470 MiB, 575.57.08\nNVIDIA RTX A4500, 20470 MiB, 575.57.08"
  },
  "module_status": {
    "current_python": {
      "libero": false,
      "mujoco": false,
      "openpi_client": false
    },
    "libero_pythonpath": "external/openpi/third_party/libero",
    "openpi_libero_venv": {
      "libero": true,
      "libero.lifelong": true,
      "mujoco": true,
      "openpi_client": true,
      "torch": true
    },
    "venv_python": "external/openpi/examples/libero/.venv/bin/python"
  },
  "ok": true,
  "openpi_git_commit": {
    "available": true,
    "command": [
      "git",
      "rev-parse",
      "HEAD"
    ],
    "returncode": 0,
    "stderr": "",
    "stdout": "c23745b5ad24e98f66967ea795a07b2588ed6c79"
  },
  "platform": "Linux-5.15.0-161-generic-x86_64-with-glibc2.35",
  "python": "3.13.5 | packaged by Anaconda, Inc. | (main, Jun 12 2025, 16:09:02) [GCC 11.2.0]",
  "resume_commands": [
    "git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git external/openpi",
    "cd external/openpi",
    "git submodule update --init --recursive",
    "uv venv --python 3.8 examples/libero/.venv",
    "source examples/libero/.venv/bin/activate",
    "uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match",
    "uv pip install -e packages/openpi-client",
    "uv pip install -e third_party/libero",
    "export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero",
    "uv run scripts/serve_policy.py --env LIBERO",
    "MUJOCO_GL=egl python examples/libero/main.py --task-suite-name libero_spatial"
  ],
  "warnings": []
}
```
