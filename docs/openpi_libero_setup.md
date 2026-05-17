# OpenPI/LIBERO Setup

This project uses OpenPI `pi05_libero` as the robot foundation policy and LIBERO as the manipulation benchmark. OpenPI lives under ignored `external/openpi`; generated model/cache data stays under repo-managed cache directories.

## Cluster Resources

Run OpenPI jobs from `/mnt/slurm_nfs/a6abdulm/robotics`. The working SLURM target used for the current results is `dualcard` with one NVIDIA RTX A4500 GPU, 6 CPUs, and 32G RAM. The setup smoke recorded:

```text
2 x NVIDIA RTX A4500, 20470 MiB, driver 575.57.08
OpenPI commit c23745b5ad24e98f66967ea795a07b2588ed6c79
```

Re-check before submitting:

```bash
sinfo
squeue -u "$USER"
```

## Install / Validate

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git external/openpi
cd external/openpi
git submodule update --init --recursive
uv venv --python 3.8 examples/libero/.venv
uv pip sync --python examples/libero/.venv/bin/python examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install --python examples/libero/.venv/bin/python -e packages/openpi-client -e third_party/libero
```

Smoke checks:

```bash
PYTHONPATH=src python scripts/openpi_libero_smoke.py --config configs/openpi_libero_smoke.yaml --strict
sbatch slurm/openpi_libero_smoke.sbatch
sbatch slurm/openpi_libero_official_smoke.sbatch
```

The official smoke starts OpenPI's policy server, runs one `pi05_libero` LIBERO episode, and writes:

- `reports/openpi_libero_official_eval_smoke.md`
- `reports/artifacts/openpi_libero_official_smoke_10092.jsonl`
- `reports/artifacts/openpi_libero_official_smoke_10092_success.mp4`

## Runtime Environment

The SLURM scripts set:

```bash
OPENPI_DATA_HOME=$REPO_ROOT/checkpoints/openpi_cache
XDG_CACHE_HOME=$REPO_ROOT/.cache
HF_HOME=$REPO_ROOT/.cache/huggingface
LIBERO_CONFIG_PATH=$REPO_ROOT/outputs/openpi_libero/libero_config
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
```

Use the scripts rather than running long OpenPI jobs on the login node.
