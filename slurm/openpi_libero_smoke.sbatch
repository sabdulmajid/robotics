#!/usr/bin/env bash
#SBATCH --job-name=openpi-libero-smoke
#SBATCH --partition=dualcard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --chdir=/mnt/slurm_nfs/a6abdulm/robotics
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/robotics/outputs/openpi_libero/smoke_%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/robotics/outputs/openpi_libero/smoke_%j.err

set -euo pipefail
export UV_PYTHON_INSTALL_DIR=/mnt/slurm_nfs/a6abdulm/robotics/.uv-python
export UV_CACHE_DIR=/mnt/slurm_nfs/a6abdulm/robotics/.uv-cache
export UV_LINK_MODE=copy

cd /mnt/slurm_nfs/a6abdulm/robotics/external/openpi
if [ ! -x examples/libero/.venv/bin/python ]; then
  uv venv --python 3.8 examples/libero/.venv
fi
uv pip sync --python examples/libero/.venv/bin/python examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install --python examples/libero/.venv/bin/python -e packages/openpi-client -e third_party/libero

cd /mnt/slurm_nfs/a6abdulm/robotics
python -m risk_aware_skill_planning.cli openpi-libero-smoke --config configs/openpi_libero_smoke.yaml
