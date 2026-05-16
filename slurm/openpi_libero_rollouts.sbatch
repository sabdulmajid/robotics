#!/usr/bin/env bash
#SBATCH --job-name=openpi-libero-rollouts
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/slurm_nfs/a6abdulm/robotics
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/robotics/outputs/openpi_libero/rollouts_%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/robotics/outputs/openpi_libero/rollouts_%j.err

set -euo pipefail

echo "This script reserves the target rollout resources. The OpenPI rollout collector is gated on the smoke setup passing."
echo "Run first:"
echo "  python -m risk_aware_skill_planning.cli openpi-libero-smoke --config configs/openpi_libero_smoke.yaml --strict"
echo
echo "After OpenPI/LIBERO is installed and smoke passes, replace this placeholder with the rollout collector command."
