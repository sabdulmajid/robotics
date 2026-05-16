#!/usr/bin/env bash
#SBATCH --job-name=openpi-libero-smoke
#SBATCH --partition=midcard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:20:00
#SBATCH --chdir=/mnt/slurm_nfs/a6abdulm/robotics
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/robotics/outputs/openpi_libero/smoke_%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/robotics/outputs/openpi_libero/smoke_%j.err

set -euo pipefail
python -m risk_aware_skill_planning.cli openpi-libero-smoke --config configs/openpi_libero_smoke.yaml
