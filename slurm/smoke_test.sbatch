#!/usr/bin/env bash
#SBATCH --job-name=rask-smoke
#SBATCH --partition=midcard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:05:00
#SBATCH --output=outputs/slurm-smoke-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m risk_aware_skill_planning.cli smoke
