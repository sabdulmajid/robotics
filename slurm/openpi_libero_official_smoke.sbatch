#!/usr/bin/env bash
#SBATCH --job-name=openpi-libero-official
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/slurm_nfs/a6abdulm/robotics
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/robotics/outputs/openpi_libero/official_smoke_%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/robotics/outputs/openpi_libero/official_smoke_%j.err

set -euo pipefail

REPO_ROOT=/mnt/slurm_nfs/a6abdulm/robotics
OPENPI_ROOT="${REPO_ROOT}/external/openpi"
JOB_ID="${SLURM_JOB_ID:-local}"
PORT=$((18000 + (${SLURM_JOB_ID:-10000} % 20000)))
SERVER_LOG="${REPO_ROOT}/outputs/openpi_libero/policy_server_${JOB_ID}.log"
JSONL_OUT="${REPO_ROOT}/datasets/openpi_libero_rollouts/openpi_libero_official_smoke_${JOB_ID}.jsonl"
VIDEO_OUT="${REPO_ROOT}/videos/openpi_libero_official_smoke_${JOB_ID}"
SUMMARY_OUT="${REPO_ROOT}/reports/openpi_libero_official_smoke_summary.json"

mkdir -p "${REPO_ROOT}/outputs/openpi_libero" "${REPO_ROOT}/datasets/openpi_libero_rollouts" "${VIDEO_OUT}"
mkdir -p "${REPO_ROOT}/checkpoints/openpi_cache" "${REPO_ROOT}/.cache"

export UV_PYTHON_INSTALL_DIR="${REPO_ROOT}/.uv-python"
export UV_CACHE_DIR="${REPO_ROOT}/.uv-cache"
export UV_LINK_MODE=copy
export OPENPI_DATA_HOME="${REPO_ROOT}/checkpoints/openpi_cache"
export XDG_CACHE_HOME="${REPO_ROOT}/.cache"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export LIBERO_CONFIG_PATH="${REPO_ROOT}/outputs/openpi_libero/libero_config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${OPENPI_ROOT}/third_party/libero:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "${OPENPI_ROOT}"
if [[ ! -x examples/libero/.venv/bin/python ]]; then
  uv venv --python 3.8 examples/libero/.venv
fi
uv pip sync --python examples/libero/.venv/bin/python examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install --python examples/libero/.venv/bin/python -e packages/openpi-client -e third_party/libero

uv run scripts/serve_policy.py --env LIBERO --port "${PORT}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cd "${REPO_ROOT}"
for _ in $(seq 1 180); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "OpenPI policy server exited before accepting connections. Last log lines:"
    tail -80 "${SERVER_LOG}" || true
    exit 1
  fi
  if python - "${PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
else:
    raise SystemExit(0)
finally:
    sock.close()
PY
  then
    break
  fi
  sleep 10
done

if ! python - "${PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
else:
    raise SystemExit(0)
finally:
    sock.close()
PY
then
  echo "Timed out waiting for OpenPI policy server on port ${PORT}. Last log lines:"
  tail -120 "${SERVER_LOG}" || true
  exit 1
fi

"${OPENPI_ROOT}/examples/libero/.venv/bin/python" scripts/openpi_libero_single_task_eval.py \
  --openpi-root "${OPENPI_ROOT}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --task-suite-name libero_spatial \
  --task-id 0 \
  --num-trials 1 \
  --seed 7 \
  --replan-steps 5 \
  --jsonl-out "${JSONL_OUT}" \
  --video-out-path "${VIDEO_OUT}"

PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" python -m risk_aware_skill_planning.cli openpi-libero-summarize \
  --input "${JSONL_OUT}" \
  --output "${SUMMARY_OUT}"

echo "Official OpenPI/LIBERO smoke JSONL: ${JSONL_OUT}"
echo "Official OpenPI/LIBERO smoke videos: ${VIDEO_OUT}"
echo "Official OpenPI/LIBERO smoke summary: ${SUMMARY_OUT}"
