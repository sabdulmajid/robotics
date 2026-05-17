#!/usr/bin/env bash
#SBATCH --job-name=openpi-libero-rollouts
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --chdir=/mnt/slurm_nfs/a6abdulm/robotics
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/robotics/outputs/openpi_libero/rollouts_%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/robotics/outputs/openpi_libero/rollouts_%j.err

set -euo pipefail

REPO_ROOT=/mnt/slurm_nfs/a6abdulm/robotics
OPENPI_ROOT="${REPO_ROOT}/external/openpi"
JOB_ID="${SLURM_JOB_ID:-local}"
PORT=$((18000 + (${SLURM_JOB_ID:-10000} % 20000)))
SERVER_LOG="${REPO_ROOT}/outputs/openpi_libero/policy_server_${JOB_ID}.log"
SUMMARY_OUT="${REPO_ROOT}/reports/openpi_libero_rollout_summary_${JOB_ID}.json"

# Override these at submit time, for example:
#   SUITES="libero_spatial libero_goal" TASK_IDS="0 1 2" NUM_TRIALS=3 sbatch ...
SUITES="${SUITES:-libero_spatial}"
TASK_IDS="${TASK_IDS:-0 1 2}"
NUM_TRIALS="${NUM_TRIALS:-3}"
SEED="${SEED:-7}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
MODE="${MODE:-direct_openpi}"
RISK_SUMMARY="${RISK_SUMMARY:-}"
LOW_RISK_THRESHOLD="${LOW_RISK_THRESHOLD:-0.35}"
MEDIUM_RISK_THRESHOLD="${MEDIUM_RISK_THRESHOLD:-0.65}"
HIGH_RISK_THRESHOLD="${HIGH_RISK_THRESHOLD:-0.85}"
ABSTAIN_THRESHOLD="${ABSTAIN_THRESHOLD:-0.95}"
STRESSORS="${STRESSORS:-none}"
STRESSOR_SEVERITY="${STRESSOR_SEVERITY:-0.0}"

mkdir -p "${REPO_ROOT}/outputs/openpi_libero" "${REPO_ROOT}/datasets/openpi_libero_rollouts"
mkdir -p "${REPO_ROOT}/videos" "${REPO_ROOT}/checkpoints/openpi_cache" "${REPO_ROOT}/.cache"

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
(
  flock 9
  if [[ ! -x examples/libero/.venv/bin/python ]]; then
    uv venv --python 3.8 examples/libero/.venv
  fi
  uv pip sync --python examples/libero/.venv/bin/python examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
  uv pip install --python examples/libero/.venv/bin/python -e packages/openpi-client -e third_party/libero
) 9>"${REPO_ROOT}/outputs/openpi_libero/openpi_env.lock"

uv run scripts/serve_policy.py --env LIBERO --port "${PORT}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cd "${REPO_ROOT}"
for _ in $(seq 1 180); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "OpenPI policy server exited before accepting connections. Last log lines:"
    tail -120 "${SERVER_LOG}" || true
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

ROLLOUT_FILES=()
for stressor in ${STRESSORS}; do
  for suite in ${SUITES}; do
    for task_id in ${TASK_IDS}; do
      jsonl_out="${REPO_ROOT}/datasets/openpi_libero_rollouts/openpi_${suite}_task${task_id}_${stressor}_${JOB_ID}.jsonl"
      video_out="${REPO_ROOT}/videos/openpi_${suite}_task${task_id}_${stressor}_${JOB_ID}"
      risk_args=()
      if [[ -n "${RISK_SUMMARY}" ]]; then
        risk_args+=(--risk-summary "${RISK_SUMMARY}")
      fi
      "${OPENPI_ROOT}/examples/libero/.venv/bin/python" scripts/openpi_libero_single_task_eval.py \
        --openpi-root "${OPENPI_ROOT}" \
        --run-id "openpi_${suite}_${stressor}_${JOB_ID}" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --mode "${MODE}" \
        --task-suite-name "${suite}" \
        --task-id "${task_id}" \
        --num-trials "${NUM_TRIALS}" \
        --seed "${SEED}" \
        --replan-steps "${REPLAN_STEPS}" \
        --low-risk-threshold "${LOW_RISK_THRESHOLD}" \
        --medium-risk-threshold "${MEDIUM_RISK_THRESHOLD}" \
        --high-risk-threshold "${HIGH_RISK_THRESHOLD}" \
        --abstain-threshold "${ABSTAIN_THRESHOLD}" \
        --stressor-name "${stressor}" \
        --stressor-severity "${STRESSOR_SEVERITY}" \
        --jsonl-out "${jsonl_out}" \
        --video-out-path "${video_out}" \
        "${risk_args[@]}"
      ROLLOUT_FILES+=("${jsonl_out}")
    done
  done
done

combined="${REPO_ROOT}/datasets/openpi_libero_rollouts/openpi_rollouts_${JOB_ID}.jsonl"
cat "${ROLLOUT_FILES[@]}" > "${combined}"

PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" python -m risk_aware_skill_planning.cli openpi-libero-summarize \
  --input "${combined}" \
  --output "${SUMMARY_OUT}"

echo "Combined rollout JSONL: ${combined}"
echo "Rollout summary: ${SUMMARY_OUT}"
