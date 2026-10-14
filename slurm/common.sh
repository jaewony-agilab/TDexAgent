# Sourced by every sbatch script. Starts services on this node and cleans them up on exit.
# Partition/account: pass on the command line (sbatch -p ... -A ...) or export SBATCH_PARTITION / SBATCH_ACCOUNT.
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
# shellcheck disable=SC1091
source "$REPO/.tdex_paths"   # written by scripts/install_rldx.sh (RLDX, SIM_PY[, VLM_PY])
export SLURM_LOG_DIR="${SLURM_LOG_DIR:-$HOME/tdex_scratch}"
export TDEX_COSTCAPS="${TDEX_COSTCAPS:-$REPO/configs/costcaps.yaml}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NO_ALBUMENTATIONS_UPDATE=1
SCRATCH="$SLURM_LOG_DIR/${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH"

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

free_port() { "$SIM_PY" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])'; }

# start_policy_server [MODEL_PATH]  -> exports RLDX_PORT
start_policy_server() {
  local model="${1:-RLWRLD/RLDX-1-FT-GR1}"
  export RLDX_PORT="$(free_port)"
  ( cd "$RLDX" && uv run python rldx/eval/run_rldx_server.py \
      --model-path "$model" --embodiment-tag GENERAL_EMBODIMENT --use-sim-policy-wrapper \
      --host 127.0.0.1 --port "$RLDX_PORT" ) > "$SCRATCH/policy_server.log" 2>&1 &
  PIDS+=($!)
  for _ in $(seq 1 120); do
    if "$SIM_PY" -c "
from rldx.policy.server_client import PolicyClient
import sys; sys.exit(0 if PolicyClient(host='127.0.0.1', port=$RLDX_PORT, timeout_ms=2000).ping() else 1)" 2>/dev/null; then
      echo "[i] policy server up on $RLDX_PORT"; return 0; fi
    sleep 5
  done
  echo "policy server did not come up; see $SCRATCH/policy_server.log"; exit 1
}

# start_vlm_server -> exports TDEX_VLM_URL
start_vlm_server() {
  local model="${TDEX_VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}" port
  port="$(free_port)"
  "$VLM_PY" -m vllm.entrypoints.openai.api_server --model "$model" --port "$port" \
      --max-model-len 16384 --gpu-memory-utilization "${VLM_GPU_FRAC:-0.45}" \
      --limit-mm-per-prompt image=4 > "$SCRATCH/vlm_server.log" 2>&1 &
  PIDS+=($!)
  export TDEX_VLM_URL="http://127.0.0.1:$port/v1"
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null && { echo "[i] vlm up on $port"; return 0; }
    sleep 5
  done
  echo "vlm server did not come up; see $SCRATCH/vlm_server.log"; exit 1
}

task_by_index() { "$SIM_PY" -c "from tdex.config import GR1_TASKS; print(GR1_TASKS[$1])"; }
cfg_list() { "$SIM_PY" -c "from tdex import config as C; print(' '.join(C.load()['$1']))"; }
