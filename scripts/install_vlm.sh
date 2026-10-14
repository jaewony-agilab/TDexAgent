#!/usr/bin/env bash
# Open-weight judge: vLLM OpenAI-compatible server in its own venv.
#   bash scripts/install_vlm.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${TDEX_THIRD_PARTY:-$REPO/third_party}"
VENV="$TP/vlm_venv"
VLLM_VERSION="${VLLM_VERSION:-0.11.0}"  # first release with Qwen3-VL. Bump deliberately, record in DECISIONS.md
MODEL="${TDEX_VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

uv venv "$VENV" --python 3.10
uv pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION"
"$VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("$MODEL"))
PY
echo "VLM_PY=$VENV/bin/python" >> "$REPO/.tdex_paths"
