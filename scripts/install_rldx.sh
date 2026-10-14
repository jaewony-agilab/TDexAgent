#!/usr/bin/env bash
# Unattended install from a clean clone, on a cluster login/compute node (Linux, CUDA 12.x, uv >= 0.8.4).
#   bash scripts/install_rldx.sh
# Env overrides: TDEX_THIRD_PARTY (default: <repo>/third_party), HF_HOME (checkpoint cache).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${TDEX_THIRD_PARTY:-$REPO/third_party}"
RLDX_COMMIT="cf67c31f7f168ebbca6d06e8c3c23914cb94435c"   # RLWRLD/RLDX-1 main, 2026-10-14
RLDX="$TP/RLDX-1"

command -v uv >/dev/null || { echo "uv not found (need >= 0.8.4)"; exit 1; }
mkdir -p "$TP"

if [ ! -d "$RLDX/.git" ]; then
  git clone https://github.com/RLWRLD/RLDX-1.git "$RLDX"
fi
git -C "$RLDX" fetch --quiet origin "$RLDX_COMMIT" || true
git -C "$RLDX" checkout --quiet "$RLDX_COMMIT"
echo "[i] RLDX-1 at $(git -C "$RLDX" rev-parse HEAD)"

# 1. Model venv (policy server)
( cd "$RLDX" && uv sync --python 3.10 && uv pip install -e . )
( cd "$RLDX" && uv run python -c "import rldx; print('rldx', rldx.__version__)" )

# 2. Sim venv (robosuite 1.5.1 + robocasa-gr1-tabletop-tasks + determinism patches + assets)
( cd "$RLDX" && bash run_scripts/eval/gr1_tabletop/setup_gr1.sh )
SIM_PY="$RLDX/rldx/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python"
[ -x "$SIM_PY" ] || { echo "sim venv missing at $SIM_PY"; exit 1; }

# 3. This package into the sim venv (harness, judge, planner clients)
# not [dev]: it pulls a newer mujoco than robosuite 1.5.1 was pinned with
uv pip install --python "$SIM_PY" -e "$REPO[llm]" pytest
"$SIM_PY" -c "import tdex, anthropic, openai, mujoco; print('tdex OK, mujoco', mujoco.__version__)"

# 4. Checkpoints into the HF cache
( cd "$RLDX" && uv run python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("RLWRLD/RLDX-1-FT-GR1",):
    print(repo, "->", snapshot_download(repo))
PY
)

cat > "$REPO/.tdex_paths" <<EOF
RLDX=$RLDX
SIM_PY=$SIM_PY
EOF
echo "[i] done. Paths written to $REPO/.tdex_paths"
