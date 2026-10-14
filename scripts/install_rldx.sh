#!/usr/bin/env bash
# Unattended install from a clean clone (Linux, CUDA 12.x, uv >= 0.8.4). Runs on a CPU node:
#   srun --cpus-per-task=8 --mem=32G -t 3:00:00 bash scripts/install_rldx.sh
# Nothing here needs a GPU. Only RLDX-1's final setup check renders (EGL); on a node without one
# that check fails, so we run its steps ourselves and verify the env without a renderer instead.
# Rendering is checked at the start of every GPU job (slurm/common.sh).
# Env overrides: TDEX_THIRD_PARTY (default: <repo>/third_party), HF_HOME (checkpoint cache).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${TDEX_THIRD_PARTY:-$REPO/third_party}"
RLDX_COMMIT="cf67c31f7f168ebbca6d06e8c3c23914cb94435c"   # RLWRLD/RLDX-1 main, 2026-10-14
RLDX="$TP/RLDX-1"
GR1_TEST_ENV="gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env"

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

# 2. Sim venv (robosuite 1.5.1 + robocasa-gr1-tabletop-tasks + determinism patches + assets).
# Same steps as RLDX-1's run_scripts/eval/gr1_tabletop/setup_gr1.sh at the pinned commit, except:
#  - flash-attn is not installed in the sim venv (setup_gr1.sh installs it, then disables it: the
#    sim venv only runs the zmq client), so there is nothing to build or move;
#  - the upstream script's last step, gym.make(..., enable_render=True), needs EGL. If it fails,
#    we accept it only when the env builds without a renderer (check below).
SIM_SETUP="rldx/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh"
SIM_PY="$RLDX/rldx/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python"
if ( cd "$RLDX" && INSTALL_FLASH_ATTN=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl bash "$SIM_SETUP" ); then
  echo "[i] sim setup OK, including the EGL render check"
else
  echo "[i] sim setup exited non-zero; checking that only the render check failed"
  [ -x "$SIM_PY" ] || { echo "sim venv missing at $SIM_PY: the failure is earlier than the render check, see above"; exit 1; }
  "$SIM_PY" - "$GR1_TEST_ENV" <<'PY' || { echo "env does not build without a renderer: real install failure, see above"; exit 1; }
import sys
import gymnasium as gym, robocasa, robosuite  # noqa: F401
import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
env = gym.make(sys.argv[1], enable_render=False)
env.reset(seed=0)
print("Env OK without renderer:", robosuite.__version__, type(env.unwrapped).__name__)
PY
  echo "[i] only the render check failed (expected on a CPU node); rendering is checked in GPU jobs"
fi

PATCH_DIR="$RLDX/run_scripts/eval/gr1_tabletop/patches"
ROBOCASA_DIR="$RLDX/external_dependencies/robocasa-gr1-tabletop-tasks"
for patch in "$PATCH_DIR"/*.patch; do
  [ -f "$patch" ] || continue
  if git -C "$ROBOCASA_DIR" apply --check "$patch" 2>/dev/null; then
    git -C "$ROBOCASA_DIR" apply "$patch"; echo "[i] applied $(basename "$patch")"
  elif git -C "$ROBOCASA_DIR" apply --reverse --check "$patch" 2>/dev/null; then
    echo "[i] already applied $(basename "$patch")"
  else
    echo "patch $(basename "$patch") neither applies nor is applied"; exit 1
  fi
done

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

# keep VLM_PY if install_vlm.sh already ran
VLM_LINE="$(grep '^VLM_PY=' "$REPO/.tdex_paths" 2>/dev/null | tail -1 || true)"
cat > "$REPO/.tdex_paths" <<EOF
RLDX=$RLDX
SIM_PY=$SIM_PY
EOF
[ -n "$VLM_LINE" ] && echo "$VLM_LINE" >> "$REPO/.tdex_paths"
echo "[i] done. Paths written to $REPO/.tdex_paths"
