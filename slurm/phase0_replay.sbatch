#!/usr/bin/env bash
#SBATCH -J tdex-p0-replay
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o slurm-%x-%j.out
# Phase 0 step 2: demo replay check. Set GR1_LEROBOT (a GR-1 tabletop task dir of
# GR00T-X-Embodiment-Sim) and/or GR1_HDF5 (a robocasa hdf5 with states) before sbatch.
source slurm/common.sh
ARGS=()
[ -n "${GR1_LEROBOT:-}" ] && ARGS+=(--lerobot "$GR1_LEROBOT")
[ -n "${GR1_HDF5:-}" ] && ARGS+=(--hdf5 "$GR1_HDF5")
[ ${#ARGS[@]} -gt 0 ] || { echo "set GR1_LEROBOT and/or GR1_HDF5"; exit 1; }
"$SIM_PY" scripts/check_replay.py "${ARGS[@]}" --out results/phase0/replay
