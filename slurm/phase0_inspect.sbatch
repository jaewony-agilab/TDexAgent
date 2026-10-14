#!/usr/bin/env bash
#SBATCH -J tdex-p0-inspect
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o slurm-%x-%j.out
# Phase 0 calibration probe + sensor timing. Output: results/phase0/inspect/
source slurm/common.sh
start_policy_server
TASK="$(task_by_index "${TASK_INDEX:-0}")"
"$SIM_PY" scripts/inspect_env.py --task "$TASK" --port "$RLDX_PORT" --out "results/phase0/inspect/$TASK"
"$SIM_PY" scripts/sensor_cost.py --task "$TASK" --out "results/phase0/inspect/$TASK"
