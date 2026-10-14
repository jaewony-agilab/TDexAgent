#!/usr/bin/env bash
#SBATCH -J tdex-p0-sensor
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o slurm-%x-%A_%a.out
# Phase 0 step 5: does the policy still work with taxels read? Same 5 init states, with / without.
source slurm/common.sh
read -r -a TASKS <<< "$(cfg_list repro_tasks)"
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
OUT="results/phase0/sensorcost/$TASK"
start_policy_server
"$SIM_PY" scripts/sensor_cost.py --task "$TASK" --out "$OUT"
"$SIM_PY" scripts/run_episodes.py --task "$TASK" --condition executor --episodes 5 --port "$RLDX_PORT" --out "$OUT/with_taxels"
"$SIM_PY" scripts/run_episodes.py --task "$TASK" --condition executor --episodes 5 --port "$RLDX_PORT" --out "$OUT/without_taxels" --no-taxels
