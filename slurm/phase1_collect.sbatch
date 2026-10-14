#!/usr/bin/env bash
#SBATCH -J tdex-p1-collect
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o slurm-%x-%A_%a.out
# Phase 1 step 2: boundary set from the executor alone on the selected tasks, 5 init states.
# Full packets -> $SLURM_LOG_DIR/boundary_set (scratch); compact summary -> results/phase1/collect.
source slurm/common.sh
read -r -a TASKS <<< "$(cfg_list selected_tasks)"
[ ${#TASKS[@]} -gt 0 ] || { echo "configs/gr1.yaml selected_tasks is empty (finish Phase 0 first)"; exit 1; }
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
start_policy_server
"$SIM_PY" scripts/run_episodes.py --task "$TASK" --condition executor --episodes "${EPISODES:-5}" \
  --port "$RLDX_PORT" --out "results/phase1/collect/$TASK" --packets-dir "$SLURM_LOG_DIR/boundary_set"
