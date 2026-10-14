#!/usr/bin/env bash
#SBATCH -J tdex-p0-sweep
#SBATCH --array=0-23
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o slurm-%x-%A_%a.out
# Phase 0 step 6: executor-alone SR on all 24 tasks (1 seed, 20 episodes). Then locally:
#   python scripts/summarize.py results/phase0/sweep --select
source slurm/common.sh
TASK="$(task_by_index "$SLURM_ARRAY_TASK_ID")"
start_policy_server
"$SIM_PY" scripts/run_episodes.py --task "$TASK" --condition executor --episodes "${EPISODES:-20}" \
  --port "$RLDX_PORT" --out "results/phase0/sweep/$TASK"
