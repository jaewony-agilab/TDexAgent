#!/usr/bin/env bash
#SBATCH -J tdex-p0-repro
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o slurm-%x-%A_%a.out
# Phase 0 step 1: reproduce RLDX-1-FT-GR1 on the repro subset (1 seed, 20 episodes),
# (a) with RLDX-1's own rollout client, (b) with our harness (executor alone, taxels on).
# Stop rule: (a) more than 10pp below the published per-task number.
source slurm/common.sh
EPISODES="${EPISODES:-20}"
read -r -a TASKS <<< "$(cfg_list repro_tasks)"
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
OUT="results/phase0/repro/$TASK"
start_policy_server

VID="$SCRATCH/rldx_eval"
mkdir -p "$VID" "$OUT/rldx_eval"
"$SIM_PY" "$RLDX/rldx/eval/rollout_policy.py" \
  --policy_client_host 127.0.0.1 --policy_client_port "$RLDX_PORT" \
  --env_name "gr1_unified/$TASK" --n_episodes "$EPISODES" --max_episode_steps 720 \
  --n_action_steps 16 --n_envs 1 --video_dir "$VID" 2>&1 | tail -n 50 > "$OUT/rldx_eval/eval_tail.log"
cp "$VID/simulation_results.csv" "$VID/summary.txt" "$OUT/rldx_eval/" 2>/dev/null || true
echo "{\"slurm_job_id\": \"$SLURM_JOB_ID\", \"commit\": \"$(git rev-parse HEAD)\", \"rldx_commit\": \"$(git -C "$RLDX" rev-parse HEAD)\"}" > "$OUT/rldx_eval/provenance.json"

"$SIM_PY" scripts/run_episodes.py --task "$TASK" --condition executor --episodes "$EPISODES" \
  --port "$RLDX_PORT" --out "$OUT/harness"
