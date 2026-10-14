#!/usr/bin/env bash
#SBATCH -J tdex-p2
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH -o slurm-%x-%A_%a.out
# Phase 2: one condition per job, array over the selected tasks.
#   screening:      sbatch --export=ALL,CONDITION=overlay,JUDGE=claude slurm/phase2.sbatch
#   full protocol:  sbatch --export=ALL,CONDITION=overlay,JUDGE=claude,SEED=43,EPISODES=50,STAGE=full slurm/phase2.sbatch
# EXTRA passes ablation flags, e.g. EXTRA="--no-recovery" or EXTRA="--recovery-mode resume".
source slurm/common.sh
: "${CONDITION:?CONDITION=executor|plain|overlay|oracle|touch|text}"
JUDGE="${JUDGE:-claude}"; SEED="${SEED:-42}"; EPISODES="${EPISODES:-20}"; STAGE="${STAGE:-screen}"; EXTRA="${EXTRA:-}"
read -r -a TASKS <<< "$(cfg_list selected_tasks)"
[ ${#TASKS[@]} -gt 0 ] || { echo "configs/gr1.yaml selected_tasks is empty"; exit 1; }
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
start_policy_server
[ "$JUDGE" = vlm ] && start_vlm_server
TAG="${CONDITION}_${JUDGE}${EXTRA:+_$(echo "$EXTRA" | tr -d ' -')}"
# shellcheck disable=SC2086
"$SIM_PY" scripts/run_episodes.py --task "$TASK" --condition "$CONDITION" --judge "$JUDGE" \
  --seed "$SEED" --episodes "$EPISODES" --port "$RLDX_PORT" $EXTRA \
  --out "results/phase2/$STAGE/$TAG/s$SEED/$TASK"
