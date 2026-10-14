#!/usr/bin/env bash
#SBATCH -J tdex-p1-judge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o slurm-%x-%j.out
# Phase 1 step 4: one (JUDGE, CONDITION) over the saved boundary set.
#   sbatch --export=ALL,JUDGE=vlm,CONDITION=overlay slurm/phase1_judge.sbatch
source slurm/common.sh
: "${JUDGE:?JUDGE=claude|vlm}" "${CONDITION:?CONDITION=plain|text|overlay}"
[ "$JUDGE" = vlm ] && start_vlm_server
"$SIM_PY" scripts/offline_judge.py --packets "$SLURM_LOG_DIR/boundary_set" --judge "$JUDGE" \
  --condition "$CONDITION" --out "results/phase1/judge/${JUDGE}_${CONDITION}"
