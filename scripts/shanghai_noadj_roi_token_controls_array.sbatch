#!/bin/bash
#SBATCH --job-name=noadj_roi_ctrl
#SBATCH --account=zx555
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --array=0-23
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/noadj_roi_ctrl_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/noadj_roi_ctrl_%A_%a.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p logs
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

folds=(fold_01 fold_02 fold_03 fold_04 fold_05 fold_06 fold_07 fold_08)
seeds=(11 22 33)
fold=${folds[$((SLURM_ARRAY_TASK_ID / 3))]}
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}

python scripts/run_noadj_roi_token_controls.py \
  --root "$ROOT" \
  --fold "$fold" \
  --seed "$seed"
