#!/usr/bin/env bash
#SBATCH --job-name=rg_t3clip8
#SBATCH --partition=gpu,debug,acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python scripts/build_repetition_t3_dataset.py \
  --root "$ROOT" \
  --folds fold_01 fold_02 fold_03 fold_04 fold_05 fold_06 fold_07 fold_08

python scripts/attach_clip_to_repetition_t3_dataset.py \
  --root "$ROOT" \
  --folds fold_01 fold_02 fold_03 fold_04 fold_05 fold_06 fold_07 fold_08
