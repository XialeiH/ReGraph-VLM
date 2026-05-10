#!/usr/bin/env bash
#SBATCH --job-name=rg_hn8_build
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python scripts/build_cross_subject_hard_negative_dataset.py \
  --root "$ROOT" \
  --input-dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold \
  --output-dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_hardneg_allfold \
  --folds fold_01 fold_02 fold_03 fold_04 fold_05 fold_06 fold_07 fold_08 \
  --negative-mode mixed_50_random_50_clip_hard
