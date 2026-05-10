#!/usr/bin/env bash
#SBATCH --job-name=rg_hold8_build
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

python scripts/build_cross_subject_heldout_image_dataset.py \
  --root "$ROOT" \
  --input-dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip \
  --output-dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_imageheldout \
  --folds fold_01 fold_02 fold_03 fold_04 fold_05 fold_06 fold_07 fold_08

python scripts/build_cross_subject_clip_control_dataset.py \
  --root "$ROOT" \
  --input-dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_imageheldout \
  --output-dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_imageheldout_random_embedding \
  --folds fold_01 fold_02 fold_03 fold_04 fold_05 fold_06 fold_07 fold_08 \
  --mode random_embedding
