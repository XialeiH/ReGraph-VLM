#!/usr/bin/env bash
#SBATCH --job-name=rg_holdbuild
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT/scripts:$ROOT:${PYTHONPATH:-}"

python scripts/build_cross_subject_heldout_image_dataset.py \
  --root "$ROOT" \
  --folds fold_01 fold_04

python scripts/build_cross_subject_clip_control_dataset.py \
  --root "$ROOT" \
  --input-dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_imageheldout \
  --output-dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_imageheldout_random_embedding \
  --mode random_embedding \
  --folds fold_01 fold_04
