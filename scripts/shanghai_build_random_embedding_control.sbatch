#!/usr/bin/env bash
#SBATCH --job-name=rg_csrand
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
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python scripts/build_cross_subject_clip_control_dataset.py \
  --root "$ROOT" \
  --mode random_embedding \
  --output-dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_random_embedding \
  --folds fold_01 fold_04
