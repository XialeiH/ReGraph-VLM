#!/usr/bin/env bash
#SBATCH --job-name=rg_cs8data
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FOLDS=(fold_01 fold_02 fold_03 fold_04 fold_05 fold_06 fold_07 fold_08)

python scripts/build_repetition_t3_dataset.py \
  --root "$ROOT" \
  --folds "${FOLDS[@]}" \
  --output-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_allfold

python scripts/attach_clip_to_repetition_t3_dataset.py \
  --root "$ROOT" \
  --folds "${FOLDS[@]}" \
  --input-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_allfold \
  --output-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_allfold

python scripts/build_cross_subject_repeat_pair_dataset.py \
  --root "$ROOT" \
  --folds "${FOLDS[@]}" \
  --input-dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_allfold \
  --output-dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold
