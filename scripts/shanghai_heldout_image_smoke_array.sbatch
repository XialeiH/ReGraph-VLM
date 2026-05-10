#!/usr/bin/env bash
#SBATCH --job-name=rg_holdvlm
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.err

set -euo pipefail
ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

FOLDS=(fold_01 fold_04)
TASK=$((SLURM_ARRAY_TASK_ID / 2))
FOLD="${FOLDS[$((SLURM_ARRAY_TASK_ID % 2))]}"
DATASET=scalar4_T3_clip_cross_subject_imageheldout
OUT=heldout_image
ENC=roi_mlp
READOUT=flat
if [ "$TASK" -eq 1 ]; then ENC=bnt_token_flat; READOUT=flat; fi
if [ "$TASK" -eq 2 ]; then ENC=bnt_token_flat; READOUT=gated_flat; fi
if [ "$TASK" -eq 3 ]; then ENC=bnt_token_flat; READOUT=gated_flat; DATASET=scalar4_T3_clip_cross_subject_imageheldout_random_embedding; OUT=heldout_image_random_embedding; fi

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root "preproc_v0/repetition_familiarity/datasets/${DATASET}" \
  --output-root "preproc_v0/repetition_familiarity/results/${OUT}" \
  --fold "$FOLD" \
  --graph-encoder "$ENC" \
  --readout "$READOUT" \
  --lambda-clip 2.0 \
  --seed 11 \
  --device cpu
