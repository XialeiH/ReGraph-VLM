#!/bin/bash
#SBATCH --job-name=laion_fs_train
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --output=slurm_logs/laion_fs_train_%A_%a.out
#SBATCH --error=slurm_logs/laion_fs_train_%A_%a.err
#SBATCH --array=0-59%20
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G

set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

DATA_DIR="external_validation/laion_fmri_freesurfer_aparc/freesurfer_aparc_scalar4_laion"
OUT_DIR="$DATA_DIR/trained_external_clip"
SEEDS=(11 22 33)
MODELS=(roi_mlp_clip gated_roi_transformer_clip)
PAIRS=(
  "sub-01 sub-03"
  "sub-01 sub-05"
  "sub-01 sub-06"
  "sub-01 sub-07"
  "sub-03 sub-05"
  "sub-03 sub-06"
  "sub-03 sub-07"
  "sub-05 sub-06"
  "sub-05 sub-07"
  "sub-06 sub-07"
)

pair_idx=$((SLURM_ARRAY_TASK_ID / 6))
rem=$((SLURM_ARRAY_TASK_ID % 6))
model_idx=$((rem / 3))
seed_idx=$((rem % 3))

read -r SUB_A SUB_B <<< "${PAIRS[$pair_idx]}"
MODEL="${MODELS[$model_idx]}"
SEED="${SEEDS[$seed_idx]}"

python scripts/train_laion_clip_external.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR/${SUB_A}_${SUB_B}" \
  --subject-a "$SUB_A" \
  --subject-b "$SUB_B" \
  --file-template '{subject}_laion_visual_roi_scalar4.pt' \
  --repetitions 1 2 3 \
  --model "$MODEL" \
  --seed "$SEED" \
  --lambda-clip 2.0
