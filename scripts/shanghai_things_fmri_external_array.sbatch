#!/bin/bash
#SBATCH --job-name=things_fmri_ext
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --output=slurm_logs/things_fmri_ext_%A_%a.out
#SBATCH --error=slurm_logs/things_fmri_ext_%A_%a.err
#SBATCH --array=0-17
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G

set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

DATA_DIR="external_validation/things_fmri/visual_roi_scalar4_smoke"
OUT_DIR="$DATA_DIR/trained_external"
SEEDS=(11 22 33)
MODELS=(roi_mlp roi_transformer_gated)
PAIRS=("sub-01 sub-02" "sub-01 sub-03" "sub-02 sub-03")

pair_idx=$((SLURM_ARRAY_TASK_ID / 6))
rem=$((SLURM_ARRAY_TASK_ID % 6))
model_idx=$((rem / 3))
seed_idx=$((rem % 3))

read -r SUB_A SUB_B <<< "${PAIRS[$pair_idx]}"
MODEL="${MODELS[$model_idx]}"
SEED="${SEEDS[$seed_idx]}"

python scripts/train_cneuromod_visual_roi_external.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR/${SUB_A}_${SUB_B}" \
  --subject-a "$SUB_A" \
  --subject-b "$SUB_B" \
  --file-template '{subject}_things_fmri_visual_roi_scalar4.pt' \
  --repetitions 1 \
  --model "$MODEL" \
  --seed "$SEED" \
  --batch-size 128 \
  --epochs 80 \
  --patience 12
