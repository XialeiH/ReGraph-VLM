#!/bin/bash
#SBATCH --job-name=rgvhn
#SBATCH --account=torch_pr_482_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/rgvhn_%A_%a.out
#SBATCH --error=logs/rgvhn_%A_%a.err

set -euo pipefail
ROOT=/scratch/xh2906/final_project_nsd/v0_shared_unit
PY=/scratch/xh2906/conda/envs/py310/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4

CONFIGS=(
  "bnt_token_flat fold_01 2.0 11"
  "bnt_token_flat fold_01 2.0 22"
  "bnt_token_flat fold_01 2.0 33"
  "bnt_token_flat fold_04 2.0 11"
  "bnt_token_flat fold_04 2.0 22"
  "bnt_token_flat fold_04 2.0 33"
  "roi_mlp fold_01 2.0 11"
  "roi_mlp fold_01 2.0 22"
  "roi_mlp fold_01 2.0 33"
  "roi_mlp fold_04 2.0 11"
  "roi_mlp fold_04 2.0 22"
  "roi_mlp fold_04 2.0 33"
)

read -r ENCODER FOLD LAMBDA_CLIP SEED <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] hardneg mixed encoder=$ENCODER fold=$FOLD lambda_clip=$LAMBDA_CLIP seed=$SEED"

$PY scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_hardneg/mixed_50_random_50_hard \
  --extra-test-dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip \
  --extra-test-name random \
  --output-root preproc_v0/repetition_familiarity/results/regraph_vlm_hardneg \
  --fold "$FOLD" \
  --graph-encoder "$ENCODER" \
  --lambda-clip "$LAMBDA_CLIP" \
  --epochs 80 \
  --patience 12 \
  --batch-size 128 \
  --hidden-dim 64 \
  --embedding-dim 128 \
  --dropout 0.3 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --seed "$SEED" \
  --device auto
