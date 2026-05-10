#!/bin/bash
#SBATCH --job-name=rgvmlp2
#SBATCH --account=torch_pr_482_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/rgvmlp2_%A_%a.out
#SBATCH --error=logs/rgvmlp2_%A_%a.err

set -euo pipefail
ROOT=/scratch/xh2906/final_project_nsd/v0_shared_unit
PY=/scratch/xh2906/conda/envs/py310/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4

CONFIGS=(
  "fold_01 2.0 11"
  "fold_01 2.0 22"
  "fold_01 2.0 33"
  "fold_04 2.0 11"
  "fold_04 2.0 22"
  "fold_04 2.0 33"
)

read -r FOLD LAMBDA_CLIP SEED <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] graph_encoder=roi_mlp fold=$FOLD lambda_clip=$LAMBDA_CLIP seed=$SEED"

$PY scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --fold "$FOLD" \
  --graph-encoder roi_mlp \
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
