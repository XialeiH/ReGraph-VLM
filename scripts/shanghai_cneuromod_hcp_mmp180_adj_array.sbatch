#!/bin/bash
#SBATCH --job-name=cnm_hcp_adj
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --output=slurm_logs/cnm_hcp_adj_%A_%a.out
#SBATCH --error=slurm_logs/cnm_hcp_adj_%A_%a.err
#SBATCH --array=0-35%12
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G

set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:$PWD/scripts:${PYTHONPATH:-}"

DATA_DIR="external_validation/cneuromod_things/hcp_mmp180_scalar4_all4"
OUT_DIR="$DATA_DIR/trained_external"
SEEDS=(11 22 33)
MODELS=(roi_transformer_adj_gated roi_transformer_attn_adj_gated)
PAIRS=(
  "sub-01 sub-02"
  "sub-01 sub-03"
  "sub-01 sub-06"
  "sub-02 sub-03"
  "sub-02 sub-06"
  "sub-03 sub-06"
)

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
  --file-template '{subject}_cneuromod_things_trial_scalar4.pt' \
  --repetitions 1 2 3 \
  --model "$MODEL" \
  --seed "$SEED" \
  --hidden-dim 64 \
  --embedding-dim 128 \
  --num-heads 4 \
  --num-layers 2 \
  --dropout 0.30 \
  --temperature 0.10 \
  --batch-size 128 \
  --epochs 80 \
  --patience 12 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --adjacency-topk 20 \
  --graph-bias-scale 1.0 \
  --attention-adjacency-scale 1.0
