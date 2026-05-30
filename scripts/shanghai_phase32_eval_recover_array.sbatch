#!/bin/bash
#SBATCH --job-name=p32_eval_recover
#SBATCH --account=zx555
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --array=0-11
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/p32_eval_recover_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/p32_eval_recover_%A_%a.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p logs
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

encoders=(edge_bias_bnt edge_bias_graph_bnt)
folds=(fold_01 fold_04)
seeds=(11 22 33)

enc_idx=$((SLURM_ARRAY_TASK_ID / 6))
rem=$((SLURM_ARRAY_TASK_ID % 6))
fold_idx=$((rem / 3))
seed_idx=$((rem % 3))

encoder=${encoders[$enc_idx]}
fold=${folds[$fold_idx]}
seed=${seeds[$seed_idx]}

adjacency_mode=default
if [[ "$encoder" == "edge_bias_bnt" ]]; then
  adjacency_mode=no_adjacency
fi

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold \
  --output-root preproc_v0/repetition_familiarity/results/phase32_learned_edge_bias \
  --fold "$fold" \
  --graph-encoder "$encoder" \
  --readout gated_flat \
  --lambda-clip 2.0 \
  --adjacency-mode "$adjacency_mode" \
  --attention-bias-scale 1.0 \
  --attention-adjacency-scale 0.1 \
  --seed "$seed" \
  --batch-size 128 \
  --device auto \
  --eval-only
