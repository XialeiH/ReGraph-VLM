#!/bin/bash
#SBATCH --job-name=p32_edge_final
#SBATCH --account=zx555
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --array=0-23
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/p32_edge_final_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/p32_edge_final_%A_%a.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p logs
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

folds=(fold_01 fold_02 fold_03 fold_04 fold_05 fold_06 fold_07 fold_08)
seeds=(11 22 33)

fold=${folds[$((SLURM_ARRAY_TASK_ID / 3))]}
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold \
  --output-root preproc_v0/repetition_familiarity/results/phase32_edge_bias_allfold_3seed \
  --fold "$fold" \
  --graph-encoder edge_bias_bnt \
  --readout gated_flat \
  --lambda-clip 2.0 \
  --adjacency-mode no_adjacency \
  --attention-bias-scale 1.0 \
  --attention-adjacency-scale 0.0 \
  --seed "$seed" \
  --epochs 80 \
  --patience 12 \
  --batch-size 128 \
  --device auto
