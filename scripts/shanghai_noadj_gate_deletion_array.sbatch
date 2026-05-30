#!/bin/bash
#SBATCH --job-name=noadj_gate_del
#SBATCH --account=zx555
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --array=0-23
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/noadj_gate_del_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/noadj_gate_del_%A_%a.err

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

python scripts/run_gate_deletion_test.py \
  --root "$ROOT" \
  --fold "$fold" \
  --seed "$seed" \
  --checkpoint-template "preproc_v0/repetition_familiarity/results/phase3c_noadj_gated_final/roi_transformer_noadj_gated_flat_clip/lambda_2/{fold}/seed_{seed}/checkpoint.pt" \
  --fallback-checkpoint-template "preproc_v0/repetition_familiarity/results/phase3b_clean_graph_ablation/roi_transformer_noadj_gated_flat_clip/lambda_2/{fold}/seed_{seed}/checkpoint.pt" \
  --gate-summary preproc_v0/repetition_familiarity/results/interpretability/gates_noadj/roi_gate_summary.csv \
  --output-dir preproc_v0/repetition_familiarity/results/interpretability/gates_noadj/deletion_tests \
  --random-repeats 20
