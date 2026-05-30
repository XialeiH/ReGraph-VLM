#!/bin/bash
#SBATCH --job-name=p4_adj_perturb
#SBATCH --account=zx555
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --array=0-7
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/p4_adj_perturb_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/p4_adj_perturb_%A_%a.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p logs
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

folds=(fold_01 fold_02 fold_03 fold_04 fold_05 fold_06 fold_07 fold_08)
fold=${folds[$SLURM_ARRAY_TASK_ID]}

python scripts/run_adjacency_perturbation_controls.py \
  --root "$ROOT" \
  --fold "$fold" \
  --seed 11
