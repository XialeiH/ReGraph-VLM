#!/bin/bash
#SBATCH --job-name=rgvhnb
#SBATCH --account=torch_pr_482_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rgvhnb_%j.out
#SBATCH --error=logs/rgvhnb_%j.err

set -euo pipefail
ROOT=/scratch/xh2906/final_project_nsd/v0_shared_unit
PY=/scratch/xh2906/conda/envs/py310/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

$PY scripts/build_repeat_pair_hard_negative_dataset.py \
  --root "$ROOT" \
  --negative-mode mixed_50_random_50_hard \
  --folds fold_01 fold_04
