#!/bin/bash
#SBATCH --job-name=rgvhns
#SBATCH --account=torch_pr_482_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/rgvhns_%j.out
#SBATCH --error=logs/rgvhns_%j.err

set -euo pipefail
ROOT=/scratch/xh2906/final_project_nsd/v0_shared_unit
PY=/scratch/xh2906/conda/envs/py310/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

$PY scripts/summarize_regraph_vlm_results.py \
  --root "$ROOT" \
  --results-dir preproc_v0/repetition_familiarity/results/regraph_vlm_hardneg
