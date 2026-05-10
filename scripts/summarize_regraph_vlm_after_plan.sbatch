#!/bin/bash
#SBATCH --job-name=rgvsum
#SBATCH --account=torch_pr_482_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/rgvsum_%j.out
#SBATCH --error=logs/rgvsum_%j.err

set -euo pipefail
ROOT=/scratch/xh2906/final_project_nsd/v0_shared_unit
PY=/scratch/xh2906/conda/envs/py310/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

$PY scripts/summarize_regraph_vlm_results.py --root "$ROOT"
