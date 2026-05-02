#!/usr/bin/env bash
#SBATCH --job-name=bridgegen_pilot
#SBATCH --account=torch_pr_482_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

mkdir -p logs
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

"${VENV_PYTHON}" -m bridgegen.run_pilot --config configs/pilot.yaml
