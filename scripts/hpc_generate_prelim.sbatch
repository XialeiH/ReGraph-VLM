#!/usr/bin/env bash
#SBATCH --job-name=bridgegen_gen
#SBATCH --account=torch_pr_482_general
#SBATCH --partition=cpu_short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

mkdir -p logs
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

"${VENV_PYTHON}" -m bridgegen.generate_corpus --config configs/bridge_v1.yaml
"${VENV_PYTHON}" -m bridgegen.run_preliminary --config configs/bridge_v1.yaml
