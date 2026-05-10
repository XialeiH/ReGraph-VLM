#!/usr/bin/env bash
#SBATCH --job-name=bridgegen_full
#SBATCH --account=torch_pr_482_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-139
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

mkdir -p logs
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

read -r TASK MODEL SEED <<EOF_CONF
$(python - <<'PY'
tasks = ["topology", "bridge_count", "attack_disconnect", "redundant"]
models = ["gin", "gatv2", "graphgps", "diffpool", "mincut", "ptr", "ptr_sup"]
seeds = [11, 22, 33, 44, 55]
index = int(__import__("os").environ["SLURM_ARRAY_TASK_ID"])
triples = [(task, model, seed) for task in tasks for model in models for seed in seeds]
task, model, seed = triples[index]
print(task, model, seed)
PY
)
EOF_CONF

"${VENV_PYTHON}" -m bridgegen.train \
  --data-root data/bridge_v1 \
  --output-root "results/full_matrix/${TASK}/${MODEL}" \
  --model "${MODEL}" \
  --task "${TASK}" \
  --seed "${SEED}"
