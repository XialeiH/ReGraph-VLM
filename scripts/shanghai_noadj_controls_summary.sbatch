#!/bin/bash
#SBATCH --job-name=noadj_ctrl_sum
#SBATCH --account=zx555
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/noadj_ctrl_sum_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/noadj_ctrl_sum_%j.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p logs
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python scripts/summarize_gate_deletion_tests.py \
  --root "$ROOT" \
  --input-dir preproc_v0/repetition_familiarity/results/interpretability/gates_noadj/deletion_tests

python scripts/summarize_noadj_roi_token_controls.py --root "$ROOT"

python scripts/make_adjacency_ablation_artifacts.py --root "$ROOT"
