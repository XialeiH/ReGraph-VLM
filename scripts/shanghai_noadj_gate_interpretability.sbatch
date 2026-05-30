#!/bin/bash
#SBATCH --job-name=noadj_gate_int
#SBATCH --account=zx555
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --mem=24G
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/noadj_gate_int_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/noadj_gate_int_%j.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p logs
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python scripts/extract_gated_regraph_gates.py \
  --root "$ROOT" \
  --checkpoint-glob "preproc_v0/repetition_familiarity/results/phase3b_clean_graph_ablation/roi_transformer_noadj_gated_flat_clip/lambda_2/fold_*/seed_11/checkpoint.pt" \
  --checkpoint-glob "preproc_v0/repetition_familiarity/results/phase3c_noadj_gated_final/roi_transformer_noadj_gated_flat_clip/lambda_2/fold_*/seed_*/checkpoint.pt" \
  --output-dir preproc_v0/repetition_familiarity/results/interpretability/gates_noadj

python scripts/analyze_gate_stability.py \
  --root "$ROOT" \
  --gate-csv preproc_v0/repetition_familiarity/results/interpretability/gates_noadj/roi_gate_values.csv

python scripts/analyze_gate_vs_repetition_effects.py \
  --root "$ROOT" \
  --gate-summary preproc_v0/repetition_familiarity/results/interpretability/gates_noadj/roi_gate_summary.csv

python scripts/compare_gate_summaries.py \
  --root "$ROOT" \
  --a-summary preproc_v0/repetition_familiarity/results/interpretability/gates/roi_gate_summary.csv \
  --b-summary preproc_v0/repetition_familiarity/results/interpretability/gates_noadj/roi_gate_summary.csv \
  --a-name gated_regraph \
  --b-name noadj_gated \
  --out-dir preproc_v0/repetition_familiarity/results/interpretability/gates_noadj_vs_adj

python scripts/make_adjacency_ablation_artifacts.py --root "$ROOT"
