#!/usr/bin/env bash
#SBATCH --job-name=rg_p3b_sum
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python scripts/audit_phase2_phase3_configs.py --root "$ROOT"

python scripts/collect_regraph_vlm_metrics.py \
  --root "$ROOT" \
  --sources preproc_v0/repetition_familiarity/results/phase3b_clean_graph_ablation \
  --out-csv preproc_v0/repetition_familiarity/results/phase3b_clean_graph_ablation/regraph_vlm_summary.csv

python scripts/freeze_final_tables.py --root "$ROOT"
