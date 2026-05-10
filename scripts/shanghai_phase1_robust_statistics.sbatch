#!/usr/bin/env bash
#SBATCH --job-name=rg_p1_stats
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python scripts/freeze_final_tables.py --root "$ROOT"
python scripts/run_final_stat_tests.py --root "$ROOT"
python scripts/run_phase1_robust_statistics.py \
  --root "$ROOT" \
  --summary-csv preproc_v0/repetition_familiarity/results/cross_subject_allfold_final/regraph_vlm_summary.csv \
  --output-dir preproc_v0/repetition_familiarity/results/phase1_robust_statistics \
  --n-bootstrap 10000
