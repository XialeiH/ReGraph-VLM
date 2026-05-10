#!/usr/bin/env bash
#SBATCH --job-name=rg_freezeall
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python scripts/freeze_cross_subject_results.py --root "$ROOT"
python scripts/qc_cross_subject_no_leakage.py --root "$ROOT"
python scripts/report_cross_subject_chance_levels.py --root "$ROOT"
python scripts/summarize_current_all_results.py --root "$ROOT"
