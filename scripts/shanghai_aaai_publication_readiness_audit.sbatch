#!/usr/bin/env bash
#SBATCH --job-name=rg_aaai_audit
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python scripts/audit_aaai_publication_readiness.py \
  --final-tables-dir preproc_v0/repetition_familiarity/results/final_tables \
  --external-summary-dir external_validation/summary

python scripts/audit_manuscript_publication_claims.py \
  --tex reports/neurips_report/may30.tex \
  --final-tables-dir preproc_v0/repetition_familiarity/results/final_tables \
  --output-dir preproc_v0/repetition_familiarity/results/final_tables
