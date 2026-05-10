#!/usr/bin/env bash
#SBATCH --job-name=rg_robust_sum
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python scripts/summarize_regraph_vlm_results.py --root "$ROOT" --results-dir preproc_v0/repetition_familiarity/results/heldout_image
python scripts/summarize_regraph_vlm_results.py --root "$ROOT" --results-dir preproc_v0/repetition_familiarity/results/heldout_image_random_embedding
python scripts/summarize_regraph_vlm_results.py --root "$ROOT" --results-dir preproc_v0/repetition_familiarity/results/cross_subject_hardneg_allfold_seed11
python scripts/summarize_regraph_vlm_results.py --root "$ROOT" --results-dir preproc_v0/repetition_familiarity/results/cross_subject_gated_graphonly_allfold
python scripts/freeze_final_tables.py --root "$ROOT"
python scripts/run_final_stat_tests.py --root "$ROOT"
python scripts/generate_final_report_figures.py --root "$ROOT"
