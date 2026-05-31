#!/bin/bash
#SBATCH --job-name=laion_ext_sum
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --output=slurm_logs/laion_ext_sum_%j.out
#SBATCH --error=slurm_logs/laion_ext_sum_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/summarize_laion_fmri_external_results.py \
  --root external_validation/laion_fmri/visual_roi_scalar4_laion/trained_external \
  --out-dir external_validation/summary
