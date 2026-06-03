#!/bin/bash
#SBATCH --job-name=laion_clip_sum
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --output=slurm_logs/laion_clip_sum_%j.out
#SBATCH --error=slurm_logs/laion_clip_sum_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/summarize_laion_fmri_external_results.py \
  --root external_validation/laion_fmri_matched_public_roi/visual_roi_scalar4_laion/trained_external_clip \
  --out-dir external_validation/summary/laion_fmri_matched_clip
