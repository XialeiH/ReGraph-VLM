#!/bin/bash
#SBATCH --job-name=laion_clip_aug
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --output=slurm_logs/laion_clip_aug_%j.out
#SBATCH --error=slurm_logs/laion_clip_aug_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/augment_laion_fmri_clip_embeddings.py \
  --data-dir external_validation/laion_fmri_matched_public_roi/visual_roi_scalar4_laion \
  --stimulus-dir external_validation/laion_fmri/stimuli \
  --subjects sub-01 sub-03 sub-05 sub-06 sub-07
