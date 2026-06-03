#!/bin/bash
#SBATCH --job-name=laion_fs_clip
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --output=slurm_logs/laion_fs_clip_%j.out
#SBATCH --error=slurm_logs/laion_fs_clip_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/augment_laion_fmri_clip_embeddings.py \
  --data-dir external_validation/laion_fmri_freesurfer_aparc/freesurfer_aparc_scalar4_laion \
  --stimulus-dir external_validation/laion_fmri/stimuli \
  --subjects sub-01 sub-03 sub-05 sub-06 sub-07
