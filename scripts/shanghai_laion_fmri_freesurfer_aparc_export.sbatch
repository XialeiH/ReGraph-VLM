#!/bin/bash
#SBATCH --job-name=laion_fs_export
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --output=slurm_logs/laion_fs_export_%j.out
#SBATCH --error=slurm_logs/laion_fs_export_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/export_laion_fmri_freesurfer_scalar4.py \
  --root external_validation/laion_fmri_freesurfer_aparc \
  --metadata-dir external_validation/laion_fmri_probe/trial_metadata/tsv \
  --source-download-dir external_validation/laion_fmri_matched_public_roi/downloads \
  --selection-manifest external_validation/laion_fmri_matched_public_roi/visual_roi_scalar4_laion/laion_download_manifest.json \
  --subjects sub-01 sub-03 sub-05 sub-06 sub-07 \
  --max-session 30 \
  --min-repeats 3 \
  --max-labels 0
