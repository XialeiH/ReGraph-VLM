#!/bin/bash
#SBATCH --job-name=laion_roi_export
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --output=slurm_logs/laion_roi_export_%j.out
#SBATCH --error=slurm_logs/laion_roi_export_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/export_laion_fmri_visual_roi_scalar4.py \
  --root external_validation/laion_fmri \
  --selection-manifest external_validation/laion_fmri/visual_roi_scalar4_laion/laion_download_manifest.json \
  --local-only \
  --metadata-dir external_validation/laion_fmri_probe/trial_metadata/tsv \
  --subjects sub-01 sub-03 sub-05 sub-06 sub-07 \
  --max-session 10 \
  --min-repeats 3 \
  --max-labels 200 \
  --max-rois 64 \
  --keep-downloads
