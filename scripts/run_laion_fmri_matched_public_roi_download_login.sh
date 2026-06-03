#!/bin/bash
set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/export_laion_fmri_visual_roi_scalar4.py \
  --root external_validation/laion_fmri_matched_public_roi \
  --metadata-dir external_validation/laion_fmri_probe/trial_metadata/tsv \
  --subjects sub-01 sub-03 sub-05 sub-06 sub-07 \
  --max-session 30 \
  --min-repeats 3 \
  --max-labels 0 \
  --max-rois 0 \
  --roi-categories laion retinotopy face object place body motion character \
  --download-only
