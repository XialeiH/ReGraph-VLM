#!/bin/bash
set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/export_laion_fmri_visual_roi_scalar4.py \
  --root external_validation/laion_fmri \
  --metadata-dir external_validation/laion_fmri_probe/trial_metadata/tsv \
  --subjects sub-01 sub-03 sub-05 sub-06 sub-07 \
  --max-session 10 \
  --min-repeats 3 \
  --max-labels 200 \
  --max-rois 64 \
  --download-only
