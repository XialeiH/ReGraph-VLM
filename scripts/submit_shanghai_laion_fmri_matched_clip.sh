#!/bin/bash
set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
mkdir -p slurm_logs

manifest="external_validation/laion_fmri_matched_public_roi/visual_roi_scalar4_laion/laion_download_manifest.json"
if [[ ! -s "$manifest" ]]; then
  echo "Missing LAION matched-public-ROI download manifest: $manifest" >&2
  echo "Run scripts/run_laion_fmri_matched_public_roi_download_login.sh on the Shanghai login node first." >&2
  exit 1
fi

export_job=$(sbatch --parsable scripts/shanghai_laion_fmri_matched_public_roi_export.sbatch)
echo "Submitted LAION matched-public-ROI export job: $export_job"

augment_job=$(sbatch --parsable --dependency=afterok:"$export_job" scripts/shanghai_laion_fmri_matched_clip_augment.sbatch)
echo "Submitted LAION CLIP augmentation job: $augment_job"

train_job=$(sbatch --parsable --dependency=afterok:"$augment_job" scripts/shanghai_laion_fmri_matched_clip_array.sbatch)
echo "Submitted LAION CLIP training array: $train_job"

summary_job=$(sbatch --parsable --dependency=afterok:"$train_job" scripts/shanghai_laion_fmri_matched_clip_summary.sbatch)
echo "Submitted LAION CLIP summary job: $summary_job"
