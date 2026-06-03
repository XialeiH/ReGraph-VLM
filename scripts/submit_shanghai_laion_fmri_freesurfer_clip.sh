#!/bin/bash
set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
mkdir -p slurm_logs

manifest="external_validation/laion_fmri_matched_public_roi/visual_roi_scalar4_laion/laion_download_manifest.json"
if [[ ! -s "$manifest" ]]; then
  echo "Missing LAION matched-public-ROI download manifest: $manifest" >&2
  exit 1
fi

export_job=$(sbatch --parsable scripts/shanghai_laion_fmri_freesurfer_aparc_export.sbatch)
echo "Submitted LAION FreeSurfer aparc export job: $export_job"

augment_job=$(sbatch --parsable --dependency=afterok:"$export_job" scripts/shanghai_laion_fmri_freesurfer_clip_augment.sbatch)
echo "Submitted LAION FreeSurfer CLIP augmentation job: $augment_job"

train_job=$(sbatch --parsable --dependency=afterok:"$augment_job" scripts/shanghai_laion_fmri_freesurfer_clip_array.sbatch)
echo "Submitted LAION FreeSurfer CLIP training array: $train_job"

summary_job=$(sbatch --parsable --dependency=afterok:"$train_job" scripts/shanghai_laion_fmri_freesurfer_clip_summary.sbatch)
echo "Submitted LAION FreeSurfer CLIP summary job: $summary_job"
