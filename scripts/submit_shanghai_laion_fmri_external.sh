#!/bin/bash
set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
mkdir -p slurm_logs

if [[ ! -s external_validation/laion_fmri/visual_roi_scalar4_laion/laion_download_manifest.json ]]; then
  echo "Missing LAION download manifest. Run scripts/run_laion_fmri_download_login.sh on the Shanghai login node first." >&2
  exit 1
fi

export_job=$(sbatch --parsable scripts/shanghai_laion_fmri_visual_roi_export.sbatch)
echo "Submitted LAION-fMRI visual-ROI export job: $export_job"

train_job=$(sbatch --parsable --dependency=afterok:"$export_job" scripts/shanghai_laion_fmri_external_array.sbatch)
echo "Submitted LAION-fMRI external training array: $train_job"

summary_job=$(sbatch --parsable --dependency=afterok:"$train_job" scripts/shanghai_laion_fmri_external_summary.sbatch)
echo "Submitted LAION-fMRI external summary job: $summary_job"
