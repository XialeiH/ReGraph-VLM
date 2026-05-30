#!/bin/bash
set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
mkdir -p slurm_logs

export_job=$(sbatch --parsable scripts/shanghai_things_fmri_export_after_download.sbatch)
echo "Submitted THINGS-fMRI export job: $export_job"

train_job=$(sbatch --parsable --dependency=afterok:"$export_job" scripts/shanghai_things_fmri_external_array.sbatch)
echo "Submitted THINGS-fMRI training array: $train_job"
