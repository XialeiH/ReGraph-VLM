#!/bin/bash
set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
mkdir -p slurm_logs

export_job=$(sbatch --parsable scripts/shanghai_cneuromod_hcp_mmp180_export.sbatch)
echo "Submitted CNeuroMod HCP-MMP180 export job: $export_job"

train_job=$(sbatch --parsable --dependency=afterok:"$export_job" scripts/shanghai_cneuromod_hcp_mmp180_array.sbatch)
echo "Submitted CNeuroMod HCP-MMP180 training array: $train_job"

summary_job=$(sbatch --parsable --dependency=afterok:"$train_job" scripts/shanghai_cneuromod_hcp_mmp180_summary.sbatch)
echo "Submitted CNeuroMod HCP-MMP180 summary job: $summary_job"
