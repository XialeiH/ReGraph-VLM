#!/usr/bin/env bash
set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p slurm_logs

FREEZE_QC_JOB=$(sbatch --parsable scripts/shanghai_cross_subject_freeze_qc.sbatch)
SHUF_BUILD_JOB=$(sbatch --parsable scripts/shanghai_build_shuffled_clip_control.sbatch)
SHUF_ARRAY_JOB=$(sbatch --parsable --dependency=afterok:${SHUF_BUILD_JOB} scripts/shanghai_cross_subject_shuffled_clip_array.sbatch)
SHUF_SUM_JOB=$(sbatch --parsable --dependency=afterok:${SHUF_ARRAY_JOB} scripts/shanghai_cross_subject_shuffled_clip_summary.sbatch)
ALLFOLD_DATA_JOB=$(sbatch --parsable scripts/shanghai_build_cross_subject_allfold.sbatch)

cat <<EOF
freeze_qc_job=${FREEZE_QC_JOB}
shuffled_clip_dataset_job=${SHUF_BUILD_JOB}
shuffled_clip_array_job=${SHUF_ARRAY_JOB}
shuffled_clip_summary_job=${SHUF_SUM_JOB}
allfold_dataset_job=${ALLFOLD_DATA_JOB}
EOF

squeue -u xh2906
