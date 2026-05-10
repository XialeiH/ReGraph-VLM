#!/usr/bin/env bash
set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p slurm_logs

FREEZE_JOB=$(sbatch --parsable scripts/shanghai_freeze_current_stage.sbatch)
BUILD_JOB=$(sbatch --parsable scripts/shanghai_build_cross_subject.sbatch)
RAW_JOB=$(sbatch --parsable --dependency=afterok:${BUILD_JOB} scripts/shanghai_cross_subject_raw.sbatch)
ARRAY_JOB=$(sbatch --parsable --dependency=afterok:${BUILD_JOB} scripts/shanghai_cross_subject_vlm_array.sbatch)
SUMMARY_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} scripts/shanghai_cross_subject_summary.sbatch)

cat <<EOF
freeze_job=${FREEZE_JOB}
build_job=${BUILD_JOB}
raw_job=${RAW_JOB}
array_job=${ARRAY_JOB}
summary_job=${SUMMARY_JOB}
EOF

squeue -u xh2906
