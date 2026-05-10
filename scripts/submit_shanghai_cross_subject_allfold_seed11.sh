#!/usr/bin/env bash
set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p slurm_logs

DATA_JOB="${1:-2161119}"

ARRAY_JOB=$(sbatch --parsable --dependency=afterok:${DATA_JOB} scripts/shanghai_cross_subject_allfold_seed11_array.sbatch)
SUMMARY_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} scripts/shanghai_cross_subject_allfold_seed11_summary.sbatch)

cat <<EOF
allfold_dataset_dependency=${DATA_JOB}
allfold_seed11_array_job=${ARRAY_JOB}
allfold_seed11_summary_job=${SUMMARY_JOB}
EOF

squeue -u xh2906
