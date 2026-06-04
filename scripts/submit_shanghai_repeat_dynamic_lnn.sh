#!/usr/bin/env bash
set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p slurm_logs

ARRAY_JOB=$(sbatch --parsable scripts/shanghai_repeat_dynamic_lnn_array.sbatch)
SUMMARY_JOB=$(sbatch --parsable --dependency=afterany:"$ARRAY_JOB" scripts/shanghai_repeat_dynamic_lnn_summary.sbatch)

echo "Submitted repeat-dynamic LNN array: ${ARRAY_JOB}"
echo "Submitted repeat-dynamic LNN summary: ${SUMMARY_JOB}"
