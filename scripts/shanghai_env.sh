#!/usr/bin/env bash
set -euo pipefail

module load anaconda3/2023.09-0
source /gpfsnyu/scratch/xh2906/venvs/regraph_cpu/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
