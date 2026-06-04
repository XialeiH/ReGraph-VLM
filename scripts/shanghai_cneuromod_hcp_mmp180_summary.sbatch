#!/bin/bash
#SBATCH --job-name=cnm_hcp_sum
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --output=slurm_logs/cnm_hcp_sum_%j.out
#SBATCH --error=slurm_logs/cnm_hcp_sum_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:$PWD/scripts:${PYTHONPATH:-}"

python scripts/summarize_cneuromod_hcp_mmp_external.py \
  --root external_validation/cneuromod_things/hcp_mmp180_scalar4_all4/trained_external \
  --out-dir external_validation/summary/cneuromod_hcp_mmp180
