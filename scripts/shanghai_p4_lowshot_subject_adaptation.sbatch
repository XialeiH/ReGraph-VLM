#!/bin/bash
#SBATCH --job-name=p4_lowshot
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/p4_lowshot_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/p4_lowshot_%j.err

set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/analyze_lowshot_subject_adaptation.py \
  --root "$PWD" \
  --folds fold_01 fold_02 fold_03 fold_04 fold_05 fold_06 fold_07 fold_08 \
  --seeds 11 22 33 \
  --shots 0 10 30 60 120 \
  --ridge 1.0 \
  --device cpu
