#!/usr/bin/env bash
#SBATCH --job-name=rg_crossnce
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.err

set -euo pipefail
ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

LAMBDAS=(0.1 0.3 1.0)
FOLDS=(fold_01 fold_04)
LAMBDA_CROSS="${LAMBDAS[$((SLURM_ARRAY_TASK_ID / 2))]}"
FOLD="${FOLDS[$((SLURM_ARRAY_TASK_ID % 2))]}"
python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject \
  --output-root preproc_v0/repetition_familiarity/results/cross_subject_cross_infonce \
  --fold "$FOLD" \
  --graph-encoder bnt_token_flat \
  --readout gated_flat \
  --lambda-clip 2.0 \
  --lambda-cross "$LAMBDA_CROSS" \
  --seed 11 \
  --device cpu
