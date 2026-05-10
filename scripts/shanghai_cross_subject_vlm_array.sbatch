#!/usr/bin/env bash
#SBATCH --job-name=rg_csvlm
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-23
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

MODELS=(roi_mlp roi_mlp bnt_token_flat bnt_token_flat)
LAMBDAS=(0.0 2.0 0.0 2.0)
FOLDS=(fold_01 fold_04)
SEEDS=(11 22 33)

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 6))
REM=$((SLURM_ARRAY_TASK_ID % 6))
FOLD_IDX=$((REM / 3))
SEED_IDX=$((REM % 3))

MODEL="${MODELS[$MODEL_IDX]}"
LAMBDA="${LAMBDAS[$MODEL_IDX]}"
FOLD="${FOLDS[$FOLD_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "task=${SLURM_ARRAY_TASK_ID} model=${MODEL} lambda=${LAMBDA} fold=${FOLD} seed=${SEED}"

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject \
  --output-root preproc_v0/repetition_familiarity/results/cross_subject_regraph_vlm \
  --fold "$FOLD" \
  --graph-encoder "$MODEL" \
  --lambda-clip "$LAMBDA" \
  --seed "$SEED" \
  --device cpu
