#!/usr/bin/env bash
#SBATCH --job-name=rg_cs8vlm
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=18:00:00
#SBATCH --array=0-23
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

# Core all-fold seed-11 models:
# 0: ROI-MLP+CLIP lambda=2.0
# 1: BNT/ReGraph graph-only lambda=0.0
# 2: BNT/ReGraph+CLIP lambda=2.0
MODELS=(roi_mlp bnt_token_flat bnt_token_flat)
LAMBDAS=(2.0 0.0 2.0)
FOLDS=(fold_01 fold_02 fold_03 fold_04 fold_05 fold_06 fold_07 fold_08)
SEED=11

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 8))
FOLD_IDX=$((SLURM_ARRAY_TASK_ID % 8))

MODEL="${MODELS[$MODEL_IDX]}"
LAMBDA="${LAMBDAS[$MODEL_IDX]}"
FOLD="${FOLDS[$FOLD_IDX]}"

echo "allfold_seed11 task=${SLURM_ARRAY_TASK_ID} model=${MODEL} lambda=${LAMBDA} fold=${FOLD} seed=${SEED}"

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold \
  --output-root preproc_v0/repetition_familiarity/results/cross_subject_allfold_seed11 \
  --fold "$FOLD" \
  --graph-encoder "$MODEL" \
  --lambda-clip "$LAMBDA" \
  --seed "$SEED" \
  --device cpu
