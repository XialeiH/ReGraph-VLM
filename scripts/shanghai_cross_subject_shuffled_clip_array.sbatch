#!/usr/bin/env bash
#SBATCH --job-name=rg_csshvlm
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

MODELS=(roi_mlp bnt_token_flat)
FOLDS=(fold_01 fold_04)
SEEDS=(11 22 33)

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 6))
REM=$((SLURM_ARRAY_TASK_ID % 6))
FOLD_IDX=$((REM / 3))
SEED_IDX=$((REM % 3))

MODEL="${MODELS[$MODEL_IDX]}"
FOLD="${FOLDS[$FOLD_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "shuffled_clip task=${SLURM_ARRAY_TASK_ID} model=${MODEL} fold=${FOLD} seed=${SEED}"

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_shuffled_clip \
  --output-root preproc_v0/repetition_familiarity/results/cross_subject_clip_controls/shuffled_clip \
  --fold "$FOLD" \
  --graph-encoder "$MODEL" \
  --lambda-clip 2.0 \
  --seed "$SEED" \
  --device cpu
