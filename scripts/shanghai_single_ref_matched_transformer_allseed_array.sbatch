#!/usr/bin/env bash
#SBATCH --job-name=rg_single_tf
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=18:00:00
#SBATCH --array=0-47%12
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.err

set -euo pipefail
ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

FOLDS=(fold_01 fold_02 fold_03 fold_04 fold_05 fold_06 fold_07 fold_08)
SEEDS=(11 22 33)

SEED_IDX=$((SLURM_ARRAY_TASK_ID / 16))
MODEL_IDX=$(((SLURM_ARRAY_TASK_ID % 16) / 8))
FOLD="${FOLDS[$((SLURM_ARRAY_TASK_ID % 8))]}"
SEED="${SEEDS[$SEED_IDX]}"

ENC=roi_transformer_noadj
MODEL_NAME=noadj_gated
if [ "$MODEL_IDX" -eq 1 ]; then
  ENC=bnt_token_flat
  MODEL_NAME=gated_regraph
fi

echo "single-ref matched transformer all-seed model=${MODEL_NAME} fold=${FOLD} seed=${SEED}"

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_single_ref_matched \
  --output-root preproc_v0/repetition_familiarity/results/single_ref_matched_allseed \
  --fold "$FOLD" \
  --graph-encoder "$ENC" \
  --readout gated_flat \
  --lambda-clip 2.0 \
  --seed "$SEED" \
  --device cpu
