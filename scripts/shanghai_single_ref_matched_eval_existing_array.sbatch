#!/usr/bin/env bash
#SBATCH --job-name=rg_single_eval
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-71
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.err

set -euo pipefail
ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

FOLDS=(fold_01 fold_02 fold_03 fold_04 fold_05 fold_06 fold_07 fold_08)
SEEDS=(11 22 33)

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 24))
REM=$((SLURM_ARRAY_TASK_ID % 24))
SEED="${SEEDS[$((REM / 8))]}"
FOLD="${FOLDS[$((REM % 8))]}"

ENC=roi_mlp
READOUT=flat
MODEL_DIR=roi_mlp_clip
SRC_ROOT=preproc_v0/repetition_familiarity/results/cross_subject_allfold_final
if [ "$SEED" -eq 11 ]; then
  SRC_ROOT=preproc_v0/repetition_familiarity/results/cross_subject_allfold_seed11
fi

if [ "$MODEL_IDX" -eq 1 ]; then
  ENC=roi_transformer_noadj
  READOUT=gated_flat
  MODEL_DIR=roi_transformer_noadj_gated_flat_clip
  SRC_ROOT=preproc_v0/repetition_familiarity/results/phase3c_noadj_gated_final
  if [ "$SEED" -eq 11 ]; then
    SRC_ROOT=preproc_v0/repetition_familiarity/results/phase3b_clean_graph_ablation
  fi
elif [ "$MODEL_IDX" -eq 2 ]; then
  ENC=bnt_token_flat
  READOUT=gated_flat
  MODEL_DIR=bnt_token_flat_gated_flat_clip
  SRC_ROOT=preproc_v0/repetition_familiarity/results/cross_subject_allfold_final
  if [ "$SEED" -eq 11 ]; then
    SRC_ROOT=preproc_v0/repetition_familiarity/results/cross_subject_gated_allfold_seed11
  fi
fi

OUT_ROOT=preproc_v0/repetition_familiarity/results/single_ref_matched_eval_existing
TARGET_DIR="$OUT_ROOT/$MODEL_DIR/lambda_2/$FOLD/seed_$SEED"
SRC_CKPT="$SRC_ROOT/$MODEL_DIR/lambda_2/$FOLD/seed_$SEED/checkpoint.pt"
mkdir -p "$TARGET_DIR"
cp "$SRC_CKPT" "$TARGET_DIR/checkpoint.pt"

echo "single-ref eval existing model=${MODEL_DIR} fold=${FOLD} seed=${SEED} src=${SRC_CKPT}"

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_single_ref_matched \
  --output-root "$OUT_ROOT" \
  --fold "$FOLD" \
  --graph-encoder "$ENC" \
  --readout "$READOUT" \
  --lambda-clip 2.0 \
  --seed "$SEED" \
  --device cpu \
  --eval-only
