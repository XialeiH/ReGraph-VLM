#!/usr/bin/env bash
#SBATCH --job-name=rg_single_ref
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

FOLDS=(fold_01 fold_02 fold_03 fold_04 fold_05 fold_06 fold_07 fold_08)
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 8))
FOLD="${FOLDS[$((SLURM_ARRAY_TASK_ID % 8))]}"

ENC=roi_mlp
READOUT=flat
MODEL_NAME=roi_mlp
if [ "$MODEL_IDX" -eq 1 ]; then
  ENC=roi_transformer_noadj
  READOUT=gated_flat
  MODEL_NAME=noadj_gated
elif [ "$MODEL_IDX" -eq 2 ]; then
  ENC=bnt_token_flat
  READOUT=gated_flat
  MODEL_NAME=gated_regraph
fi

echo "single-ref matched seed11 model=${MODEL_NAME} fold=${FOLD}"

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_single_ref_matched \
  --output-root preproc_v0/repetition_familiarity/results/single_ref_matched_seed11 \
  --fold "$FOLD" \
  --graph-encoder "$ENC" \
  --readout "$READOUT" \
  --lambda-clip 2.0 \
  --seed 11 \
  --device cpu
