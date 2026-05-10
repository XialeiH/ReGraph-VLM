#!/usr/bin/env bash
#SBATCH --job-name=rg_cshard
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-17
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.err

set -euo pipefail
ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

FOLDS=(fold_01 fold_04)
SEEDS=(11 22 33)
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 6))
REM=$((SLURM_ARRAY_TASK_ID % 6))
FOLD="${FOLDS[$((REM / 3))]}"
SEED="${SEEDS[$((REM % 3))]}"
ENC=roi_mlp
READOUT=flat
OUT=roi_mlp
if [ "$MODEL_IDX" -eq 1 ]; then ENC=bnt_token_flat; READOUT=flat; OUT=bnt_flat; fi
if [ "$MODEL_IDX" -eq 2 ]; then ENC=bnt_token_flat; READOUT=gated_flat; OUT=bnt_gated; fi

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_hardneg/mixed_50_random_50_clip_hard \
  --output-root "preproc_v0/repetition_familiarity/results/cross_subject_hardneg/${OUT}" \
  --fold "$FOLD" \
  --graph-encoder "$ENC" \
  --readout "$READOUT" \
  --lambda-clip 2.0 \
  --seed "$SEED" \
  --device cpu
