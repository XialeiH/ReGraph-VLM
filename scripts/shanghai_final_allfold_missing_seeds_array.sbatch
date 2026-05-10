#!/usr/bin/env bash
#SBATCH --job-name=rg_final8
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=18:00:00
#SBATCH --array=0-47
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
SEEDS=(22 33)

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 16))
REM=$((SLURM_ARRAY_TASK_ID % 16))
FOLD="${FOLDS[$((REM / 2))]}"
SEED="${SEEDS[$((REM % 2))]}"

ENC=roi_mlp
READOUT=flat
if [ "$MODEL_IDX" -eq 1 ]; then
  ENC=bnt_token_flat
  READOUT=flat
elif [ "$MODEL_IDX" -eq 2 ]; then
  ENC=bnt_token_flat
  READOUT=gated_flat
fi

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold \
  --output-root preproc_v0/repetition_familiarity/results/cross_subject_allfold_final \
  --fold "$FOLD" \
  --graph-encoder "$ENC" \
  --readout "$READOUT" \
  --lambda-clip 2.0 \
  --seed "$SEED" \
  --device cpu
