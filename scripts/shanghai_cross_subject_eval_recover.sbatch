#!/usr/bin/env bash
#SBATCH --job-name=rg_cseval
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

SPECS=(
  "bnt_token_flat 0.0 fold_01 11"
  "bnt_token_flat 0.0 fold_01 22"
  "bnt_token_flat 0.0 fold_04 11"
  "bnt_token_flat 0.0 fold_04 33"
  "bnt_token_flat 2.0 fold_04 33"
)

read -r MODEL LAMBDA FOLD SEED <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"

echo "recover task=${SLURM_ARRAY_TASK_ID} model=${MODEL} lambda=${LAMBDA} fold=${FOLD} seed=${SEED}"

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject \
  --output-root preproc_v0/repetition_familiarity/results/cross_subject_regraph_vlm \
  --fold "$FOLD" \
  --graph-encoder "$MODEL" \
  --lambda-clip "$LAMBDA" \
  --seed "$SEED" \
  --device cpu \
  --eval-only
