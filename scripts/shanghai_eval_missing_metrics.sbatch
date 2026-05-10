#!/usr/bin/env bash
#SBATCH --job-name=rg_eval_missing
#SBATCH --partition=gpu,debug,acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-6
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.err

set -euo pipefail
ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

TYPE=graphonly
FOLD=fold_03
SEED=22
DATASET=preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold
OUT=preproc_v0/repetition_familiarity/results/cross_subject_gated_graphonly_allfold
ENC=bnt_token_flat
READOUT=gated_flat
LAMBDA=0.0

case "$SLURM_ARRAY_TASK_ID" in
  0) FOLD=fold_03; SEED=22 ;;
  1) FOLD=fold_03; SEED=33 ;;
  2) FOLD=fold_04; SEED=11 ;;
  3) FOLD=fold_05; SEED=33 ;;
  4) FOLD=fold_08; SEED=22 ;;
  5) FOLD=fold_08; SEED=33 ;;
  6)
    TYPE=hardneg_flat
    FOLD=fold_08
    SEED=22
    DATASET=preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_hardneg_allfold/mixed_50_random_50_clip_hard
    OUT=preproc_v0/repetition_familiarity/results/cross_subject_hardneg_allfold_seed11
    ENC=bnt_token_flat
    READOUT=flat
    LAMBDA=2.0
    ;;
esac

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root "$DATASET" \
  --output-root "$OUT" \
  --fold "$FOLD" \
  --graph-encoder "$ENC" \
  --readout "$READOUT" \
  --lambda-clip "$LAMBDA" \
  --seed "$SEED" \
  --device cpu \
  --eval-only
