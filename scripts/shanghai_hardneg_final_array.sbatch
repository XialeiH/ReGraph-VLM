#!/usr/bin/env bash
#SBATCH --job-name=rg_hn_final
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-55
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
SEEDS2=(22 33)
SEEDS3=(11 22 33)
IDX=$SLURM_ARRAY_TASK_ID

if [ "$IDX" -lt 16 ]; then
  ENC=roi_mlp
  READOUT=flat
  FOLD="${FOLDS[$((IDX / 2))]}"
  SEED="${SEEDS2[$((IDX % 2))]}"
elif [ "$IDX" -lt 40 ]; then
  J=$((IDX - 16))
  ENC=bnt_token_flat
  READOUT=flat
  FOLD="${FOLDS[$((J / 3))]}"
  SEED="${SEEDS3[$((J % 3))]}"
else
  J=$((IDX - 40))
  ENC=bnt_token_flat
  READOUT=gated_flat
  FOLD="${FOLDS[$((J / 2))]}"
  SEED="${SEEDS2[$((J % 2))]}"
fi

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_hardneg_allfold/mixed_50_random_50_clip_hard \
  --output-root preproc_v0/repetition_familiarity/results/cross_subject_hardneg_allfold_seed11 \
  --fold "$FOLD" \
  --graph-encoder "$ENC" \
  --readout "$READOUT" \
  --lambda-clip 2.0 \
  --seed "$SEED" \
  --device cpu
