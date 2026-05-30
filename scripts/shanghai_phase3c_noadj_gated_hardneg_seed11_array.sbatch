#!/usr/bin/env bash
#SBATCH --job-name=rg_p3c_hard
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=18:00:00
#SBATCH --array=0-7
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
FOLD="${FOLDS[$SLURM_ARRAY_TASK_ID]}"

echo "phase3c hardneg noadj gated seed=11 fold=${FOLD}"

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_hardneg_allfold/mixed_50_random_50_clip_hard \
  --output-root preproc_v0/repetition_familiarity/results/phase3c_noadj_gated_hardneg \
  --fold "$FOLD" \
  --graph-encoder roi_transformer_noadj \
  --readout gated_flat \
  --lambda-clip 2.0 \
  --seed 11 \
  --device cpu
