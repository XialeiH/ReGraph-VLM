#!/usr/bin/env bash
#SBATCH --job-name=rg_p3c_hold
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=18:00:00
#SBATCH --array=0-5
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
FOLD="${FOLDS[$((SLURM_ARRAY_TASK_ID % 2))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID / 2))]}"

echo "phase3c heldout noadj gated seed=${SEED} fold=${FOLD}"

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_imageheldout \
  --output-root preproc_v0/repetition_familiarity/results/phase3c_noadj_gated_heldout \
  --fold "$FOLD" \
  --graph-encoder roi_transformer_noadj \
  --readout gated_flat \
  --lambda-clip 2.0 \
  --seed "$SEED" \
  --device cpu
