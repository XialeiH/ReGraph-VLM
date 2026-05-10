#!/usr/bin/env bash
#SBATCH --job-name=rg_gatectrl
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%A_%a.err

set -euo pipefail
ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

MODES=(shuffled_clip random_embedding)
DATASETS=(scalar4_T3_clip_cross_subject_shuffled_clip scalar4_T3_clip_cross_subject_random_embedding)
FOLDS=(fold_01 fold_04)
MODE_IDX=$((SLURM_ARRAY_TASK_ID / 2))
FOLD="${FOLDS[$((SLURM_ARRAY_TASK_ID % 2))]}"
MODE="${MODES[$MODE_IDX]}"
DATASET="${DATASETS[$MODE_IDX]}"
python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root "preproc_v0/repetition_familiarity/datasets/${DATASET}" \
  --output-root "preproc_v0/repetition_familiarity/results/regraph_readout_ablation/gated_clip_controls/${MODE}" \
  --fold "$FOLD" \
  --graph-encoder bnt_token_flat \
  --readout gated_flat \
  --lambda-clip 2.0 \
  --seed 11 \
  --device cpu
