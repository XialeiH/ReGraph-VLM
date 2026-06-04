#!/usr/bin/env bash
#SBATCH --job-name=rg_lnn_dyn
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=20:00:00
#SBATCH --array=0-71%4
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

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / 24))
REM=$((SLURM_ARRAY_TASK_ID % 24))
FOLD="${FOLDS[$((REM / 3))]}"
SEED="${SEEDS[$((REM % 3))]}"

BASE=roi_transformer_noadj
TEMPORAL=gru
case "$CONFIG_IDX" in
  0)
    BASE=roi_transformer_noadj
    TEMPORAL=gru
    ;;
  1)
    BASE=roi_transformer_noadj
    TEMPORAL=cfc
    ;;
  2)
    BASE=roi_mlp
    TEMPORAL=cfc
    ;;
  *)
    echo "Unknown CONFIG_IDX=$CONFIG_IDX"
    exit 2
    ;;
esac

echo "repeat_dynamic_lnn config=${CONFIG_IDX} base=${BASE} temporal=${TEMPORAL} fold=${FOLD} seed=${SEED}"

python scripts/run_repeat_dynamic_vlm_fold.py \
  --root "$ROOT" \
  --pair-dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold \
  --sequence-dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_allfold \
  --output-root preproc_v0/repetition_familiarity/results/repeat_dynamic_lnn \
  --fold "$FOLD" \
  --base-encoder "$BASE" \
  --temporal-module "$TEMPORAL" \
  --lambda-clip 2.0 \
  --seed "$SEED" \
  --device auto
