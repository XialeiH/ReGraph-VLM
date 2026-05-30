#!/usr/bin/env bash
#SBATCH --job-name=rg_p3_graph
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=18:00:00
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
ADJ_MODES=(default topk20_corr dense_corr identity random shuffled no_adjacency)

ADJ_MODE="${ADJ_MODES[$((SLURM_ARRAY_TASK_ID / 8))]}"
FOLD="${FOLDS[$((SLURM_ARRAY_TASK_ID % 8))]}"

echo "phase3 graph ablation task=${SLURM_ARRAY_TASK_ID} fold=${FOLD} adjacency=${ADJ_MODE}"

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold \
  --output-root preproc_v0/repetition_familiarity/results/phase3_graph_ablation \
  --fold "$FOLD" \
  --graph-encoder graph_bnt \
  --readout gated_flat \
  --adjacency-mode "$ADJ_MODE" \
  --lambda-clip 2.0 \
  --seed 11 \
  --device cpu
