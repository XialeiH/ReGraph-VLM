#!/usr/bin/env bash
#SBATCH --job-name=rg_p3b_clean
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
TASK=$((SLURM_ARRAY_TASK_ID / 8))
FOLD="${FOLDS[$((SLURM_ARRAY_TASK_ID % 8))]}"

ENC=roi_mlp
READOUT=flat
ADJ=default

case "$TASK" in
  0)
    # ROI-MLP+CLIP.
    ENC=roi_mlp
    READOUT=flat
    ADJ=default
    ;;
  1)
    # ROI Transformer no adjacency + flatten.
    ENC=roi_transformer_noadj
    READOUT=flat
    ADJ=default
    ;;
  2)
    # ROI Transformer no adjacency + gated-flatten.
    ENC=roi_transformer_noadj
    READOUT=gated_flat
    ADJ=default
    ;;
  3)
    # ReGraph top-k correlation adjacency + gated-flatten.
    ENC=graph_bnt
    READOUT=gated_flat
    ADJ=topk20_corr
    ;;
  4)
    # ReGraph identity adjacency + gated-flatten.
    ENC=graph_bnt
    READOUT=gated_flat
    ADJ=identity
    ;;
  5)
    # ReGraph shuffled adjacency + gated-flatten.
    ENC=graph_bnt
    READOUT=gated_flat
    ADJ=shuffled
    ;;
  *)
    echo "Unknown TASK=$TASK"
    exit 2
    ;;
esac

echo "phase3b clean ablation task=${TASK} fold=${FOLD} encoder=${ENC} readout=${READOUT} adjacency=${ADJ}"

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold \
  --output-root preproc_v0/repetition_familiarity/results/phase3b_clean_graph_ablation \
  --fold "$FOLD" \
  --graph-encoder "$ENC" \
  --readout "$READOUT" \
  --adjacency-mode "$ADJ" \
  --lambda-clip 2.0 \
  --seed 11 \
  --device cpu
