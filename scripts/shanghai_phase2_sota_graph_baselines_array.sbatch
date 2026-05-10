#!/usr/bin/env bash
#SBATCH --job-name=rg_p2_base
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
TASK=$((SLURM_ARRAY_TASK_ID / 8))
FOLD="${FOLDS[$((SLURM_ARRAY_TASK_ID % 8))]}"

ENC=mindeye2_shared
READOUT=flat
LAMBDA_SUBJ=0.0
GRAPH_BIAS=1.0
EXTRA_ARGS=()

case "$TASK" in
  0)
    # P2-1: MindEye2-style subject-specific mapper into a shared ROI latent.
    ENC=mindeye2_shared
    READOUT=flat
    ;;
  1)
    # P2-2: UMBRAE-style universal ROI encoder with subject embeddings.
    ENC=umbrae_subject
    READOUT=flat
    ;;
  2)
    # P2-3a: MindLink-inspired adversarial ROI-MLP+CLIP.
    ENC=roi_mlp
    READOUT=flat
    LAMBDA_SUBJ=0.1
    ;;
  3)
    # P2-3b: MindLink-inspired adversarial graph-biased ReGraph+CLIP.
    ENC=graph_bnt
    READOUT=gated_flat
    LAMBDA_SUBJ=0.1
    ;;
  4)
    # P2-4: same-parameter ROI transformer without graph adjacency.
    ENC=roi_transformer_noadj
    READOUT=gated_flat
    ;;
  5)
    # P2-5a: gated ROI-MLP control.
    ENC=gated_roi_mlp
    READOUT=flat
    ;;
  6)
    # P2-5b: gated token-MLP control, no transformer interaction.
    ENC=gated_token_mlp
    READOUT=flat
    ;;
  *)
    echo "Unknown TASK=$TASK"
    exit 2
    ;;
esac

python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold \
  --output-root preproc_v0/repetition_familiarity/results/phase2_sota_graph_baselines \
  --fold "$FOLD" \
  --graph-encoder "$ENC" \
  --readout "$READOUT" \
  --lambda-clip 2.0 \
  --lambda-subject-adv "$LAMBDA_SUBJ" \
  --graph-bias-scale "$GRAPH_BIAS" \
  --seed 11 \
  --device cpu \
  "${EXTRA_ARGS[@]}"
