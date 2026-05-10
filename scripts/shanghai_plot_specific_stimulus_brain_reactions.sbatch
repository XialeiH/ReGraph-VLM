#!/usr/bin/env bash
#SBATCH --job-name=plot_stim_rxn
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT/scripts:$ROOT:${PYTHONPATH:-}"

DATA=${ROOT}/preproc_v0/repetition_familiarity/datasets/scalar4_T3
CLIP_DATA=${ROOT}/preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip
RESULTS=${ROOT}/preproc_v0/repetition_familiarity/results/cross_subject_gated_allfold_seed11/regraph_vlm_summary.csv
FALLBACK_RESULTS=${ROOT}/preproc_v0/repetition_familiarity/results/cross_subject_allfold_seed11/regraph_vlm_summary.csv
OUT=${ROOT}/preproc_v0/repetition_familiarity/figures/stimulus_brain_reaction_examples

if [ ! -f "$RESULTS" ]; then
  RESULTS="$FALLBACK_RESULTS"
fi

mkdir -p "$OUT"

ATLAS=$(find "$ROOT" /gpfsnyu/scratch/xh2906 -name "HCP_MMP1.nii.gz" 2>/dev/null | head -n 1 || true)

echo "ROOT=$ROOT"
echo "DATA=$DATA"
echo "CLIP_DATA=$CLIP_DATA"
echo "RESULTS=$RESULTS"
echo "ATLAS=${ATLAS:-not_found_optional}"
echo "OUT=$OUT"

python scripts/select_example_stimuli_for_brain_reaction.py \
  --dataset_root "$DATA" \
  --clip_dataset_root "$CLIP_DATA" \
  --results_csv "$RESULTS" \
  --folds fold_01 fold_04 \
  --criteria high_stability strong_delta model_success model_failure \
  --n_per_criterion 2 \
  --out_csv "$OUT/example_stimuli.csv"

python scripts/plot_stimulus_brain_reaction.py \
  --dataset_root "$DATA" \
  --clip_dataset_root "$CLIP_DATA" \
  --atlas "${ATLAS:-}" \
  --example_csv "$OUT/example_stimuli.csv" \
  --out_dir "$OUT" \
  --feature mean_beta \
  --plot_repeats \
  --plot_delta31 \
  --plot_subject_grid \
  --plot_roi_heatmap \
  --plot_similarity_matrix \
  --make_png \
  --make_svg

find "$OUT" -maxdepth 1 -type f | sort
