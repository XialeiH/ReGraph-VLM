#!/bin/bash
#SBATCH --job-name=plot_fsavg_rxn
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:45:00
#SBATCH --mem=16G
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/plot_fsavg_rxn_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/slurm_logs/plot_fsavg_rxn_%j.err

set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
source scripts/setup_freesurfer_shanghai.sh
export PYTHONPATH="$PWD/scripts:$PWD:${PYTHONPATH:-}"

ROOT="$PWD/preproc_v0/repetition_familiarity"
DATA="$ROOT/datasets/scalar4_T3"
EXAMPLES="$ROOT/figures/stimulus_brain_reaction_examples/example_stimuli.csv"
OUT="$ROOT/figures/stimulus_brain_fsaverage_examples"

mkdir -p "$OUT"

python scripts/plot_fsaverage_brain_reaction.py \
  --dataset_root "$DATA" \
  --example_csv "$EXAMPLES" \
  --out_dir "$OUT" \
  --clip_metadata_csv "$ROOT/vlm/clip_embeddings/clip_image_metadata.csv" \
  --feature mean_beta \
  --max_examples 4 \
  --face_stride 24 \
  --mesh_source freesurfer \
  --make_png

echo "Saved fsaverage surface-style figures to $OUT"
