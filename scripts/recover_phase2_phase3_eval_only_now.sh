#!/usr/bin/env bash
set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "[recover] phase2 graph_bnt subjadv fold_01"
python scripts/run_regraph_vlm_fold.py \
  --root "$ROOT" \
  --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold \
  --output-root preproc_v0/repetition_familiarity/results/phase2_sota_graph_baselines \
  --fold fold_01 \
  --graph-encoder graph_bnt \
  --readout gated_flat \
  --lambda-clip 2.0 \
  --lambda-subject-adv 0.1 \
  --seed 11 \
  --device cpu \
  --eval-only

declare -a P3=(
  "default fold_06"
  "topk20_corr fold_06"
  "shuffled fold_01"
  "shuffled fold_02"
  "shuffled fold_04"
  "shuffled fold_06"
  "shuffled fold_07"
  "no_adjacency fold_01"
  "no_adjacency fold_02"
)

for item in "${P3[@]}"; do
  mode=${item%% *}
  fold=${item##* }
  echo "[recover] phase3 mode=${mode} fold=${fold}"
  python scripts/run_regraph_vlm_fold.py \
    --root "$ROOT" \
    --dataset-root preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold \
    --output-root preproc_v0/repetition_familiarity/results/phase3_graph_ablation \
    --fold "$fold" \
    --graph-encoder graph_bnt \
    --readout gated_flat \
    --adjacency-mode "$mode" \
    --lambda-clip 2.0 \
    --seed 11 \
    --device cpu \
    --eval-only
done

echo "[recover] collect summaries"
python scripts/collect_regraph_vlm_metrics.py \
  --root "$ROOT" \
  --sources preproc_v0/repetition_familiarity/results/phase2_sota_graph_baselines \
  --out-csv preproc_v0/repetition_familiarity/results/phase2_sota_graph_baselines/regraph_vlm_summary.csv

python scripts/collect_regraph_vlm_metrics.py \
  --root "$ROOT" \
  --sources preproc_v0/repetition_familiarity/results/phase3_graph_ablation \
  --out-csv preproc_v0/repetition_familiarity/results/phase3_graph_ablation/regraph_vlm_summary.csv

python scripts/freeze_final_tables.py --root "$ROOT"
echo "[recover] done"
