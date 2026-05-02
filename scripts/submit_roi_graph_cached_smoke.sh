#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scratch/xh2906/final_project_nsd/v0_shared_unit}"
PY="${PY:-/scratch/xh2906/conda/envs/py310/bin/python}"
ACCOUNT="${ACCOUNT:-torch_pr_482_general}"
OUT="$ROOT/preproc_v0/roi_graph_atlas_v1"
FOLDROOT="$ROOT/preproc_v0/all8_ge2_766"
NODESET="$OUT/roi_node_set_v1.json"
CACHE="$OUT/roi_feature_cache"
GRAPH_ROOT="$OUT/cached_smoke"
SANITY_ROOT="$OUT/roi_graph_sanity_cached_smoke"
BRAIN_ROOT="$OUT/braingnn_nsd_cached_smoke"
LOG_DIR="$ROOT/logs"

mkdir -p "$CACHE" "$GRAPH_ROOT" "$SANITY_ROOT" "$BRAIN_ROOT" "$LOG_DIR"

export_jobs=()
for sid in 01 02 03 04 05 06 07 08; do
  subj="subj${sid}"
  job=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --job-name="roic${sid}" \
    --partition=cs \
    --time=24:00:00 \
    --cpus-per-task=8 \
    --mem=128G \
    --output="$LOG_DIR/roic_${sid}_%j.out" \
    --error="$LOG_DIR/roic_${sid}_%j.err" \
    --wrap="$PY scripts/export_subject_atlas_roi_features.py --root $ROOT --subject $subj --node-set $NODESET --output-dir $CACHE")
  export_jobs+=("$job")
done
export_dep="$(IFS=:; echo "${export_jobs[*]}")"

dataset_jobs=()
for fold_id in 01 04; do
  fold="fold_${fold_id}"
  graph_dir="$GRAPH_ROOT/$fold"
  mkdir -p "$graph_dir"
  job=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --dependency="afterok:$export_dep" \
    --job-name="roidc${fold_id}" \
    --partition=cs \
    --time=04:00:00 \
    --cpus-per-task=8 \
    --mem=64G \
    --output="$LOG_DIR/roidc_${fold_id}_%j.out" \
    --error="$LOG_DIR/roidc_${fold_id}_%j.err" \
    --wrap="$PY scripts/build_roi_graph_dataset_from_cache.py --root $ROOT --fold-root $FOLDROOT --fold-name $fold --node-set $NODESET --cache-dir $CACHE --output-dir $graph_dir")
  dataset_jobs+=("$job")
done
dataset_dep="$(IFS=:; echo "${dataset_jobs[*]}")"

sanity_jobs=()
for fold_id in 01 04; do
  fold="fold_${fold_id}"
  graph_dir="$GRAPH_ROOT/$fold"
  for model in roi_mlp gcn gat; do
    out_dir="$SANITY_ROOT/${fold}_${model}"
    mkdir -p "$out_dir"
    job=$(sbatch --parsable \
      --account="$ACCOUNT" \
      --dependency="afterok:$dataset_dep" \
      --job-name="roi${model:0:1}${fold_id}" \
      --partition=cs \
      --time=08:00:00 \
      --cpus-per-task=8 \
      --mem=64G \
      --output="$LOG_DIR/${model}_${fold_id}_cached_%j.out" \
      --error="$LOG_DIR/${model}_${fold_id}_cached_%j.err" \
      --wrap="$PY scripts/run_roi_graph_baseline_fold.py --graph-dir $graph_dir --fold-name $fold --model $model --output-dir $out_dir --device auto")
    sanity_jobs+=("$job")
  done
done
sanity_dep="$(IFS=:; echo "${sanity_jobs[*]}")"

sanity_summary=$(sbatch --parsable \
  --account="$ACCOUNT" \
  --dependency="afterok:$sanity_dep" \
  --job-name=roisumc \
  --partition=cs \
  --time=00:20:00 \
  --cpus-per-task=1 \
  --mem=4G \
  --output="$LOG_DIR/roisum_cached_%j.out" \
  --error="$LOG_DIR/roisum_cached_%j.err" \
  --wrap="$PY scripts/summarize_roi_graph_sanity.py --root $SANITY_ROOT --output $OUT/roi_graph_sanity_cached_smoke_summary.csv")

brain_jobs=()
for fold_id in 01 04; do
  fold="fold_${fold_id}"
  graph_dir="$GRAPH_ROOT/$fold"
  out_dir="$BRAIN_ROOT/$fold"
  mkdir -p "$out_dir"
  job=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --dependency="afterok:$sanity_summary" \
    --job-name="brainc${fold_id}" \
    --partition=cs \
    --time=12:00:00 \
    --cpus-per-task=8 \
    --mem=96G \
    --output="$LOG_DIR/brain_cached_${fold_id}_%j.out" \
    --error="$LOG_DIR/brain_cached_${fold_id}_%j.err" \
    --wrap="$PY scripts/run_braingnn_nsd_fold.py --graph-dir $graph_dir --fold-name $fold --braingnn-root $ROOT/external/BrainGNN_Pytorch --output-dir $out_dir --device auto")
  brain_jobs+=("$job")
done
brain_dep="$(IFS=:; echo "${brain_jobs[*]}")"

brain_summary=$(sbatch --parsable \
  --account="$ACCOUNT" \
  --dependency="afterok:$brain_dep" \
  --job-name=brainsumc \
  --partition=cs \
  --time=00:20:00 \
  --cpus-per-task=1 \
  --mem=4G \
  --output="$LOG_DIR/brainsum_cached_%j.out" \
  --error="$LOG_DIR/brainsum_cached_%j.err" \
  --wrap="$PY scripts/summarize_braingnn_smoke.py --root $BRAIN_ROOT --sanity-summary $OUT/roi_graph_sanity_cached_smoke_summary.csv --output $OUT/braingnn_nsd_cached_smoke_summary.csv")

cat <<EOF
EXPORT_JOBS=${export_jobs[*]}
DATASET_JOBS=${dataset_jobs[*]}
SANITY_JOBS=${sanity_jobs[*]}
SANITY_SUMMARY=$sanity_summary
BRAIN_JOBS=${brain_jobs[*]}
BRAIN_SUMMARY=$brain_summary
EOF
