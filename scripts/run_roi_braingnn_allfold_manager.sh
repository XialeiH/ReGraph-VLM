#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scratch/xh2906/final_project_nsd/v0_shared_unit}"
PY="${PY:-/scratch/xh2906/conda/envs/py310/bin/python}"
ACCOUNT="${ACCOUNT:-torch_pr_482_general}"
OUT="$ROOT/preproc_v0/roi_graph_atlas_v1"
FOLDROOT="$ROOT/preproc_v0/all8_ge2_766"
NODESET="$OUT/roi_node_set_v1.json"
BRAIN_ROOT="$OUT/braingnn_nsd_allfold"
GRAPH_ROOT="$OUT/allfold"
REPORT="$OUT/atlas_roi_graph_scheduled_report.md"
LOG_DIR="$ROOT/logs"

mkdir -p "$BRAIN_ROOT" "$GRAPH_ROOT" "$LOG_DIR"

{
  echo "# Atlas ROI Graph Scheduled Report"
  echo
  echo "- Started: $(date -Is)"
  echo "- Host: $(hostname)"
  echo "- User: $(whoami)"
  echo "- Root: $ROOT"
  echo "- Output root: $OUT"
  echo
  echo "## Existing Smoke Job Status"
  sacct -j 7217709,7217710,7219959,7219960,7219961,7219962,7219963,7219964,7219965,7223853,7223854,7223855 \
    --format=JobID,JobName%18,State,Elapsed,MaxRSS -n -P 2>/dev/null || true
  echo
  echo "## Existing Result Files"
  for path in \
    "$OUT/roi_inventory.csv" \
    "$OUT/roi_node_set_v1.json" \
    "$OUT/fold_01/graph_dataset_qc.json" \
    "$OUT/fold_04/graph_dataset_qc.json" \
    "$OUT/roi_graph_sanity_summary.csv" \
    "$OUT/braingnn_nsd_smoke_summary.csv"; do
    if [[ -e "$path" ]]; then
      echo "- present: $path"
    else
      echo "- missing: $path"
    fi
  done
  echo
  echo "## Submitted All-Fold Jobs"
} > "$REPORT"

dataset_jobs=()
brain_jobs=()
for fold_id in 01 02 03 04 05 06 07 08; do
  fold="fold_${fold_id}"
  graph_dir="$GRAPH_ROOT/$fold"
  brain_dir="$BRAIN_ROOT/$fold"
  mkdir -p "$graph_dir" "$brain_dir"

  dataset_job=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --job-name="roid${fold_id}" \
    --partition=cs \
    --time=12:00:00 \
    --cpus-per-task=4 \
    --mem=96G \
    --output="$LOG_DIR/roid_all_${fold_id}_%j.out" \
    --error="$LOG_DIR/roid_all_${fold_id}_%j.err" \
    --wrap="$PY scripts/build_roi_graph_dataset.py --root $ROOT --fold-root $FOLDROOT --fold-name $fold --node-set $NODESET --output-dir $graph_dir")

  brain_job=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --dependency="afterok:$dataset_job" \
    --job-name="brain${fold_id}" \
    --partition=cs \
    --time=06:00:00 \
    --cpus-per-task=4 \
    --mem=48G \
    --output="$LOG_DIR/brain_all_${fold_id}_%j.out" \
    --error="$LOG_DIR/brain_all_${fold_id}_%j.err" \
    --wrap="$PY scripts/run_braingnn_nsd_fold.py --graph-dir $graph_dir --fold-name $fold --braingnn-root $ROOT/external/BrainGNN_Pytorch --output-dir $brain_dir --device auto")

  dataset_jobs+=("$dataset_job")
  brain_jobs+=("$brain_job")
  echo "- $fold dataset: $dataset_job brain: $brain_job" >> "$REPORT"
done

brain_dep="$(IFS=:; echo "${brain_jobs[*]}")"
summary_job=$(sbatch --parsable \
  --account="$ACCOUNT" \
  --dependency="afterok:$brain_dep" \
  --job-name=brainsum8 \
  --partition=cs \
  --time=00:20:00 \
  --cpus-per-task=1 \
  --mem=4G \
  --output="$LOG_DIR/brainsum8_%j.out" \
  --error="$LOG_DIR/brainsum8_%j.err" \
  --wrap="$PY scripts/summarize_braingnn_smoke.py --root $BRAIN_ROOT --sanity-summary $OUT/roi_graph_sanity_summary.csv --output $OUT/braingnn_nsd_allfold_summary.csv")

{
  echo "- allfold summary: $summary_job"
  echo
  echo "## Queue Snapshot"
  squeue -j "$(IFS=,; echo "${dataset_jobs[*]},${brain_jobs[*]},$summary_job")" \
    -o "%.18i %.9P %.20j %.8T %.10M %.9l %.6D %R" || true
} >> "$REPORT"

echo "REPORT=$REPORT"
echo "SUMMARY_JOB=$summary_job"
