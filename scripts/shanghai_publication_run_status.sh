#!/bin/bash
set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM

echo "== time =="
date

echo
echo "== queue =="
squeue -u xh2906 -o "%.18i %.28j %.10T %.10M %.9l %.6D %R" || true

echo
echo "== recent publication jobs =="
sacct -j 2398042,2398069,2398070,2398071,2398072,2398094,2398095,2398096 \
  --format=JobID,JobName%24,Partition,State,Elapsed,ExitCode,MaxRSS 2>/dev/null || true

echo
echo "== single-reference BNT strict control =="
BNT_ROOT="preproc_v0/repetition_familiarity/results/single_ref_matched_allseed/bnt_token_flat_gated_flat_clip/lambda_2"
echo -n "metrics.json: "
find "$BNT_ROOT" -path "*/metrics.json" 2>/dev/null | wc -l
echo -n "checkpoint.pt: "
find "$BNT_ROOT" -path "*/checkpoint.pt" 2>/dev/null | wc -l
find "$BNT_ROOT" -path "*/metrics.json" 2>/dev/null | sort | tail -5 || true

echo
echo "== single-reference summaries =="
for path in \
  preproc_v0/repetition_familiarity/results/final_tables/single_ref_matched_allseed_summary.csv \
  preproc_v0/repetition_familiarity/results/final_tables/single_ref_matched_allseed_pairwise_tests.csv \
  preproc_v0/repetition_familiarity/results/final_tables/publication_paired_stats.csv \
  preproc_v0/repetition_familiarity/results/final_tables/aaai_publication_readiness_audit.md \
  preproc_v0/repetition_familiarity/results/final_tables/manuscript_publication_claims_audit.md
do
  if [[ -s "$path" ]]; then
    ls -lh "$path"
  else
    echo "missing: $path"
  fi
done

echo
echo "== LAION-fMRI external validation =="
LAION_ROOT="external_validation/laion_fmri"
pgrep -af "export_laion_fmri_visual_roi_scalar4|run_laion_fmri_download_login" || true
echo -n "downloaded files: "
find "$LAION_ROOT/downloads" -type f 2>/dev/null | wc -l
du -sh "$LAION_ROOT/downloads" 2>/dev/null || true
for path in \
  "$LAION_ROOT/visual_roi_scalar4_laion/laion_download_manifest.json" \
  "$LAION_ROOT/visual_roi_scalar4_laion/laion_visual_roi_scalar4_manifest.json" \
  external_validation/summary/laion_fmri_visual_roi_summary.csv \
  external_validation/summary/laion_fmri_visual_roi_summary.md
do
  if [[ -s "$path" ]]; then
    ls -lh "$path"
  else
    echo "missing: $path"
  fi
done
echo -n "exported LAION tensors: "
find "$LAION_ROOT/visual_roi_scalar4_laion" -name "*_laion_visual_roi_scalar4.pt" 2>/dev/null | wc -l
echo -n "LAION train summaries: "
find "$LAION_ROOT/visual_roi_scalar4_laion/trained_external" -path "*/summary.csv" 2>/dev/null | wc -l

echo
echo "== latest logs =="
ls -lt slurm_logs/rg_single_bnt_2398042_*.out 2>/dev/null | head -5 || true
ls -lt slurm_logs/laion_* 2>/dev/null | head -10 || true
