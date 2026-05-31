#!/bin/bash
set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p slurm_logs

python -m py_compile \
  scripts/export_laion_fmri_visual_roi_scalar4.py \
  scripts/summarize_laion_fmri_external_results.py \
  scripts/audit_aaai_publication_readiness.py \
  scripts/audit_manuscript_publication_claims.py

bash -n \
  scripts/run_laion_fmri_download_login.sh \
  scripts/shanghai_laion_fmri_visual_roi_export.sbatch \
  scripts/shanghai_laion_fmri_external_array.sbatch \
  scripts/shanghai_laion_fmri_external_summary.sbatch \
  scripts/submit_shanghai_laion_fmri_external.sh \
  scripts/shanghai_publication_run_status.sh

echo "== current status before actions =="
bash scripts/shanghai_publication_run_status.sh

download_manifest="external_validation/laion_fmri/visual_roi_scalar4_laion/laion_download_manifest.json"
export_manifest="external_validation/laion_fmri/visual_roi_scalar4_laion/laion_visual_roi_scalar4_manifest.json"

if [[ ! -s "$download_manifest" ]]; then
  if pgrep -f "export_laion_fmri_visual_roi_scalar4.py.*--download-only" >/dev/null; then
    echo "LAION download-only process already running."
  else
    stamp=$(date +%Y%m%d_%H%M%S)
    echo "Starting LAION download-only process on login node."
    nohup bash scripts/run_laion_fmri_download_login.sh \
      > "slurm_logs/laion_download_login_${stamp}.out" \
      2> "slurm_logs/laion_download_login_${stamp}.err" \
      < /dev/null &
    echo "LAION download PID: $!"
  fi
else
  echo "LAION download manifest exists: $download_manifest"
fi

if [[ -s "$download_manifest" && ! -s "$export_manifest" ]]; then
  if squeue -u xh2906 -h -n laion_roi_export,laion_ext,laion_ext_sum | grep -q .; then
    echo "LAION Slurm export/training chain already queued or running."
  else
    echo "Submitting LAION local-only export/training/summary chain."
    bash scripts/submit_shanghai_laion_fmri_external.sh
  fi
fi

echo "== current status after actions =="
bash scripts/shanghai_publication_run_status.sh
