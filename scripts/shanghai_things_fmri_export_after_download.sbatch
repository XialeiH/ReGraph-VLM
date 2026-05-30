#!/bin/bash
#SBATCH --job-name=things_fmri_export
#SBATCH --partition=debug,acc,gpu,chemcourses,aquila,aml,ai
#SBATCH --output=slurm_logs/things_fmri_export_%j.out
#SBATCH --error=slurm_logs/things_fmri_export_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G

set -euo pipefail

cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

ROOT="external_validation/things_fmri/figshare_table"
ZIP="$ROOT/betas_csv.zip"
UNPACKED="$ROOT/betas_csv_unpacked"
OUT_DIR="external_validation/things_fmri/visual_roi_scalar4_smoke"
EXPECTED_BYTES=42955087212

if [[ ! -f "$ZIP" ]]; then
  echo "Missing $ZIP"
  exit 1
fi

actual_bytes=$(stat -c%s "$ZIP")
if [[ "$actual_bytes" -lt "$EXPECTED_BYTES" ]]; then
  echo "Archive is incomplete: $actual_bytes < $EXPECTED_BYTES"
  exit 1
fi

mkdir -p "$UNPACKED" "$OUT_DIR"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q -n "$ZIP" -d "$UNPACKED"

first_h5=$(find "$UNPACKED" -name 'sub-01_ResponseData.h5' -print -quit)
if [[ -z "$first_h5" ]]; then
  echo "Could not find sub-01_ResponseData.h5 after unzip"
  exit 1
fi
BETAS_DIR=$(dirname "$first_h5")
echo "Using BETAS_DIR=$BETAS_DIR"

python scripts/export_things_fmri_visual_roi_scalar4.py \
  --betas-csv-dir "$BETAS_DIR" \
  --out-dir "$OUT_DIR" \
  --subjects 01 02 03 \
  --max-shared-images 1000
