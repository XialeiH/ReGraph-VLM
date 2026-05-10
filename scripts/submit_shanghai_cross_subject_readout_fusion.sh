#!/usr/bin/env bash
set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"

jid_gated=$(sbatch scripts/shanghai_cross_subject_gated_flatten_array.sbatch | awk '{print $4}')
jid_gated_sum=$(sbatch --dependency=afterok:${jid_gated} scripts/shanghai_cross_subject_gated_flatten_summary.sbatch | awk '{print $4}')
jid_fusion=$(sbatch scripts/shanghai_cross_subject_fusion_array.sbatch | awk '{print $4}')
jid_fusion_sum=$(sbatch --dependency=afterok:${jid_fusion} scripts/shanghai_cross_subject_fusion_summary.sbatch | awk '{print $4}')

cat <<EOF
gated_flatten_array=${jid_gated}
gated_flatten_summary=${jid_gated_sum}
fusion_array=${jid_fusion}
fusion_summary=${jid_fusion_sum}
EOF
