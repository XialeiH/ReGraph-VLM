#!/usr/bin/env bash
set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p slurm_logs

jid_p3=$(sbatch scripts/shanghai_phase3_graph_ablation_array.sbatch | awk '{print $4}')
jid_p3_sum=$(sbatch --dependency=afterany:${jid_p3} scripts/shanghai_phase3_graph_ablation_summary.sbatch | awk '{print $4}')

echo "Submitted Phase 3.1 graph ablation array: ${jid_p3}"
echo "Submitted Phase 3.1 summary: ${jid_p3_sum}"
