#!/usr/bin/env bash
set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p slurm_logs

jid_p1=$(sbatch scripts/shanghai_phase1_robust_statistics.sbatch | awk '{print $4}')
jid_p2=$(sbatch scripts/shanghai_phase2_sota_graph_baselines_array.sbatch | awk '{print $4}')
jid_p2_sum=$(sbatch --dependency=afterany:${jid_p2} scripts/shanghai_phase2_sota_graph_baselines_summary.sbatch | awk '{print $4}')

echo "Submitted Phase 1 robust statistics: ${jid_p1}"
echo "Submitted Phase 2 SOTA/graph baseline array: ${jid_p2}"
echo "Submitted Phase 2 summary: ${jid_p2_sum}"
