#!/usr/bin/env bash
set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p slurm_logs

jid_seed=$(sbatch scripts/shanghai_phase3c_noadj_gated_missing_seeds_array.sbatch | awk '{print $4}')
jid_hold=$(sbatch scripts/shanghai_phase3c_noadj_gated_heldout_array.sbatch | awk '{print $4}')
jid_hard=$(sbatch scripts/shanghai_phase3c_noadj_gated_hardneg_seed11_array.sbatch | awk '{print $4}')
jid_sum=$(sbatch --dependency=afterany:${jid_seed}:${jid_hold}:${jid_hard} scripts/shanghai_phase3c_noadj_gated_summary.sbatch | awk '{print $4}')

echo "Submitted Phase 3c no-adj gated missing seeds: ${jid_seed}"
echo "Submitted Phase 3c no-adj gated held-out: ${jid_hold}"
echo "Submitted Phase 3c no-adj gated hard-negative: ${jid_hard}"
echo "Submitted Phase 3c no-adj gated summary: ${jid_sum}"
