#!/usr/bin/env bash
set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p slurm_logs

jid_audit=$(sbatch --wrap="cd $ROOT && source scripts/shanghai_env.sh && export PYTHONPATH=$ROOT:\${PYTHONPATH:-} && python scripts/audit_phase2_phase3_configs.py --root $ROOT" \
  --job-name=rg_p3b_audit \
  --partition=debug,acc,gpu,chemcourses,aquila,aml,ai \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:30:00 \
  --output="$ROOT/slurm_logs/%x_%j.out" \
  --error="$ROOT/slurm_logs/%x_%j.err" | awk '{print $4}')

jid_p3b=$(sbatch scripts/shanghai_phase3b_clean_graph_ablation_array.sbatch | awk '{print $4}')
jid_p3b_sum=$(sbatch --dependency=afterany:${jid_p3b} scripts/shanghai_phase3b_clean_graph_ablation_summary.sbatch | awk '{print $4}')

echo "Submitted Phase 3b config audit: ${jid_audit}"
echo "Submitted Phase 3b clean graph ablation array: ${jid_p3b}"
echo "Submitted Phase 3b summary: ${jid_p3b_sum}"
