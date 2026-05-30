#!/bin/bash
#SBATCH --job-name=p32_edge_sum
#SBATCH --account=zx555
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=6G
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/p32_edge_sum_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/p32_edge_sum_%j.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p logs
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python scripts/collect_regraph_vlm_metrics.py \
  --root "$ROOT" \
  --sources preproc_v0/repetition_familiarity/results/phase32_learned_edge_bias \
  --out-csv preproc_v0/repetition_familiarity/results/phase32_learned_edge_bias/regraph_vlm_summary.csv

python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("/gpfsnyu/scratch/xh2906/ReGraph-VLM")
path = root / "preproc_v0/repetition_familiarity/results/phase32_learned_edge_bias/regraph_vlm_summary.csv"
df = pd.read_csv(path)
metrics = ["AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR"]
summary = df.groupby(["graph_encoder", "readout", "adjacency_mode"]).agg(
    n=("AUROC", "count"),
    **{f"{m}_mean": (m, "mean") for m in metrics},
    **{f"{m}_std": (m, "std") for m in metrics},
).reset_index()
out = path.parent / "phase32_learned_edge_bias_group_summary.csv"
summary.to_csv(out, index=False)
try:
    md = summary.to_markdown(index=False, floatfmt=".4f")
except Exception:
    md = summary.to_csv(index=False)
(path.parent / "phase32_learned_edge_bias_group_summary.md").write_text(md, encoding="utf-8")
print(summary.to_string(index=False))
PY
