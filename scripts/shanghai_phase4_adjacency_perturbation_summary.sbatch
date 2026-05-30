#!/bin/bash
#SBATCH --job-name=p4_adj_sum
#SBATCH --account=zx555
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/p4_adj_sum_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/p4_adj_sum_%j.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p logs
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python scripts/summarize_adjacency_perturbation_controls.py --root "$ROOT"
