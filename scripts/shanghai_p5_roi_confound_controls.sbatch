#!/bin/bash
#SBATCH --job-name=p5_roi_conf
#SBATCH --account=zx555
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/p5_roi_conf_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/p5_roi_conf_%j.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p logs
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python scripts/analyze_roi_confound_interpretability.py --root "$ROOT"
