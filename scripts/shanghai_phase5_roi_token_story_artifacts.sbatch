#!/bin/bash
#SBATCH --job-name=p5_roi_story
#SBATCH --account=zx555
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/p5_roi_story_%j.out
#SBATCH --error=/gpfsnyu/scratch/xh2906/ReGraph-VLM/logs/p5_roi_story_%j.err

set -euo pipefail

ROOT=/gpfsnyu/scratch/xh2906/ReGraph-VLM
cd "$ROOT"
mkdir -p logs
source scripts/shanghai_env.sh
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python scripts/make_roi_token_story_artifacts.py --root "$ROOT"
