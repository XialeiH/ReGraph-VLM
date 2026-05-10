#!/usr/bin/env bash
#SBATCH --job-name=bridgegen_gate
#SBATCH --account=torch_pr_482_general
#SBATCH --partition=cpu_short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

mkdir -p logs
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

"${VENV_PYTHON}" - <<'PY'
import json
from pathlib import Path
import subprocess

pilot_summary = Path("results/pilot/pilot_summary.json")
if not pilot_summary.exists():
    raise SystemExit("pilot_summary.json not found")

payload = json.loads(pilot_summary.read_text())
if not payload["gate"]["pilot_passed"]:
    print("Pilot gate failed; full matrix will not be submitted.")
    raise SystemExit(0)

command = ["sbatch", "--chdir", str(Path.cwd()), "scripts/hpc_full_matrix_array.sbatch"]
print("Submitting full matrix:", " ".join(command))
completed = subprocess.run(command, check=True, text=True, capture_output=True)
print(completed.stdout.strip())
PY
