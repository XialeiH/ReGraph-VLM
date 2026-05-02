#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

mkdir -p logs

gen_job=$(sbatch --parsable --chdir="${PROJECT_ROOT}" scripts/hpc_generate_prelim.sbatch)
pilot_job=$(sbatch --parsable --chdir="${PROJECT_ROOT}" --dependency=afterok:${gen_job} scripts/hpc_pilot.sbatch)
gate_job=$(sbatch --parsable --chdir="${PROJECT_ROOT}" --dependency=afterok:${pilot_job} scripts/hpc_pilot_gate.sbatch)

echo "generate_prelim_job=${gen_job}"
echo "pilot_job=${pilot_job}"
echo "gate_job=${gate_job}"
