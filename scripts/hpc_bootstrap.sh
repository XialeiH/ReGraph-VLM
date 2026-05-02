#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

export TMPDIR="${PROJECT_ROOT}/.tmp"
export PIP_CACHE_DIR="${PROJECT_ROOT}/.pip-cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

/usr/bin/python3.12 -m venv --copies .venv
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${VENV_PYTHON}" -m pip install --no-cache-dir -e '.[dev]'
