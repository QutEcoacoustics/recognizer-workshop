#!/usr/bin/env bash

set -euo pipefail

# Reproducible Python environment setup for local dev and CI test runs.
#
# What this script does:
# 1) Upgrades packaging tooling.
# 2) Installs TensorFlow explicitly.
# 3) Installs perch-hoplite from PyPI.
# 4) Installs test/runtime deps and resolves known pyarrow pin.
# 5) Installs repo packages in editable mode.
# 6) Verifies imports and prints installed versions.
#
# Usage:
#   bash scripts/setup_test_python_env.sh
#   PERCH_HOPLITE_VERSION=0.1.3 PYARROW_VERSION=22.0.0 bash scripts/setup_test_python_env.sh
#
# Notes:
# - Run this inside your target Python environment (conda/venv already activated).
# - Uses the current interpreter via `python -m pip` to avoid PATH mismatch issues.

PERCH_HOPLITE_VERSION="${PERCH_HOPLITE_VERSION:-0.1.3}"
PYARROW_VERSION="${PYARROW_VERSION:-22.0.0}"
TF_SPEC="${TF_SPEC:-tensorflow>=2.20,<3}"
NUMPY_SPEC="${NUMPY_SPEC:-numpy>=2.0,<2.1}"
PANDAS_SPEC="${PANDAS_SPEC:-pandas>=2.1.1,<3}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() {
  printf "\n[%s] %s\n" "setup" "$1"
}

log "Using Python: $(python -c 'import sys; print(sys.executable)')"
log "Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

log "Installing TensorFlow (${TF_SPEC})"
python -m pip install --upgrade "${TF_SPEC}"

log "Installing perch-hoplite==${PERCH_HOPLITE_VERSION}"
python -m pip install --force-reinstall "perch-hoplite==${PERCH_HOPLITE_VERSION}"

log "Installing core test/runtime dependencies"
python -m pip install --upgrade "${NUMPY_SPEC}" scipy pytest "${PANDAS_SPEC}" "pyarrow==${PYARROW_VERSION}" pyarrow

log "Re-pinning pyarrow==${PYARROW_VERSION}"
python -m pip install --force-reinstall "pyarrow==${PYARROW_VERSION}"

log "Installing workspace packages in editable mode"
python -m pip install --no-cache-dir -e "${ROOT_DIR}/packages/baw-helpers"
python -m pip install --no-cache-dir -e "${ROOT_DIR}/packages/recognizer-workshop"

log "Re-applying compatibility pins after editable installs"
python -m pip install --force-reinstall "${NUMPY_SPEC}" "${PANDAS_SPEC}" "pyarrow==${PYARROW_VERSION}"

log "Running pip check"
python -m pip check

log "Verifying imports and reporting key versions"
python - <<'PY'
from importlib.metadata import version
import tensorflow as tf
import pyarrow
import perch_hoplite
import baw_helpers
import recognizer_workshop

print("Python environment verification passed")
print("tensorflow", tf.__version__)
print("pyarrow", pyarrow.__version__)
print("perch-hoplite", version("perch-hoplite"))
print("baw-helpers", version("baw-helpers"))
print("recognizer-workshop", version("recognizer-workshop"))
PY

log "Environment setup complete"
