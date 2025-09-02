#!/usr/bin/env bash
set -eu

# Get the directory of this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# The absolute path to the src directory
SRC_PATH_ABSOLUTE=$(realpath "$SCRIPT_DIR/../packages/recognizer-workshop/src")

# Change into the src directory. This is crucial.
cd "$SRC_PATH_ABSOLUTE"

echo "Running from: $(pwd)"
echo "Serving app via: launch_app.py"

# --- THE FINAL FIX ---
# Use `python -m panel serve` on the simple launcher script.
# This ensures Python's import system is correctly initialized,
# and the server is given a simple, unambiguous file to run.
python -m panel serve launch_app.py \
  --port 8866 \
  --autoreload \
  --log-level debug \
  --allow-websocket-origin=localhost:8866 \
  --allow-websocket-origin=127.0.0.1:8866

