#!/bin/bash
#
# This script launches the Panel application with the debugpy server
# attached, waiting for a connection from the VS Code debugger.

echo "Starting Python app with debug server on port 5678..."

python -m debugpy --listen 5678 --wait-for-client ./scripts/run_server.py --debug