#!/bin/bash
echo "============================================================"
echo "  SwarmLLM Setup and Run"
echo "============================================================"
echo

cd "$(dirname "$0")"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv || { echo "ERROR: Failed to create virtual environment."; exit 1; }
    echo
fi

# Activate venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt --quiet
echo

# Run interactive setup + launch
python scripts/setup_run.py
