#!/usr/bin/env bash
set -euo pipefail
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pytest -q
echo "Setup selesai. Edit config.yaml lalu jalankan: python run_project.py validate"
