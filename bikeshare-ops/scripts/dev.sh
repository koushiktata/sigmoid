#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=".venv/bin/python"
if [[ ! -f artifacts/transformer.pt ]]; then
  "$PY" -m bikeshare.train
fi
"$PY" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT
"$PY" -m streamlit run ui/app.py --server.port 8501 --server.headless true
