#!/bin/bash
cd "$(dirname "$0")"
set -e

PY="$(command -v python3.10 || true)"
if [ -z "$PY" ]; then
  echo "This app needs Python 3.10. DrumScript does not run on Python 3.13."
  echo "Install it with: brew install python@3.10"
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Setting up the app (first launch can take a few minutes)..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install -U pip
  .venv/bin/python -m pip install -r requirements.txt
fi

export PYTHONPATH="$(pwd)/vendor/DrumScript${PYTHONPATH:+:$PYTHONPATH}"

(
  for _ in $(seq 1 80); do
    if nc -z 127.0.0.1 8765 2>/dev/null; then
      open "http://127.0.0.1:8765"
      exit 0
    fi
    sleep 0.25
  done
) &

exec .venv/bin/python app/server.py
