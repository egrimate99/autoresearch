#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v python >/dev/null 2>&1; then
  PYTHON=${PYTHON:-python}
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON=${PYTHON:-python.exe}
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=${PYTHON:-python3}
else
  echo "Could not find python, python3, or python.exe on PATH." >&2
  exit 127
fi

if ! "$PYTHON" -m uv --version >/dev/null 2>&1; then
  "$PYTHON" -m pip install --user uv
fi

"$PYTHON" -m uv sync
