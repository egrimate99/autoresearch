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

if [[ -z "${UV:-}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    UV=uv
  elif "$PYTHON" -m uv --version >/dev/null 2>&1; then
    UV="$PYTHON -m uv"
  else
    "$PYTHON" -m pip install --user uv
    if command -v uv >/dev/null 2>&1; then
      UV=uv
    else
      UV="$PYTHON -m uv"
    fi
  fi
fi

echo "PYTHON=$PYTHON"
echo "UV=$UV"

# shellcheck disable=SC2086
$UV sync
