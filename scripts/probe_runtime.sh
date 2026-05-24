#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON=python
  elif command -v python.exe >/dev/null 2>&1; then
    PYTHON=python.exe
  else
    echo "Could not find python or python.exe on PATH." >&2
    exit 127
  fi
fi

echo "== system =="
uname -a || true
command -v lscpu >/dev/null 2>&1 && lscpu | sed -n '1,20p' || true
command -v free >/dev/null 2>&1 && free -h || true

echo "== python =="
echo "PYTHON=$PYTHON"
"$PYTHON" -m uv run python - <<'PY'
import platform
import sys

import torch

print("executable", sys.executable)
print("platform", platform.platform())
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

echo "== nvidia-smi =="
nvidia-smi || true
