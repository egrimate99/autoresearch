#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON=python
  elif command -v python.exe >/dev/null 2>&1; then
    PYTHON=python.exe
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    echo "Could not find python, python3, or python.exe on PATH." >&2
    exit 127
  fi
fi

echo "== system =="
uname -a || true
command -v lscpu >/dev/null 2>&1 && lscpu | sed -n '1,20p' || true
command -v free >/dev/null 2>&1 && free -h || true

echo "== python =="
echo "PYTHON=$PYTHON"
if "$PYTHON" -m uv --version >/dev/null 2>&1; then
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
else
  echo "uv_available=false"
  "$PYTHON" - <<'PY'
import platform
import sys

print("executable", sys.executable)
print("platform", platform.platform())
try:
    import torch
except Exception as exc:
    print("torch_import_error", repr(exc))
else:
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device", torch.cuda.get_device_name(0))
PY
fi

echo "== nvidia-smi =="
nvidia-smi || true
