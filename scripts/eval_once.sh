#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p results/latest

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

run_timeout() {
  local limit="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$limit" "$@"
  else
    "$@"
  fi
}

run_timeout 30m "$PYTHON" -m uv run python train.py \
  --config configs/train.yaml \
  --out results/latest

run_timeout 10m "$PYTHON" -m uv run python verify.py \
  --config configs/eval_train_scrs.yaml \
  --checkpoint results/latest \
  --json results/verify_train.json

run_timeout 10m "$PYTHON" -m uv run python verify.py \
  --config configs/eval_holdout_scrs.yaml \
  --checkpoint results/latest \
  --json results/verify_holdout.json

run_timeout 10m "$PYTHON" -m uv run python verify.py \
  --config configs/eval_negative_scrs.yaml \
  --checkpoint results/latest \
  --json results/verify_negative.json

"$PYTHON" -m uv run python aggregate.py \
  results/verify_train.json \
  results/verify_holdout.json \
  results/verify_negative.json
