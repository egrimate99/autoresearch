#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p results/latest

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/eval_once.sh [cpu|gpu|cuda|auto]

Backend selection:
  cpu   Run local training on CPU.
  gpu   Run local training on CUDA GPU. Alias: cuda.
  auto  Use CUDA if PyTorch can see it, otherwise CPU.

If no argument is supplied, configs/train.yaml decides the device. You can also
set AUTORESEARCH_DEVICE=cpu|cuda|auto.

Remote OpenAI Codex Cloud runs are started from Codex web against the pushed
GitHub branch. See docs/codex_cloud.md and .github/codex/prompts/autoresearch-loop.md.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

case "${1:-}" in
  "")
    ;;
  cpu|local-cpu)
    AUTORESEARCH_DEVICE=cpu
    ;;
  gpu|cuda|local-gpu|local-cuda)
    AUTORESEARCH_DEVICE=cuda
    ;;
  auto|local-auto)
    AUTORESEARCH_DEVICE=auto
    ;;
  remote|codex-cloud)
    echo "Remote Codex Cloud tasks cannot be launched from this local eval script." >&2
    echo "Push this branch, then use the prompt in .github/codex/prompts/autoresearch-loop.md at chatgpt.com/codex." >&2
    exit 64
    ;;
  *)
    usage
    exit 64
    ;;
esac

export AUTORESEARCH_DEVICE

echo "AUTORESEARCH_DEVICE=${AUTORESEARCH_DEVICE:-config}"

TRAIN_DEVICE_ARGS=()
if [[ -n "${AUTORESEARCH_DEVICE:-}" ]]; then
  TRAIN_DEVICE_ARGS=(--device "$AUTORESEARCH_DEVICE")
fi

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
  --out results/latest \
  "${TRAIN_DEVICE_ARGS[@]}"

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
