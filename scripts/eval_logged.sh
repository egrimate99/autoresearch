#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/eval_logged.sh [cpu|gpu|cuda|auto] --hypothesis "idea text"

Runs scripts/eval_once.sh, saves the complete stdout/stderr stream, archives
verification JSON outputs, captures git state, and appends an index record to:

  results/eval_logs/index.jsonl

Examples:
  bash scripts/eval_logged.sh auto --hypothesis "remote baseline"
  bash scripts/eval_logged.sh gpu --hypothesis "increase hidden_dim to 288"
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

backend="${1:-auto}"
case "$backend" in
  cpu|gpu|cuda|auto)
    shift || true
    ;;
  *)
    backend=auto
    ;;
esac

hypothesis=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hypothesis)
      shift
      hypothesis="${1:-}"
      ;;
    --hypothesis=*)
      hypothesis="${1#--hypothesis=}"
      ;;
    *)
      if [[ -z "$hypothesis" ]]; then
        hypothesis="$*"
        break
      fi
      ;;
  esac
  shift || true
done

if [[ -z "$hypothesis" ]]; then
  hypothesis="unspecified"
fi

timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_dir="results/eval_logs/$run_id"
mkdir -p "$run_dir"

printf '%s\n' "$hypothesis" > "$run_dir/hypothesis.txt"
git rev-parse --abbrev-ref HEAD > "$run_dir/git_branch.txt" 2>&1 || true
git rev-parse HEAD > "$run_dir/git_commit.txt" 2>&1 || true
git status --short > "$run_dir/git_status_before.txt" 2>&1 || true
git diff --stat > "$run_dir/git_diff_stat_before.txt" 2>&1 || true
git diff > "$run_dir/git_diff_before.patch" 2>&1 || true

set +e
bash scripts/eval_once.sh "$backend" 2>&1 | tee "$run_dir/output.log"
exit_code=${PIPESTATUS[0]}
set -e

for name in verify_train verify_holdout verify_negative; do
  if [[ -f "results/${name}.json" ]]; then
    cp "results/${name}.json" "$run_dir/${name}.json"
  fi
done

if [[ -f results/latest/train_summary.json ]]; then
  cp results/latest/train_summary.json "$run_dir/train_summary.json"
fi

git status --short > "$run_dir/git_status_after.txt" 2>&1 || true

if command -v python >/dev/null 2>&1; then
  META_PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  META_PYTHON=python3
elif command -v python.exe >/dev/null 2>&1; then
  META_PYTHON=python.exe
else
  echo "Could not find python, python3, or python.exe for log indexing." >&2
  exit 127
fi

RUN_ID="$run_id" \
RUN_TIMESTAMP="$timestamp" \
RUN_BACKEND="$backend" \
RUN_HYPOTHESIS="$hypothesis" \
RUN_EXIT_CODE="$exit_code" \
RUN_DIR="$run_dir" \
"$META_PYTHON" - <<'PY'
import json
import os
import re
from pathlib import Path


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def parse_value(raw: str):
    raw = raw.strip()
    try:
        if re.fullmatch(r"[-+]?\d+", raw):
            return int(raw)
        if re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", raw):
            return float(raw)
    except Exception:
        pass
    return raw


run_dir = Path(os.environ["RUN_DIR"])
output = read_text(run_dir / "output.log")
metric_pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
metrics_sequence = []
metrics = {}
for line in output.splitlines():
    match = metric_pattern.match(line.strip())
    if not match:
        continue
    key, value = match.groups()
    parsed = parse_value(value)
    metrics_sequence.append({"key": key, "value": parsed})
    metrics[key] = parsed

summaries = {}
for split in ("train", "holdout", "negative"):
    path = run_dir / f"verify_{split}.json"
    if not path.exists():
        continue
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        summaries[split] = {"json_error": repr(exc)}
    else:
        summaries[split] = data.get("summary", {})

train_summary_path = run_dir / "train_summary.json"
train_summary = {}
if train_summary_path.exists():
    try:
        train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        train_summary = {"json_error": repr(exc)}

record = {
    "timestamp": os.environ["RUN_TIMESTAMP"],
    "run_id": os.environ["RUN_ID"],
    "backend": os.environ["RUN_BACKEND"],
    "hypothesis": os.environ["RUN_HYPOTHESIS"],
    "exit_code": int(os.environ["RUN_EXIT_CODE"]),
    "git_branch": read_text(run_dir / "git_branch.txt").strip(),
    "git_commit": read_text(run_dir / "git_commit.txt").strip(),
    "git_status_before": read_text(run_dir / "git_status_before.txt").splitlines(),
    "git_status_after": read_text(run_dir / "git_status_after.txt").splitlines(),
    "git_diff_stat_before": read_text(run_dir / "git_diff_stat_before.txt"),
    "paths": {
        "run_dir": str(run_dir),
        "output_log": str(run_dir / "output.log"),
        "diff_patch": str(run_dir / "git_diff_before.patch"),
        "verify_train_json": str(run_dir / "verify_train.json"),
        "verify_holdout_json": str(run_dir / "verify_holdout.json"),
        "verify_negative_json": str(run_dir / "verify_negative.json"),
        "train_summary_json": str(run_dir / "train_summary.json"),
    },
    "metrics": metrics,
    "metrics_sequence": metrics_sequence,
    "verify_summaries": summaries,
    "train_summary": train_summary,
}

index_path = Path("results/eval_logs/index.jsonl")
index_path.parent.mkdir(parents=True, exist_ok=True)
with index_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=True) + "\n")

print(f"EVAL_LOG_DIR={run_dir}")
print(f"EVAL_LOG_INDEX={index_path}")
PY

exit "$exit_code"
