# minimal mechanism-synthesis autoresearch

This repo benchmarks a learned synthesizer:

```text
A_omega(F) -> G_F
```

`F` is a finite social choice rule/correspondence. `G_F` is a generated finite-message mechanism. The fixed verifier checks exact pure-Nash implementation by enumerating every message profile at every state.

The current benchmark is about compact implementation, not rediscovering a state-report construction:

- positive SCRs are generated from varied small latent mechanisms
- the latent mechanism implements the SCR by construction
- the model sees only the SCR at evaluation time
- generated mechanisms may use variable message counts
- the score penalizes both implementation errors and complexity relative to an oracle reference
- negative controls perturb SCR targets and must not be falsely verified

## Run Once

From PowerShell:

```powershell
cd C:\Mate\Mathematics\implementation_autoresearch
bash scripts/eval_once.sh
```

Choose a runtime explicitly:

```powershell
# Local CPU
bash scripts/eval_once.sh cpu

# Local CUDA GPU
bash scripts/eval_once.sh gpu

# Auto-detect CUDA, otherwise CPU
bash scripts/eval_once.sh auto
```

`configs/train.yaml` currently defaults to CUDA. The command-line selector
passes `--device` to `train.py`, so you do not need to edit YAML to switch
between CPU and GPU.

If `bash` is not available, run the same stages directly:

```powershell
python -m uv run python train.py --config configs/train.yaml --out results/latest
python -m uv run python verify.py --config configs/eval_train_scrs.yaml --checkpoint results/latest --json results/verify_train.json
python -m uv run python verify.py --config configs/eval_holdout_scrs.yaml --checkpoint results/latest --json results/verify_holdout.json
python -m uv run python verify.py --config configs/eval_negative_scrs.yaml --checkpoint results/latest --json results/verify_negative.json
python -m uv run python aggregate.py results/verify_train.json results/verify_holdout.json results/verify_negative.json
```

The last command prints `FINAL_SCORE=...`. Lower is better.

## Logged Evaluations

For autoresearch loops, prefer the logged wrapper:

```powershell
bash scripts/eval_logged.sh auto --hypothesis "baseline or attempted idea"
```

It runs `scripts/eval_once.sh`, saves the complete command output, archives the
train summary and verifier JSON files, records git state/diff, and appends an
index record to `results/eval_logs/index.jsonl`.

Each attempt gets a directory:

```text
results/eval_logs/<run_id>/
  output.log
  train_summary.json
  verify_train.json
  verify_holdout.json
  verify_negative.json
  git_diff_before.patch
  git_status_before.txt
  git_status_after.txt
```

## Codex Cloud

Codex Cloud uses a GitHub checkout, not this local directory. Push the current
branch, then start a task at `https://chatgpt.com/codex` using the prompt in
`.github/codex/prompts/autoresearch-loop.md`.

Details are in `docs/codex_cloud.md`.

## Autoresearch Files

Editable by the research agent:

- `train.py`
- `synthesizer.py`
- `mechanism_decoder.py`
- `losses.py`

Fixed evaluator/generator files:

- `scr_dataset.py`
- `envs.py`
- `equilibrium.py`
- `verify.py`
- `aggregate.py`
- `configs/*.yaml`
- `scripts/eval_once.sh`

The detailed loop and acceptance rule are in `program.md`.
