# Autoresearch Agent Instructions

This repository benchmarks learned finite mechanism synthesis.

For local one-shot evaluation, use one of:

```bash
bash scripts/eval_once.sh cpu
bash scripts/eval_once.sh gpu
bash scripts/eval_once.sh auto
```

For autoresearch attempts, use the logged wrapper instead:

```bash
bash scripts/eval_logged.sh auto --hypothesis "short attempt description"
```

Keep and commit `results/eval_logs/` artifacts after every attempt, even when
the code change is rejected.

For OpenAI Codex Cloud, use:

```bash
git remote -v || true
git status --short --branch || true
bash scripts/probe_runtime.sh
bash scripts/eval_logged.sh auto --hypothesis "remote baseline"
```

If no git remote is configured, continue from the checked-out files. Commit
logs/results inside the task, then use the Codex UI to create a PR or export
the patch before ending the session.

During an autoresearch loop, read `program.md` and edit only:

- `train.py`
- `synthesizer.py`
- `mechanism_decoder.py`
- `losses.py`

Do not edit fixed benchmark files, configs, scripts, or prompts during the
research loop unless the user explicitly asks to change the benchmark or
runtime setup.
