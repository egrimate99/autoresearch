Read `program.md` first.

We are improving a learned mechanism synthesizer:

```text
A_omega(F) -> G_F
```

This is a minimal finite pure-Nash implementation benchmark. We care about
aggregate heldout synthesis correctness and message-space compactness, not a
single hand-solved SCR.

Before starting the loop, run:

```bash
bash scripts/probe_runtime.sh
bash scripts/eval_once.sh auto
```

Treat that first evaluation as the baseline for this remote run.

Then repeat until I interrupt you:

1. Propose one concrete hypothesis about improving `train.py`,
   `synthesizer.py`, `mechanism_decoder.py`, or `losses.py`.
2. Edit only those four files during the research loop.
3. Run:

   ```bash
   bash scripts/eval_once.sh auto
   ```

4. Accept only if `FINAL_SCORE` decreases and
   `negative_false_success_rate` remains `0`.
5. Prefer accepted changes that reduce `holdout_bad_equilibrium_error` and
   `holdout_missing_good_equilibrium_error`.
6. If a change fails, revert it and try a different class of idea.
7. Append a detailed JSON record to `results/runs.jsonl`.
8. Also log detailed ideas, failure modes, and observations in
   `results/idea_log.md`.
9. Commit accepted code changes.

Do not edit:

- `scr_dataset.py`
- `envs.py`
- `equilibrium.py`
- `verify.py`
- `aggregate.py`
- `configs/*.yaml`
- `scripts/*.sh`
- `.github/codex/prompts/*.md`

Do not stop because several ideas fail.
