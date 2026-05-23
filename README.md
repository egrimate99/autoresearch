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

If `bash` is not available, run the same stages directly:

```powershell
python -m uv run python train.py --config configs/train.yaml --out results/latest
python -m uv run python verify.py --config configs/eval_train_scrs.yaml --checkpoint results/latest --json results/verify_train.json
python -m uv run python verify.py --config configs/eval_holdout_scrs.yaml --checkpoint results/latest --json results/verify_holdout.json
python -m uv run python verify.py --config configs/eval_negative_scrs.yaml --checkpoint results/latest --json results/verify_negative.json
python -m uv run python aggregate.py results/verify_train.json results/verify_holdout.json results/verify_negative.json
```

The last command prints `FINAL_SCORE=...`. Lower is better.

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
