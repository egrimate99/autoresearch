# learned mechanism-synthesis autoresearch

This repo is set up for autoresearch over a learned synthesizer:

```text
A_omega(F) -> G_F
```

`F` is a finite social choice rule/correspondence. `G_F` is a generated finite-message mechanism. The fixed verifier checks exact pure-Nash implementation by enumerating all message profiles for every state.

The current setup is intentionally a hard learning benchmark:

- `scr_dataset.py` generates larger, arbitrary state-to-outcome SCRs.
- Positive SCRs are implementable by a report/challenge mechanism, but chosen outcomes are often not top alternatives overall.
- Private distractor alternatives make the utility tensor matter; copying top choices is not enough.
- `synthesizer.py` emits logits for a full mechanism outcome table.
- `mechanism_decoder.py` only reshapes learned logits and takes argmax.
- Heldout SCRs are fresh samples.
- Negative controls perturb targets away from the positive implementable structure.

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
