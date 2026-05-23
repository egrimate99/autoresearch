# Objective

We are improving a mechanism-synthesis model, not a single mechanism.

The learned object is a synthesizer:

```text
A_omega(F) -> G_F
```

Input:
- a finite social choice rule/correspondence F
- finite agents
- finite states
- finite alternatives
- finite preferences/utilities

Output:
- a finite-message mechanism/game form G_F

For every implementable SCR F, the generated mechanism G_F should pure-Nash-implement F:

```text
NE(G_F, theta) outcomes = F(theta)
```

for every state theta.

This requires:
1. at least one good pure Nash equilibrium for each theta
2. no bad pure Nash equilibria for any theta

A generated mechanism with one good equilibrium and one bad equilibrium is a failure.

# Editable files

You may edit:
- train.py
- synthesizer.py
- mechanism_decoder.py
- losses.py

You may not edit:
- scr_dataset.py
- envs.py
- equilibrium.py
- verify.py
- aggregate.py
- configs/*.yaml
- scripts/eval_once.sh

If a fixed file seems wrong, log the issue but do not modify it.

# Evaluation loop

For each evaluation set:

1. Load many SCR instances F.
2. Run the synthesizer A_omega(F).
3. Decode its output into a candidate finite-message mechanism G_F.
4. For every state theta of F:
   - enumerate all message profiles
   - compute pure Nash equilibria
   - compare equilibrium outcomes to F(theta)
5. Score the generated mechanism.
6. Aggregate scores over SCR instances.

Success means better synthesis accuracy over many SCRs, especially held-out SCRs.

# Commands

After every attempted change, run:

```bash
./scripts/eval_once.sh
```

This script trains the synthesizer, evaluates it on train SCRs, held-out SCRs, and negative-control SCRs, then prints a scalar FINAL_SCORE.

# Acceptance rule

Accept a change only if:

1. FINAL_SCORE decreases
2. holdout_bad_equilibrium_error decreases or stays zero
3. holdout_missing_good_equilibrium_error decreases or stays zero
4. negative-control SCRs are not falsely verified as implementable
5. no fixed files were modified

Reject and revert otherwise.

# Optimization priority

Optimize in this order:

1. eliminate bad equilibria in generated mechanisms
2. ensure good equilibria exist
3. improve held-out SCR generalization
4. reduce exploitability of intended equilibria
5. reduce mechanism size/complexity

# Allowed ideas

Try:
- better SCR encodings
- better mechanism decoders
- tabular game-form output
- Maskin-style message scaffolds
- learned off-equilibrium punishments
- hard-coded canonical scaffold plus learned residual
- equivariant architectures over agents/states/alternatives
- curriculum over SCR size
- stronger training loss for bad-equilibrium exclusion
- adversarial search over near-equilibria during training

Do not start with:
- continuous messages
- mixed Nash equilibrium
- large unrestricted domains
- transformer-only black box with no structure
- changing the verifier
- changing the SCR generator
- changing negative controls

# Logging

Append one JSON object to results/runs.jsonl after every experiment:

```json
{
  "timestamp": "...",
  "hypothesis": "...",
  "files_changed": ["..."],
  "train_score": 0.0,
  "holdout_score": 0.0,
  "negative_score": 0.0,
  "holdout_bad_equilibrium_error": 0.0,
  "holdout_missing_good_equilibrium_error": 0.0,
  "negative_false_success_rate": 0.0,
  "accepted": true,
  "reason": "..."
}
```

Never claim success from one SCR. Success is aggregate synthesis performance over many SCRs.
