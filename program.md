# Objective

We are improving a learned mechanism synthesizer, not a single mechanism and not a hand-coded searcher.

The learned object is:

```text
A_omega(F) -> G_F
```

Input:
- a finite social choice rule/correspondence F
- finite agents
- finite states
- finite alternatives
- finite ordinal utilities

Output:
- a finite-message mechanism/game form G_F

The current benchmark is deliberately harder than the initial template toy setup:

- positive SCRs contain arbitrary state-to-outcome maps on a larger domain
- chosen outcomes need not be each agent's top alternative overall
- private distractor alternatives often outrank the chosen target
- implementation requires a report/challenge style off-equilibrium structure
- the model emits a full finite outcome table, not a template id
- heldout SCRs are freshly generated and cannot be memorized
- negative controls perturb targets away from unanimous-top implementability

Success means the checkpointed neural synthesizer learns to generate mechanisms whose exact pure Nash equilibria implement the input SCRs.

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

If a fixed file seems wrong, log the issue but do not modify it during an autoresearch run.

# What Counts As Learning

Allowed:
- neural architectures that map SCR tensors to mechanism outcome tables
- differentiable losses over generated tables
- curriculum inside train.py using the fixed dataset API
- regularization and decoding improvements that do not inspect verifier output per instance
- learned residuals over the fixed report/challenge scaffold

Disallowed:
- hard-coded direct reconstruction of the canonical teacher table
- if/else decoding from target_mapping metadata
- per-SCR brute force mechanism search
- changing the verifier, generator, configs, or aggregate score during a run
- using negative-control labels at inference time

# Evaluation loop

For each evaluation set:

1. Load many SCR instances F.
2. Run the checkpointed synthesizer A_omega(F).
3. Decode learned logits into a candidate finite-message mechanism G_F.
4. For every state theta of F:
   - enumerate all message profiles
   - compute pure Nash equilibria
   - compare equilibrium outcomes to F(theta)
5. Aggregate exact implementation errors over SCR instances.

Success is aggregate heldout synthesis performance, not one SCR.

# Command

After every attempted change, run:

```bash
./scripts/eval_once.sh
```

This trains the synthesizer, evaluates train SCRs, heldout SCRs, and negative-control SCRs, then prints a scalar FINAL_SCORE.

# Acceptance rule

Accept a change only if:

1. FINAL_SCORE decreases
2. holdout_bad_equilibrium_error decreases, or stays zero if already zero
3. holdout_missing_good_equilibrium_error decreases, or stays zero if already zero
4. negative_false_success_rate stays zero
5. no fixed files were modified

Reject and revert otherwise.

# Optimization priority

Optimize in this order:

1. eliminate bad equilibria in generated mechanisms
2. ensure good equilibria exist
3. improve heldout SCR generalization
4. reduce exploitability of intended equilibria
5. reduce table size or complexity only after correctness improves

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
  "final_score": 0.0,
  "holdout_bad_equilibrium_error": 0.0,
  "holdout_missing_good_equilibrium_error": 0.0,
  "negative_false_success_rate": 0.0,
  "accepted": true,
  "reason": "..."
}
```

Never claim success from one SCR. Success is aggregate synthesis performance over many SCRs.
