# Objective

We are improving a learned synthesizer for minimal finite Nash implementation.

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

The verifier checks exact pure Nash implementation:

```text
NE(G_F, theta) outcomes = F(theta)
```

for every state theta.

The benchmark now has two goals:

1. correctness: implement the SCR exactly
2. minimality: use as little message space / outcome-table complexity as possible

# Benchmark Structure

Positive SCRs are generated from varied small latent mechanisms, not from a
state-report construction. For each latent mechanism H and each state theta,
the generator samples utilities and sets F(theta) equal to the outcomes of H's
pure Nash equilibria at theta. Therefore H implements F by construction.

The model sees only F. It does not see the latent mechanism at evaluation time.

The evaluator accepts any generated mechanism that implements F. It then scores
message-space complexity against the latent mechanism's reference complexity.
This is not a proof of global minimality, but it gives the learner a concrete
compactness target and makes the task about finding small implementations, not
reproducing a known Maskin/Moore-Repullo style construction with states in the
message.

# Editable Files

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
- neural architectures that map SCR tensors to finite mechanisms
- predicting variable message counts
- differentiable losses over generated tables
- verifier-reward fine-tuning or policy-gradient/RL inside train.py
- curriculum inside train.py using the fixed dataset API
- decoding improvements that do not inspect verifier output per instance

Disallowed:
- hard-coded direct reconstruction of the fixed generator
- reading latent mechanism metadata at inference time
- per-SCR brute force search over mechanisms
- changing the verifier, generator, configs, or aggregate score during a run
- using negative-control labels at inference time

# Evaluation Loop

After every attempted change, run:

```bash
./scripts/eval_logged.sh auto --hypothesis "short description of this attempt"
```

This wraps `eval_once.sh`, trains the synthesizer, evaluates train SCRs,
heldout SCRs, and negative controls, then prints a scalar FINAL_SCORE.

The wrapper must be used for every iteration because it archives the full
stdout/stderr output, train summary, verifier JSONs, git status, and pre-run
diff under `results/eval_logs/<run_id>/`, then appends an index record to
`results/eval_logs/index.jsonl`.

# Acceptance Rule

Accept a change only if:

1. FINAL_SCORE decreases
2. holdout_bad_equilibrium_error decreases, or stays zero if already zero
3. holdout_missing_good_equilibrium_error decreases, or stays zero if already zero
4. negative_false_success_rate stays zero
5. no fixed files were modified

Reject and revert otherwise.

# Optimization Priority

Optimize in this order:

1. eliminate bad equilibria in generated mechanisms
2. ensure good equilibria exist
3. keep negative false success at zero
4. improve heldout generalization
5. minimize message-space complexity relative to the oracle reference

# Logging

Preserve all experiment information, even rejected ideas. Every iteration must
leave enough data in the repository to reconstruct what happened when the same
branch is later resumed locally or in Codex Cloud.

For every attempted change:

1. run `scripts/eval_logged.sh`
2. keep the generated `results/eval_logs/<run_id>/` directory
3. keep `results/eval_logs/index.jsonl`
4. append a decision record to `results/runs.jsonl`
5. append a human-readable note to `results/idea_log.md`
6. commit the logs after the decision is made

Append one JSON object to results/runs.jsonl after every experiment:

```json
{
  "timestamp": "...",
  "hypothesis": "...",
  "eval_log_dir": "results/eval_logs/...",
  "eval_output_log": "results/eval_logs/.../output.log",
  "files_changed": ["..."],
  "train_score": 0.0,
  "holdout_score": 0.0,
  "negative_score": 0.0,
  "minimality_score": 0.0,
  "final_score": 0.0,
  "holdout_bad_equilibrium_error": 0.0,
  "holdout_missing_good_equilibrium_error": 0.0,
  "negative_false_success_rate": 0.0,
  "avg_complexity_ratio": 0.0,
  "accepted": true,
  "reason": "..."
}
```

Never claim success from one SCR. Success is aggregate heldout correctness plus
compactness over many SCRs.
