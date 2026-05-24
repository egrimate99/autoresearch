# Codex Cloud handoff

This repo supports three practical execution modes:

```powershell
# Local CPU
bash scripts/eval_once.sh cpu

# Local CUDA GPU
bash scripts/eval_once.sh gpu

# Local auto-detect
bash scripts/eval_once.sh auto
```

The same selector works inside an autoresearch prompt: tell the agent which
command to run after each attempted change.

## OpenAI Codex Cloud

Codex Cloud runs in an OpenAI-managed container checked out from GitHub. It
cannot see `C:\Mate\Mathematics\implementation_autoresearch` directly, so the
branch must be pushed first.

From PowerShell:

```powershell
git status --short
git push winrtx winrtx-setup
```

Then:

1. Open `https://chatgpt.com/codex`.
2. Connect GitHub if needed.
3. Select `jsegov/autoresearch-win-rtx`.
4. Select branch `winrtx-setup`.
5. Paste the prompt from `.github/codex/prompts/autoresearch-loop.md`.

Use `bash scripts/probe_runtime.sh` as the first command in any new cloud
environment. Codex Cloud does not document a guaranteed GPU for these tasks, so
the prompt uses:

```bash
bash scripts/eval_logged.sh auto --hypothesis "remote baseline before autoresearch loop"
```

That uses CUDA if the cloud PyTorch runtime sees it, otherwise CPU.

If environment setup asks for a setup script, use:

```bash
bash scripts/setup_env.sh
```

The scripts accept either a standalone `uv` binary or `python -m uv`. Codex
Cloud often has `/root/.local/bin/uv` already installed; that is enough.

Use `scripts/eval_logged.sh` for every cloud and local attempt. It stores the
full raw output and archived metric files under `results/eval_logs/`, so a
cloud run can later be resumed locally without losing rejected-idea history.

## Important limitation

The ChatGPT/Codex subscription remote path is started from Codex web, IDE, or
GitHub integration. This repository can make the handoff easy and reproducible,
but it cannot itself force-start a Codex Cloud task from a local shell.

There is also a GitHub Actions Codex path, but that requires an OpenAI API key
stored as a GitHub secret and uses API billing/secrets rather than just the
ChatGPT subscription UI.
