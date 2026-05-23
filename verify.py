"""Fixed verifier for checkpointed SCR-to-mechanism synthesizers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from envs import Domain, load_config
from equilibrium import verify_generated_mechanism
from mechanism_decoder import MechanismDecoder
from scr_dataset import make_dataset
from synthesizer import TableSynthesizer


def _checkpoint_file(path: str | Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def _load_model(path: str | Path) -> tuple[TableSynthesizer, Domain]:
    checkpoint = torch.load(_checkpoint_file(path), map_location="cpu")
    domain = Domain(**checkpoint["domain"])
    model = TableSynthesizer(domain, hidden_dim=int(checkpoint.get("hidden_dim", 256)))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, domain


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify generated mechanisms exactly.")
    parser.add_argument("--config", required=True, help="Evaluation config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or .pt file.")
    parser.add_argument("--json", required=True, help="Output JSON path.")
    args = parser.parse_args()

    config = load_config(args.config)
    scrs = make_dataset(config)
    model, domain = _load_model(args.checkpoint)
    decoder = MechanismDecoder(domain)

    results = []
    with torch.no_grad():
        for scr in scrs:
            logits = model.forward_scrs([scr])[0]
            mechanism = decoder.decode_logits(scr, logits)
            result = verify_generated_mechanism(mechanism, scr)
            result["prediction_table_shape"] = list(logits.shape)
            results.append(result)

    n = max(1, len(results))
    bad_error = sum(item["bad_equilibrium_error"] for item in results) / n
    missing_error = sum(item["missing_good_equilibrium_error"] for item in results) / n
    success_rate = sum(1 for item in results if item["verified"]) / n
    avg_complexity = sum(item["mechanism_complexity"] for item in results) / n
    split = str(config.get("dataset", {}).get("split", "unknown"))
    is_negative = bool(config.get("dataset", {}).get("only_nonimplementable", False)) or split == "negative"

    summary = {
        "split": split,
        "num_scrs": len(results),
        "bad_equilibrium_error": bad_error,
        "missing_good_equilibrium_error": missing_error,
        "synthesis_success_rate": success_rate,
        "negative_false_success_rate": success_rate if is_negative else 0.0,
        "avg_mechanism_complexity": avg_complexity,
    }

    payload = {
        "summary": summary,
        "results": results,
    }
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"VERIFY_SPLIT={split}")
    print(f"VERIFY_NUM_SCRS={len(results)}")
    print(f"VERIFY_BAD_EQUILIBRIUM_ERROR={bad_error:.6f}")
    print(f"VERIFY_MISSING_GOOD_EQUILIBRIUM_ERROR={missing_error:.6f}")
    print(f"VERIFY_SUCCESS_RATE={success_rate:.6f}")
    print(f"VERIFY_JSON={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
