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


def _load_model(path: str | Path) -> tuple[TableSynthesizer, Domain, int]:
    checkpoint = torch.load(_checkpoint_file(path), map_location="cpu")
    domain = Domain(**checkpoint["domain"])
    max_messages = int(checkpoint.get("max_messages", 3))
    model = TableSynthesizer(
        domain,
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
        max_messages=max_messages,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, domain, max_messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify generated mechanisms exactly.")
    parser.add_argument("--config", required=True, help="Evaluation config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or .pt file.")
    parser.add_argument("--json", required=True, help="Output JSON path.")
    args = parser.parse_args()

    config = load_config(args.config)
    scrs = make_dataset(config)
    model, domain, max_messages = _load_model(args.checkpoint)
    decoder = MechanismDecoder(domain, max_messages=max_messages)

    results = []
    with torch.no_grad():
        for scr in scrs:
            outputs = model.forward_scrs([scr])
            mechanism = decoder.decode_logits(scr, outputs)
            result = verify_generated_mechanism(mechanism, scr)
            result["message_sizes"] = list(mechanism.message_sizes)
            result["oracle_complexity"] = float(scr.metadata.get("oracle_complexity", 0.0))
            results.append(result)

    n = max(1, len(results))
    bad_error = sum(item["bad_equilibrium_error"] for item in results) / n
    missing_error = sum(item["missing_good_equilibrium_error"] for item in results) / n
    success_rate = sum(1 for item in results if item["verified"]) / n
    avg_complexity = sum(item["mechanism_complexity"] for item in results) / n
    oracle_values = [float(item.get("oracle_complexity", 0.0)) for item in results]
    positive_oracles = [value for value in oracle_values if value > 0]
    avg_oracle_complexity = sum(positive_oracles) / max(1, len(positive_oracles))
    complexity_ratios = []
    complexity_gaps = []
    for item in results:
        oracle = float(item.get("oracle_complexity", 0.0))
        if oracle > 0:
            complexity_ratios.append(float(item["mechanism_complexity"]) / oracle)
            complexity_gaps.append(float(item["mechanism_complexity"]) - oracle)
    avg_complexity_ratio = sum(complexity_ratios) / max(1, len(complexity_ratios))
    avg_complexity_gap = sum(complexity_gaps) / max(1, len(complexity_gaps))
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
        "avg_oracle_complexity": avg_oracle_complexity,
        "avg_complexity_ratio": avg_complexity_ratio,
        "avg_complexity_gap": avg_complexity_gap,
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
    print(f"VERIFY_AVG_COMPLEXITY={avg_complexity:.6f}")
    print(f"VERIFY_AVG_ORACLE_COMPLEXITY={avg_oracle_complexity:.6f}")
    print(f"VERIFY_AVG_COMPLEXITY_RATIO={avg_complexity_ratio:.6f}")
    print(f"VERIFY_JSON={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
