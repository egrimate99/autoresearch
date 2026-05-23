"""Aggregate exact verification outputs into one autoresearch score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _summary(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["summary"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate SCR synthesis metrics.")
    parser.add_argument("train_json")
    parser.add_argument("holdout_json")
    parser.add_argument("negative_json")
    args = parser.parse_args()

    train = _summary(args.train_json)
    holdout = _summary(args.holdout_json)
    negative = _summary(args.negative_json)

    train_bad = float(train["bad_equilibrium_error"])
    train_missing = float(train["missing_good_equilibrium_error"])
    holdout_bad = float(holdout["bad_equilibrium_error"])
    holdout_missing = float(holdout["missing_good_equilibrium_error"])
    negative_false = float(negative["negative_false_success_rate"])
    avg_complexity = float(holdout["avg_mechanism_complexity"])
    avg_oracle_complexity = float(holdout.get("avg_oracle_complexity", 0.0))
    avg_complexity_ratio = float(holdout.get("avg_complexity_ratio", 0.0))
    avg_complexity_gap = float(holdout.get("avg_complexity_gap", 0.0))

    train_score = 20.0 * train_bad + 10.0 * train_missing
    holdout_score = 200.0 * holdout_bad + 100.0 * holdout_missing
    negative_score = 1000.0 * negative_false
    minimality_score = 5.0 * avg_complexity_ratio + 0.1 * avg_complexity_gap
    final_score = train_score + holdout_score + negative_score + minimality_score

    print(f"FINAL_SCORE={final_score:.6f}")
    print(f"TRAIN_SCORE={train_score:.6f}")
    print(f"HOLDOUT_SCORE={holdout_score:.6f}")
    print(f"NEGATIVE_SCORE={negative_score:.6f}")
    print(f"MINIMALITY_SCORE={minimality_score:.6f}")
    print("ACCEPT=false")
    print(f"train_bad_equilibrium_error={train_bad:.6f}")
    print(f"train_missing_good_equilibrium_error={train_missing:.6f}")
    print(f"holdout_bad_equilibrium_error={holdout_bad:.6f}")
    print(f"holdout_missing_good_equilibrium_error={holdout_missing:.6f}")
    print(f"negative_false_success_rate={negative_false:.6f}")
    print(f"avg_mechanism_complexity={avg_complexity:.6f}")
    print(f"avg_oracle_complexity={avg_oracle_complexity:.6f}")
    print(f"avg_complexity_ratio={avg_complexity_ratio:.6f}")
    print(f"avg_complexity_gap={avg_complexity_gap:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
