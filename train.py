"""Train a learned SCR-to-mechanism table synthesizer.

Usage:
    python train.py --config configs/train.yaml --out results/latest
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from envs import Domain, load_config
from losses import synthesis_loss, table_accuracy
from scr_dataset import make_dataset
from synthesizer import TableSynthesizer, scr_features


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _teacher_tables(scrs) -> torch.Tensor:
    tables = []
    for scr in scrs:
        if scr.teacher_outcome_table is None:
            raise ValueError("Training data must include teacher outcome tables.")
        tables.append(torch.tensor(scr.teacher_outcome_table.reshape(-1), dtype=torch.long))
    return torch.stack(tables, dim=0)


def _training_config(config: dict[str, Any], seed_override: int | None) -> dict[str, Any]:
    training = dict(config.get("training", {}))
    if seed_override is not None:
        training["seed"] = int(seed_override)
        config.setdefault("training", {})["seed"] = int(seed_override)
    training.setdefault("seed", 0)
    training.setdefault("epochs", 80)
    training.setdefault("batch_size", 32)
    training.setdefault("lr", 0.0003)
    training.setdefault("hidden_dim", 256)
    training.setdefault("device", "cpu")
    return training


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a finite-mechanism table synthesizer.")
    parser.add_argument("--config", required=True, help="Training config YAML.")
    parser.add_argument("--out", required=True, help="Output checkpoint directory.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    args = parser.parse_args()

    config = load_config(args.config)
    training = _training_config(config, args.seed)
    seed = int(training["seed"])
    _set_seed(seed)

    domain = Domain.from_config(config)
    scrs = make_dataset(config)
    features = torch.stack([scr_features(scr) for scr in scrs], dim=0)
    targets = _teacher_tables(scrs)

    requested_device = str(training.get("device", "cpu"))
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = TableSynthesizer(domain, hidden_dim=int(training["hidden_dim"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["lr"]), weight_decay=1e-4)
    features = features.to(device)
    targets = targets.to(device)

    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    n = features.shape[0]
    last_loss = 0.0
    last_acc = 0.0

    for epoch in range(epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            logits = model(features[idx])
            loss = synthesis_loss(logits, targets[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if epoch == epochs - 1 or epoch % max(1, epochs // 10) == 0:
            with torch.no_grad():
                logits = model(features)
                last_loss = float(synthesis_loss(logits, targets).item())
                last_acc = table_accuracy(logits, targets)
            print(f"epoch={epoch:04d} train_loss={last_loss:.6f} table_acc={last_acc:.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.cpu().state_dict(),
        "domain": domain.to_dict(),
        "hidden_dim": int(training["hidden_dim"]),
        "config": config,
        "train_loss": last_loss,
        "table_acc": last_acc,
    }
    torch.save(checkpoint, out_dir / "checkpoint.pt")

    summary = {
        "num_scrs": len(scrs),
        "seed": seed,
        "train_loss": last_loss,
        "table_acc": last_acc,
        "checkpoint": str(out_dir / "checkpoint.pt"),
    }
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"TRAIN_LOSS={last_loss:.6f}")
    print(f"TRAIN_TABLE_ACC={last_acc:.6f}")
    print(f"CHECKPOINT={out_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
