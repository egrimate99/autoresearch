"""Train a learned minimal finite-mechanism synthesizer.

Usage:
    python train.py --config configs/train.yaml --out results/latest
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from envs import Domain, load_config
from losses import size_accuracy, synthesis_loss, table_accuracy
from scr_dataset import make_dataset, max_message_size, max_message_profiles
from synthesizer import TableSynthesizer, scr_features


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _training_config(config: dict[str, Any], seed_override: int | None) -> dict[str, Any]:
    training = dict(config.get("training", {}))
    if seed_override is not None:
        training["seed"] = int(seed_override)
        config.setdefault("training", {})["seed"] = int(seed_override)
    training.setdefault("seed", 0)
    training.setdefault("epochs", 80)
    training.setdefault("batch_size", 32)
    training.setdefault("lr", 0.0005)
    training.setdefault("hidden_dim", 256)
    training.setdefault("device", "cpu")
    training.setdefault("effective_num_scrs", 2048)
    training.setdefault("rl_steps", 0)
    training.setdefault("equilibrium_loss_weight", 0.05)
    training["effective_num_scrs"] = max(int(training["effective_num_scrs"]), 8192)
    return training


def _targets(scrs, domain: Domain, max_messages: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    profile_count = max_message_profiles(domain, max_messages)
    tables = []
    masks = []
    sizes = []
    for scr in scrs:
        if scr.teacher_outcome_table is None or scr.teacher_message_sizes is None:
            raise ValueError("Training data must include teacher mechanisms.")
        flat_table = torch.tensor(scr.teacher_outcome_table.reshape(-1), dtype=torch.long)
        mask = torch.zeros(profile_count, dtype=torch.bool)
        for profile in itertools.product(*(range(size) for size in scr.teacher_message_sizes)):
            flat_idx = 0
            stride = 1
            for message in reversed(profile):
                flat_idx += message * stride
                stride *= max_messages
            mask[flat_idx] = True
        tables.append(flat_table)
        masks.append(mask)
        sizes.append(torch.tensor([size - 1 for size in scr.teacher_message_sizes], dtype=torch.long))
    return torch.stack(tables, dim=0), torch.stack(masks, dim=0), torch.stack(sizes, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a minimal finite-mechanism synthesizer.")
    parser.add_argument("--config", required=True, help="Training config YAML.")
    parser.add_argument("--out", required=True, help="Output checkpoint directory.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    args = parser.parse_args()

    config = load_config(args.config)
    training = _training_config(config, args.seed)
    seed = int(training["seed"])
    _set_seed(seed)

    domain = Domain.from_config(config)
    max_messages = max_message_size(config)
    train_config = dict(config)
    train_config["dataset"] = dict(config.get("dataset", {}))
    train_config["dataset"]["num_scrs"] = max(
        int(train_config["dataset"].get("num_scrs", 0)),
        int(training["effective_num_scrs"]),
    )
    scrs = make_dataset(train_config)
    features = torch.stack([scr_features(scr) for scr in scrs], dim=0)
    target_tables, valid_profile_mask, target_sizes = _targets(scrs, domain, max_messages)
    utilities = torch.tensor(np.stack([scr.utilities for scr in scrs], axis=0), dtype=torch.float32)
    utility_scale = max(1.0, float(domain.n_alternatives - 1))
    utilities = utilities / utility_scale
    scr_target_mask = torch.tensor(
        np.stack([scr.target_mask for scr in scrs], axis=0),
        dtype=torch.float32,
    )

    requested_device = str(training.get("device", "cpu"))
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = TableSynthesizer(
        domain,
        hidden_dim=int(training["hidden_dim"]),
        max_messages=max_messages,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["lr"]), weight_decay=1e-4)
    features = features.to(device)
    target_tables = target_tables.to(device)
    valid_profile_mask = valid_profile_mask.to(device)
    target_sizes = target_sizes.to(device)
    utilities = utilities.to(device)
    scr_target_mask = scr_target_mask.to(device)

    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    equilibrium_weight = float(training["equilibrium_loss_weight"])
    n = features.shape[0]
    last_loss = 0.0
    last_table_acc = 0.0
    last_size_acc = 0.0

    for epoch in range(epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            outputs = model(features[idx])
            loss = synthesis_loss(
                outputs,
                target_tables[idx],
                valid_profile_mask[idx],
                target_sizes[idx],
                utilities[idx],
                scr_target_mask[idx],
                equilibrium_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if epoch == epochs - 1 or epoch % max(1, epochs // 10) == 0:
            with torch.no_grad():
                outputs = model(features)
                last_loss = float(
                    synthesis_loss(
                        outputs,
                        target_tables,
                        valid_profile_mask,
                        target_sizes,
                        utilities,
                        scr_target_mask,
                        equilibrium_weight,
                    ).item()
                )
                last_table_acc = table_accuracy(outputs, target_tables, valid_profile_mask)
                last_size_acc = size_accuracy(outputs, target_sizes)
            print(
                f"epoch={epoch:04d} train_loss={last_loss:.6f} "
                f"table_acc={last_table_acc:.4f} size_acc={last_size_acc:.4f}"
            )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.cpu().state_dict(),
        "domain": domain.to_dict(),
        "hidden_dim": int(training["hidden_dim"]),
        "max_messages": max_messages,
        "config": config,
        "train_loss": last_loss,
        "table_acc": last_table_acc,
        "size_acc": last_size_acc,
    }
    torch.save(checkpoint, out_dir / "checkpoint.pt")

    summary = {
        "num_scrs": len(scrs),
        "seed": seed,
        "train_loss": last_loss,
        "table_acc": last_table_acc,
        "size_acc": last_size_acc,
        "checkpoint": str(out_dir / "checkpoint.pt"),
    }
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"TRAIN_LOSS={last_loss:.6f}")
    print(f"TRAIN_TABLE_ACC={last_table_acc:.6f}")
    print(f"TRAIN_SIZE_ACC={last_size_acc:.6f}")
    print(f"CHECKPOINT={out_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
