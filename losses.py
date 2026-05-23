"""Editable learning losses for variable finite-mechanism synthesis."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def synthesis_loss(
    outputs: dict[str, torch.Tensor],
    target_tables: torch.Tensor,
    valid_profile_mask: torch.Tensor,
    target_sizes: torch.Tensor,
) -> torch.Tensor:
    table_logits = outputs["table_logits"]
    size_logits = outputs["size_logits"]
    alternatives = table_logits.shape[-1]

    flat_logits = table_logits.reshape(-1, alternatives)
    flat_targets = target_tables.reshape(-1)
    flat_mask = valid_profile_mask.reshape(-1).bool()
    table_loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])

    size_loss = F.cross_entropy(
        size_logits.reshape(-1, size_logits.shape[-1]),
        target_sizes.reshape(-1),
    )
    return table_loss + 0.25 * size_loss


def table_accuracy(
    outputs: dict[str, torch.Tensor],
    target_tables: torch.Tensor,
    valid_profile_mask: torch.Tensor,
) -> float:
    predictions = torch.argmax(outputs["table_logits"], dim=-1)
    correct = (predictions == target_tables) & valid_profile_mask.bool()
    denom = valid_profile_mask.sum().clamp_min(1)
    return float(correct.sum().float().div(denom).item())


def size_accuracy(outputs: dict[str, torch.Tensor], target_sizes: torch.Tensor) -> float:
    predictions = torch.argmax(outputs["size_logits"], dim=-1)
    return float((predictions == target_sizes).float().mean().item())
