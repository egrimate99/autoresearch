"""Editable learning losses for full-table mechanism synthesis."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def synthesis_loss(logits: torch.Tensor, target_tables: torch.Tensor) -> torch.Tensor:
    """Supervise every finite message profile's outcome.

    logits: [batch, profiles, alternatives]
    target_tables: [batch, profiles]
    """

    alternatives = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, alternatives), target_tables.reshape(-1))


def table_accuracy(logits: torch.Tensor, target_tables: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=-1)
    return float((predictions == target_tables).float().mean().item())
