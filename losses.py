"""Editable losses for the mechanism synthesizer."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def synthesis_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def template_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=-1)
    return float((predictions == labels).float().mean().item())
