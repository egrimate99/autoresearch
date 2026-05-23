"""Editable learning losses for variable finite-mechanism synthesis."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _full_deviation_indices(n_agents: int, max_messages: int, device: torch.device) -> torch.Tensor:
    profiles = torch.cartesian_prod(*[torch.arange(max_messages, device=device) for _ in range(n_agents)])
    strides = torch.tensor(
        [max_messages ** power for power in reversed(range(n_agents))],
        dtype=torch.long,
        device=device,
    )
    indices = []
    for agent in range(n_agents):
        agent_indices = []
        for message in range(max_messages):
            deviated = profiles.clone()
            deviated[:, agent] = message
            agent_indices.append((deviated * strides).sum(dim=-1))
        indices.append(torch.stack(agent_indices, dim=-1))
    return torch.stack(indices, dim=1)


def equilibrium_surrogate_loss(
    outputs: dict[str, torch.Tensor],
    utilities: torch.Tensor,
    target_mask: torch.Tensor,
    deviation_margin: float = 0.10,
    temperature: float = 0.25,
) -> torch.Tensor:
    table_logits = outputs["table_logits"]
    batch_size, profile_count, _ = table_logits.shape
    n_agents = utilities.shape[2]
    max_messages = int(round(profile_count ** (1.0 / n_agents)))
    if max_messages ** n_agents != profile_count:
        raise ValueError("Profile count must be max_messages ** n_agents.")

    probs = F.softmax(table_logits, dim=-1)
    expected_utilities = torch.einsum("bpa,bsia->bspi", probs, utilities)
    target_probability = torch.einsum("bpa,bsa->bsp", probs, target_mask.float())

    deviation_indices = _full_deviation_indices(n_agents, max_messages, table_logits.device)
    gain_terms = []
    for agent in range(n_agents):
        current = expected_utilities[..., agent]
        for message in range(max_messages):
            deviated = expected_utilities[:, :, deviation_indices[:, agent, message], agent]
            gain_terms.append(deviated - current)
    max_gain = torch.stack(gain_terms, dim=0).amax(dim=0)

    non_target_instability = (1.0 - target_probability) * F.relu(deviation_margin - max_gain)
    stable_target_score = target_probability * torch.sigmoid(-max_gain / max(temperature, 1.0e-6))
    best_stable_target = stable_target_score.amax(dim=-1)
    missing_good = F.relu(1.0 - best_stable_target)
    return non_target_instability.mean() + 0.5 * missing_good.mean()


def synthesis_loss(
    outputs: dict[str, torch.Tensor],
    target_tables: torch.Tensor,
    valid_profile_mask: torch.Tensor,
    target_sizes: torch.Tensor,
    utilities: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    equilibrium_weight: float = 0.0,
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
    loss = table_loss + 0.25 * size_loss
    if equilibrium_weight > 0.0:
        if utilities is None or target_mask is None:
            raise ValueError("utilities and target_mask are required for equilibrium loss.")
        loss = loss + equilibrium_weight * equilibrium_surrogate_loss(
            outputs,
            utilities,
            target_mask,
        )
    return loss


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
