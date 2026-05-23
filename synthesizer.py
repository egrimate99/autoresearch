"""Editable SCR-to-mechanism synthesizer for variable finite mechanisms."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from envs import Domain, SocialChoiceRule


def scr_features(scr: SocialChoiceRule) -> torch.Tensor:
    utilities = scr.utilities.astype(np.float32)
    scale = max(1.0, float(scr.n_alternatives - 1))
    utilities = utilities / scale

    target = scr.target_mask.astype(np.float32)
    target_sizes = target.sum(axis=1, keepdims=True) / max(1, scr.n_alternatives)

    state_eye = np.eye(scr.n_states, dtype=np.float32)
    alt_eye = np.eye(scr.n_alternatives, dtype=np.float32)
    agent_eye = np.eye(scr.n_agents, dtype=np.float32)

    features = [
        utilities.reshape(-1),
        target.reshape(-1),
        target_sizes.reshape(-1),
        state_eye.reshape(-1),
        alt_eye.reshape(-1),
        agent_eye.reshape(-1),
    ]
    return torch.tensor(np.concatenate(features), dtype=torch.float32)


def feature_dim(domain: Domain) -> int:
    return (
        domain.n_states * domain.n_agents * domain.n_alternatives
        + domain.n_states * domain.n_alternatives
        + domain.n_states
        + domain.n_states * domain.n_states
        + domain.n_alternatives * domain.n_alternatives
        + domain.n_agents * domain.n_agents
    )


def num_max_profiles(domain: Domain, max_messages: int) -> int:
    return max_messages ** domain.n_agents


class TableSynthesizer(nn.Module):
    """Neural generator for variable-size finite mechanisms.

    The model predicts:
    - message-count logits for each agent
    - outcome logits for the padded max-message outcome table

    The decoder chooses message counts and slices the learned table. There is no
    per-SCR brute-force search at inference time.
    """

    def __init__(self, domain: Domain, hidden_dim: int = 256, max_messages: int = 3):
        super().__init__()
        self.domain = domain
        self.hidden_dim = hidden_dim
        self.max_messages = max_messages
        self.profile_count = num_max_profiles(domain, max_messages)
        self.table_output_dim = self.profile_count * domain.n_alternatives
        self.size_output_dim = domain.n_agents * max_messages

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim(domain), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.table_head = nn.Linear(hidden_dim, self.table_output_dim)
        self.size_head = nn.Linear(hidden_dim, self.size_output_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(features)
        table_logits = self.table_head(hidden).view(
            features.shape[0],
            self.profile_count,
            self.domain.n_alternatives,
        )
        size_logits = self.size_head(hidden).view(
            features.shape[0],
            self.domain.n_agents,
            self.max_messages,
        )
        return {"table_logits": table_logits, "size_logits": size_logits}

    def forward_scrs(self, scrs: list[SocialChoiceRule]) -> dict[str, torch.Tensor]:
        features = torch.stack([scr_features(scr) for scr in scrs], dim=0)
        device = next(self.parameters()).device
        return self.forward(features.to(device))
