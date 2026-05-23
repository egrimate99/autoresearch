"""Editable SCR-to-mechanism table synthesizer."""

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

    state_eye = np.eye(scr.n_states, dtype=np.float32)
    alt_eye = np.eye(scr.n_alternatives, dtype=np.float32)
    agent_eye = np.eye(scr.n_agents, dtype=np.float32)

    features = [
        utilities.reshape(-1),
        target.reshape(-1),
        state_eye.reshape(-1),
        alt_eye.reshape(-1),
        agent_eye.reshape(-1),
    ]
    return torch.tensor(np.concatenate(features), dtype=torch.float32)


def feature_dim(domain: Domain) -> int:
    return (
        domain.n_states * domain.n_agents * domain.n_alternatives
        + domain.n_states * domain.n_alternatives
        + domain.n_states * domain.n_states
        + domain.n_alternatives * domain.n_alternatives
        + domain.n_agents * domain.n_agents
    )


def num_message_profiles(domain: Domain) -> int:
    return (2 * domain.n_states) ** domain.n_agents


class TableSynthesizer(nn.Module):
    """Neural generator for full finite-mechanism outcome tables.

    The model outputs logits with shape [profiles, alternatives]. Decoding uses
    argmax independently for each finite message profile; no verifier feedback
    or per-instance search is used at inference time.
    """

    def __init__(self, domain: Domain, hidden_dim: int = 256):
        super().__init__()
        self.domain = domain
        self.hidden_dim = hidden_dim
        self.profile_count = num_message_profiles(domain)
        self.output_dim = self.profile_count * domain.n_alternatives

        self.net = nn.Sequential(
            nn.Linear(feature_dim(domain), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features)
        return logits.view(features.shape[0], self.profile_count, self.domain.n_alternatives)

    def forward_scrs(self, scrs: list[SocialChoiceRule]) -> torch.Tensor:
        features = torch.stack([scr_features(scr) for scr in scrs], dim=0)
        device = next(self.parameters()).device
        return self.forward(features.to(device))
