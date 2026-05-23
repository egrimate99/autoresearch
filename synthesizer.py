"""Editable SCR-to-mechanism synthesizer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from envs import Domain, SocialChoiceRule


def scr_features(scr: SocialChoiceRule) -> torch.Tensor:
    utilities = scr.utilities.astype(np.float32)
    target = scr.target_mask.astype(bool)
    states = np.arange(scr.n_states)
    denom = max(1, scr.n_alternatives - 1)

    top_match = []
    target_utility = []
    for agent in range(scr.n_agents):
        tops = np.argmax(utilities[:, agent, :], axis=1)
        top_match.append(float(np.mean(target[states, tops])))

        values = []
        for theta in range(scr.n_states):
            outcomes = np.flatnonzero(target[theta])
            values.append(float(np.mean(utilities[theta, agent, outcomes]) / denom))
        target_utility.append(float(np.mean(values)))

    target_frequency = target.mean(axis=0).astype(np.float32).tolist()
    target_sizes = target.sum(axis=1).astype(np.float32) / scr.n_alternatives
    features = top_match + target_utility + target_frequency
    features += [float(np.mean(target_sizes)), float(np.std(target_sizes))]
    return torch.tensor(features, dtype=torch.float32)


class TemplateSynthesizer(nn.Module):
    """Small neural classifier over mechanism templates.

    The decoder turns the chosen template into the finite game form G_F.
    """

    def __init__(self, domain: Domain, hidden_dim: int = 64):
        super().__init__()
        self.domain = domain
        self.hidden_dim = hidden_dim
        input_dim = 2 * domain.n_agents + domain.n_alternatives + 2
        output_dim = domain.n_agents + domain.n_alternatives
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def forward_scrs(self, scrs: list[SocialChoiceRule]) -> torch.Tensor:
        features = torch.stack([scr_features(scr) for scr in scrs], dim=0)
        device = next(self.parameters()).device
        return self.forward(features.to(device))
