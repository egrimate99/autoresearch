"""Editable decoder from synthesizer logits to finite mechanisms."""

from __future__ import annotations

from itertools import product

import numpy as np
import torch

from envs import Domain, Mechanism, SocialChoiceRule


class MechanismDecoder:
    """Decode template logits into small finite mechanisms.

    Initial templates:
    - dictator_i: only agent i sends an alternative; others send one dummy message
    - constant_a: every agent sends one dummy message
    """

    def __init__(self, domain: Domain):
        self.domain = domain

    @property
    def num_templates(self) -> int:
        return self.domain.n_agents + self.domain.n_alternatives

    def template_name(self, index: int) -> str:
        if index < self.domain.n_agents:
            return f"dictator_{index}"
        return f"constant_{index - self.domain.n_agents}"

    def decode_logits(self, scr: SocialChoiceRule, logits: torch.Tensor) -> Mechanism:
        index = int(torch.argmax(logits).item())
        return self.decode_template(scr, index)

    def decode_template(self, scr: SocialChoiceRule, index: int) -> Mechanism:
        if index < scr.n_agents:
            dictator = index
            message_sizes = tuple(
                scr.n_alternatives if agent == dictator else 1
                for agent in range(scr.n_agents)
            )
            outcome_table = np.zeros(message_sizes, dtype=np.int64)
            for profile in product(*(range(size) for size in message_sizes)):
                outcome_table[profile] = profile[dictator]
            metadata = {"dictator": dictator}
        else:
            alternative = index - scr.n_agents
            message_sizes = tuple([1] * scr.n_agents)
            outcome_table = np.zeros(message_sizes, dtype=np.int64)
            outcome_table.fill(alternative)
            metadata = {"constant_alternative": alternative}

        return Mechanism(
            domain=scr.domain,
            message_sizes=message_sizes,
            outcome_table=outcome_table,
            template_name=self.template_name(index),
            metadata=metadata,
        )
