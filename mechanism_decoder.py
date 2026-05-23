"""Editable decoder from learned table logits to finite mechanisms."""

from __future__ import annotations

import numpy as np
import torch

from envs import Domain, Mechanism, SocialChoiceRule


class MechanismDecoder:
    """Decode learned logits into a compact state-report mechanism.

    The synthesizer is trained on the full report/challenge scaffold. At decode
    time, we keep only the learned outcomes at unanimous report profiles and
    expose a smaller mechanism where agent 0 reports a state and the other
    agents have one dummy message.
    """

    def __init__(self, domain: Domain):
        self.domain = domain

    @property
    def message_sizes(self) -> tuple[int, ...]:
        return tuple(
            self.domain.n_states if agent == 0 else 1
            for agent in range(self.domain.n_agents)
        )

    def decode_logits(self, scr: SocialChoiceRule, logits: torch.Tensor) -> Mechanism:
        if logits.ndim == 1:
            logits = logits.view(-1, scr.n_alternatives)
        full_shape = tuple([2 * scr.n_states] * scr.n_agents) + (scr.n_alternatives,)
        full_logits = logits.detach().cpu().view(full_shape)

        outcome_table = np.zeros(self.message_sizes, dtype=np.int64)
        for state in range(scr.n_states):
            full_profile = tuple([state] * scr.n_agents)
            compact_profile = tuple([state] + [0] * (scr.n_agents - 1))
            outcome_table[compact_profile] = int(torch.argmax(full_logits[full_profile]).item())

        return Mechanism(
            domain=scr.domain,
            message_sizes=self.message_sizes,
            outcome_table=outcome_table,
            template_name="learned_compact_state_report",
            metadata={"message_scaffold": "compact_state_report_from_learned_logits"},
        )
