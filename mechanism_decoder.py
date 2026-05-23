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
        return tuple([self.domain.n_states] + [1] * (self.domain.n_agents - 1))

    def decode_logits(self, scr: SocialChoiceRule, logits: torch.Tensor) -> Mechanism:
        if logits.ndim == 1:
            logits = logits.view(-1, scr.n_alternatives)
        full_shape = tuple([2 * scr.n_states] * scr.n_agents) + (scr.n_alternatives,)
        full_logits = logits.detach().cpu().view(full_shape)

        state_outcomes = []
        for state in range(scr.n_states):
            full_profile = tuple([state] * scr.n_agents)
            state_outcomes.append(int(torch.argmax(full_logits[full_profile]).item()))

        unique_outcomes = sorted(set(state_outcomes))
        outcome_to_message = {outcome: i for i, outcome in enumerate(unique_outcomes)}
        message_sizes = tuple([len(unique_outcomes)] + [1] * (scr.n_agents - 1))
        outcome_table = np.zeros(message_sizes, dtype=np.int64)
        for outcome, message in outcome_to_message.items():
            profile = tuple([message] + [0] * (scr.n_agents - 1))
            outcome_table[profile] = outcome

        return Mechanism(
            domain=scr.domain,
            message_sizes=message_sizes,
            outcome_table=outcome_table,
            template_name="learned_outcome_class_report",
            metadata={"message_scaffold": "outcome_class_report_from_learned_logits"},
        )
