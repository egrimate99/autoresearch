"""Editable decoder from learned table logits to finite mechanisms."""

from __future__ import annotations

import numpy as np
import torch

from envs import Domain, Mechanism, SocialChoiceRule


class MechanismDecoder:
    """Decode learned logits on the fixed report/challenge scaffold.

    There are no hand-coded solution templates here. The synthesizer must learn
    which alternative each finite message profile should map to.
    """

    def __init__(self, domain: Domain):
        self.domain = domain

    @property
    def message_sizes(self) -> tuple[int, ...]:
        return tuple([2 * self.domain.n_states] * self.domain.n_agents)

    def decode_logits(self, scr: SocialChoiceRule, logits: torch.Tensor) -> Mechanism:
        if logits.ndim == 1:
            logits = logits.view(-1, scr.n_alternatives)
        predicted = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64)
        outcome_table = predicted.reshape(self.message_sizes)
        return Mechanism(
            domain=scr.domain,
            message_sizes=self.message_sizes,
            outcome_table=outcome_table,
            template_name="learned_outcome_table",
            metadata={"message_scaffold": "report_or_challenge_state"},
        )
