"""Editable decoder from learned logits to finite mechanisms."""

from __future__ import annotations

import itertools

import numpy as np
import torch

from envs import Domain, Mechanism, SocialChoiceRule


class MechanismDecoder:
    """Decode variable-size mechanisms from learned logits.

    The decoder picks each agent's message count from the learned size logits
    and slices the learned padded outcome table. It does not inspect the
    verifier or solve a per-SCR search problem.
    """

    def __init__(self, domain: Domain, max_messages: int = 3):
        self.domain = domain
        self.max_messages = max_messages

    def decode_logits(self, scr: SocialChoiceRule, outputs: dict[str, torch.Tensor]) -> Mechanism:
        table_logits = outputs["table_logits"]
        size_logits = outputs["size_logits"]
        if table_logits.ndim == 3:
            table_logits = table_logits[0]
        if size_logits.ndim == 3:
            size_logits = size_logits[0]

        size_labels = torch.argmax(size_logits, dim=-1).detach().cpu().numpy().astype(int)
        message_sizes = tuple((size_labels + 1).tolist())

        allowed = torch.as_tensor(
            scr.target_mask.any(axis=0),
            dtype=torch.bool,
            device=table_logits.device,
        )
        masked_logits = table_logits.clone()
        masked_logits[:, ~allowed] = -1.0e9
        target_counts = torch.as_tensor(
            scr.target_mask.sum(axis=0),
            dtype=masked_logits.dtype,
            device=masked_logits.device,
        )
        masked_logits = masked_logits + 0.10 * torch.log(target_counts.clamp_min(1.0))
        predicted = torch.argmax(masked_logits, dim=-1).detach().cpu().numpy().astype(np.int64)
        full_table = predicted.reshape(tuple([self.max_messages] * scr.n_agents))
        outcome_table = np.zeros(message_sizes, dtype=np.int64)
        for profile in itertools.product(*(range(size) for size in message_sizes)):
            outcome_table[profile] = full_table[profile]

        return Mechanism(
            domain=scr.domain,
            message_sizes=message_sizes,
            outcome_table=outcome_table,
            template_name="learned_variable_table",
            metadata={"max_messages": self.max_messages},
        )
