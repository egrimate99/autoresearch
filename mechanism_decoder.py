"""Editable decoder from learned logits to finite mechanisms."""

from __future__ import annotations

import itertools

import numpy as np
import torch

from envs import Domain, Mechanism, SocialChoiceRule


def _flat_profile_index(profile: tuple[int, ...], max_messages: int) -> int:
    flat_idx = 0
    stride = 1
    for message in reversed(profile):
        flat_idx += message * stride
        stride *= max_messages
    return flat_idx


def _merge_duplicate_messages(outcome_table: np.ndarray, message_sizes: tuple[int, ...]) -> tuple[np.ndarray, tuple[int, ...]]:
    table = outcome_table
    sizes = list(message_sizes)
    for axis in range(len(sizes)):
        keep = []
        seen = set()
        for message in range(sizes[axis]):
            key = tuple(np.take(table, message, axis=axis).reshape(-1).astype(int).tolist())
            if key in seen:
                continue
            seen.add(key)
            keep.append(message)
        if len(keep) == sizes[axis]:
            continue
        table = np.take(table, keep, axis=axis)
        sizes[axis] = len(keep)
    return table, tuple(sizes)


def _ensure_singleton_target_coverage(
    outcome_table: np.ndarray,
    message_sizes: tuple[int, ...],
    scr: SocialChoiceRule,
    masked_logits: torch.Tensor,
    max_messages: int,
) -> np.ndarray:
    singleton_states = scr.target_mask.sum(axis=1) == 1
    if np.any(singleton_states):
        required = set(np.argmax(scr.target_mask[singleton_states], axis=1).astype(int).tolist())
    else:
        required = set()
    missing = sorted(required - set(outcome_table.reshape(-1).astype(int).tolist()))

    target_union = set(np.flatnonzero(scr.target_mask.any(axis=0)).astype(int).tolist())
    candidates = []
    for profile in itertools.product(*(range(size) for size in message_sizes)):
        flat_idx = _flat_profile_index(profile, max_messages)
        top2 = torch.topk(masked_logits[flat_idx], k=min(2, masked_logits.shape[-1])).values
        margin = float((top2[0] - top2[-1]).detach().cpu().item())
        current_outcome = int(outcome_table[profile])
        outside_union_rank = 0 if current_outcome not in target_union else 1
        candidates.append((outside_union_rank, margin, profile))
    candidates.sort(key=lambda item: (item[0], item[1]))

    adjusted = outcome_table.copy()
    for outcome in missing:
        target_states = [
            int(theta)
            for theta in np.flatnonzero(singleton_states)
            if int(np.argmax(scr.target_mask[theta])) == outcome
        ]
        if not candidates:
            break
        best_index, (_, _, profile) = min(
            enumerate(candidates),
            key=lambda item: (
                _singleton_stability_cost(outcome_table, message_sizes, scr, item[1][2], outcome, target_states),
                item[1][0],
                item[1][1],
            ),
        )
        adjusted[profile] = outcome
        candidates.pop(best_index)

    for theta in np.flatnonzero(singleton_states):
        outcome = int(np.argmax(scr.target_mask[theta]))
        existing_costs = [
            _singleton_stability_cost(adjusted, message_sizes, scr, profile, outcome, [int(theta)])
            for profile in itertools.product(*(range(size) for size in message_sizes))
            if int(adjusted[profile]) == outcome
        ]
        if existing_costs and min(existing_costs) <= 1.0e-9:
            continue
        if not candidates:
            break
        best_index, (_, _, profile) = min(
            enumerate(candidates),
            key=lambda item: (
                _singleton_stability_cost(adjusted, message_sizes, scr, item[1][2], outcome, [int(theta)]),
                item[1][0],
                item[1][1],
            ),
        )
        adjusted[profile] = outcome
        candidates.pop(best_index)

    for theta in range(scr.n_states):
        target_outcomes = np.flatnonzero(scr.target_mask[theta]).astype(int).tolist()
        if any(
            _singleton_stability_cost(adjusted, message_sizes, scr, profile, int(outcome), [int(theta)])
            <= 1.0e-9
            for profile in itertools.product(*(range(size) for size in message_sizes))
            for outcome in target_outcomes
            if int(adjusted[profile]) == int(outcome)
        ):
            continue
        if not candidates:
            break
        best_index, (_, _, profile, outcome, cost) = min(
            (
                (candidate_index, (*candidate, int(outcome), _singleton_stability_cost(
                    adjusted,
                    message_sizes,
                    scr,
                    candidate[2],
                    int(outcome),
                    [int(theta)],
                )))
                for candidate_index, candidate in enumerate(candidates)
                for outcome in target_outcomes
            ),
            key=lambda item: (item[1][4], item[1][0], item[1][1]),
        )
        if cost <= 2.0:
            _stabilize_target_profile(
                adjusted,
                message_sizes,
                scr,
                masked_logits,
                max_messages,
                int(theta),
                profile,
                int(outcome),
            )
            candidates.pop(best_index)

    for theta in np.flatnonzero(singleton_states):
        theta = int(theta)
        outcome = int(np.argmax(scr.target_mask[theta]))
        for profile in itertools.product(*(range(size) for size in message_sizes)):
            if int(adjusted[profile]) == outcome:
                continue
            if _is_profile_stable(adjusted, message_sizes, scr, theta, profile):
                adjusted[profile] = outcome
    binary_states = np.flatnonzero((scr.target_mask.sum(axis=1) >= 2) & (scr.target_mask.sum(axis=1) <= 3))
    for theta in binary_states:
        theta = int(theta)
        target_outcomes = np.flatnonzero(scr.target_mask[theta]).astype(int).tolist()
        for profile in itertools.product(*(range(size) for size in message_sizes)):
            if int(adjusted[profile]) in target_outcomes:
                continue
            if not _is_profile_stable(adjusted, message_sizes, scr, theta, profile):
                continue
            best_outcome = min(
                target_outcomes,
                key=lambda outcome: _singleton_stability_cost(
                    adjusted,
                    message_sizes,
                    scr,
                    profile,
                    int(outcome),
                    [theta],
                ),
            )
            adjusted[profile] = int(best_outcome)
    for theta in np.flatnonzero(singleton_states):
        theta = int(theta)
        outcome = int(np.argmax(scr.target_mask[theta]))
        for profile in itertools.product(*(range(size) for size in message_sizes)):
            if int(adjusted[profile]) == outcome:
                continue
            if _is_profile_stable(adjusted, message_sizes, scr, theta, profile):
                adjusted[profile] = outcome
    for theta in binary_states:
        theta = int(theta)
        target_outcomes = np.flatnonzero(scr.target_mask[theta]).astype(int).tolist()
        for profile in itertools.product(*(range(size) for size in message_sizes)):
            if int(adjusted[profile]) in target_outcomes:
                continue
            if not _is_profile_stable(adjusted, message_sizes, scr, theta, profile):
                continue
            best_outcome = min(
                target_outcomes,
                key=lambda outcome: _singleton_stability_cost(
                    adjusted,
                    message_sizes,
                    scr,
                    profile,
                    int(outcome),
                    [theta],
                ),
            )
            adjusted[profile] = int(best_outcome)
    for theta in np.flatnonzero(singleton_states):
        theta = int(theta)
        outcome = int(np.argmax(scr.target_mask[theta]))
        for profile in itertools.product(*(range(size) for size in message_sizes)):
            if int(adjusted[profile]) == outcome:
                continue
            if _is_profile_stable(adjusted, message_sizes, scr, theta, profile):
                adjusted[profile] = outcome
    for theta in binary_states:
        theta = int(theta)
        target_outcomes = np.flatnonzero(scr.target_mask[theta]).astype(int).tolist()
        for profile in itertools.product(*(range(size) for size in message_sizes)):
            if int(adjusted[profile]) in target_outcomes:
                continue
            if not _is_profile_stable(adjusted, message_sizes, scr, theta, profile):
                continue
            best_outcome = min(
                target_outcomes,
                key=lambda outcome: _singleton_stability_cost(
                    adjusted,
                    message_sizes,
                    scr,
                    profile,
                    int(outcome),
                    [theta],
                ),
            )
            adjusted[profile] = int(best_outcome)
    return adjusted


def _stabilize_target_profile(
    outcome_table: np.ndarray,
    message_sizes: tuple[int, ...],
    scr: SocialChoiceRule,
    masked_logits: torch.Tensor,
    max_messages: int,
    theta: int,
    profile: tuple[int, ...],
    outcome: int,
) -> None:
    outcome_table[profile] = outcome
    for agent in range(scr.n_agents):
        current_utility = scr.utility(theta, agent, outcome)
        for message in range(message_sizes[agent]):
            if message == profile[agent]:
                continue
            deviated = list(profile)
            deviated[agent] = message
            deviated_profile = tuple(deviated)
            deviated_outcome = int(outcome_table[deviated_profile])
            if scr.utility(theta, agent, deviated_outcome) <= current_utility + 1.0e-9:
                continue
            outcome_table[deviated_profile] = _best_safe_deviation_outcome(
                scr,
                masked_logits,
                max_messages,
                theta,
                agent,
                current_utility,
                deviated_profile,
            )


def _best_safe_deviation_outcome(
    scr: SocialChoiceRule,
    masked_logits: torch.Tensor,
    max_messages: int,
    theta: int,
    agent: int,
    current_utility: float,
    profile: tuple[int, ...],
) -> int:
    flat_idx = _flat_profile_index(profile, max_messages)
    safe_outcomes = [
        outcome
        for outcome in range(scr.n_alternatives)
        if scr.utility(theta, agent, outcome) <= current_utility + 1.0e-9
    ]
    return max(
        safe_outcomes,
        key=lambda outcome: float(masked_logits[flat_idx, outcome].detach().cpu().item()),
    )


def _is_profile_stable(
    outcome_table: np.ndarray,
    message_sizes: tuple[int, ...],
    scr: SocialChoiceRule,
    theta: int,
    profile: tuple[int, ...],
) -> bool:
    outcome = int(outcome_table[profile])
    for agent in range(scr.n_agents):
        current_utility = scr.utility(theta, agent, outcome)
        for message in range(message_sizes[agent]):
            if message == profile[agent]:
                continue
            deviated = list(profile)
            deviated[agent] = message
            deviated_outcome = int(outcome_table[tuple(deviated)])
            if scr.utility(theta, agent, deviated_outcome) > current_utility + 1.0e-9:
                return False
    return True


def _singleton_stability_cost(
    outcome_table: np.ndarray,
    message_sizes: tuple[int, ...],
    scr: SocialChoiceRule,
    profile: tuple[int, ...],
    outcome: int,
    target_states: list[int],
) -> float:
    cost = 0.0
    for theta in target_states:
        for agent in range(scr.n_agents):
            current_utility = scr.utility(theta, agent, outcome)
            best_gain = 0.0
            for message in range(message_sizes[agent]):
                if message == profile[agent]:
                    continue
                deviated = list(profile)
                deviated[agent] = message
                deviated_outcome = int(outcome_table[tuple(deviated)])
                gain = scr.utility(theta, agent, deviated_outcome) - current_utility
                if gain > best_gain:
                    best_gain = gain
            if best_gain > cost:
                cost = best_gain
    return cost


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
        masked_logits[:, ~allowed] = masked_logits[:, ~allowed] - 2.5
        target_counts = torch.as_tensor(
            scr.target_mask.sum(axis=0),
            dtype=masked_logits.dtype,
            device=masked_logits.device,
        )
        masked_logits = masked_logits + 0.20 * torch.log(target_counts.clamp_min(1.0))
        utilities = torch.as_tensor(
            scr.utilities,
            dtype=masked_logits.dtype,
            device=masked_logits.device,
        ) / max(1.0, float(scr.n_alternatives - 1))
        target_by_state = torch.as_tensor(
            scr.target_mask,
            dtype=torch.bool,
            device=masked_logits.device,
        )
        non_target = (~target_by_state).to(masked_logits.dtype)
        non_target_mass = non_target.sum(dim=0).clamp_min(1.0)
        bad_attraction = (utilities.mean(dim=1) * non_target).sum(dim=0) / non_target_mass
        masked_logits = masked_logits - 0.03 * bad_attraction
        predicted = torch.argmax(masked_logits, dim=-1).detach().cpu().numpy().astype(np.int64)
        full_table = predicted.reshape(tuple([self.max_messages] * scr.n_agents))
        outcome_table = np.zeros(message_sizes, dtype=np.int64)
        for profile in itertools.product(*(range(size) for size in message_sizes)):
            outcome_table[profile] = full_table[profile]
        outcome_table = _ensure_singleton_target_coverage(
            outcome_table,
            message_sizes,
            scr,
            masked_logits,
            self.max_messages,
        )
        outcome_table, message_sizes = _merge_duplicate_messages(outcome_table, message_sizes)

        return Mechanism(
            domain=scr.domain,
            message_sizes=message_sizes,
            outcome_table=outcome_table,
            template_name="learned_variable_table",
            metadata={"max_messages": self.max_messages},
        )
