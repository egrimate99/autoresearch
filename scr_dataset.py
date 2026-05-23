"""Fixed hard SCR generator for learned mechanism synthesis.

Positive SCRs are sampled from a broad family of singleton social choice
functions. At every state, the chosen alternative is made the strict unanimous
top alternative, so a finite report/challenge mechanism can implement it.

The training label is not a template id. It is the full canonical mechanism
outcome table on a fixed finite message scaffold:

- message 0..S-1: report a state
- message S..2S-1: challenge a state

The learner must map SCR tensors to this outcome table. Verification never
uses the teacher table; it only enumerates pure Nash equilibria of the
checkpointed synthesizer's generated mechanism.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from envs import Domain, SocialChoiceRule


SPLIT_OFFSETS = {
    "train": 0,
    "holdout": 100_000,
    "negative": 200_000,
}


def message_size(domain: Domain) -> int:
    return 2 * domain.n_states


def message_sizes(domain: Domain) -> tuple[int, ...]:
    return tuple([message_size(domain)] * domain.n_agents)


def _config_seed(config: dict[str, Any]) -> int:
    dataset = config.get("dataset", {})
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    return int(dataset.get("seed", training.get("seed", evaluation.get("seed", 0))))


def _rng_for_split(seed: int, split: str) -> np.random.Generator:
    return np.random.default_rng(seed + SPLIT_OFFSETS.get(split, 300_000))


def _target_from_mapping(domain: Domain, mapping: np.ndarray) -> np.ndarray:
    target = np.zeros((domain.n_states, domain.n_alternatives), dtype=bool)
    for theta, alternative in enumerate(mapping.astype(int).tolist()):
        target[theta, alternative] = True
    return target


def _random_target_mapping(rng: np.random.Generator, domain: Domain) -> np.ndarray:
    # A latent nonlinear rule creates structured but varied state-to-outcome
    # maps. Holdout uses different seeds, so memorizing examples does not work.
    state_codes = rng.normal(size=(domain.n_states, 4))
    alt_codes = rng.normal(size=(domain.n_alternatives, 4))
    weights = rng.normal(size=(4,))
    pairwise = state_codes @ np.diag(weights) @ alt_codes.T
    pairwise += 0.35 * rng.normal(size=(domain.n_states, domain.n_alternatives))
    mapping = np.argmax(pairwise, axis=1)

    # Avoid degenerate all-constant maps unless the sample naturally needs one.
    if np.all(mapping == mapping[0]):
        mapping[int(rng.integers(domain.n_states))] = int(rng.integers(domain.n_alternatives))
    return mapping.astype(np.int64)


def _utilities_with_implementable_targets(
    rng: np.random.Generator,
    domain: Domain,
    mapping: np.ndarray,
) -> np.ndarray:
    selected = sorted(set(mapping.astype(int).tolist()))
    utilities = np.zeros(
        (domain.n_states, domain.n_agents, domain.n_alternatives),
        dtype=np.float32,
    )
    for theta in range(domain.n_states):
        target = int(mapping[theta])
        for agent in range(domain.n_agents):
            other_selected = [a for a in selected if a != target]
            nonselected = [a for a in range(domain.n_alternatives) if a not in selected]
            rng.shuffle(other_selected)
            rng.shuffle(nonselected)

            # Private distractors can outrank the target. The target only has
            # to dominate alternatives that the canonical mechanism can reach.
            split = int(rng.integers(0, len(nonselected) + 1)) if nonselected else 0
            high_distractors = nonselected[:split]
            low_distractors = nonselected[split:]
            ranking = high_distractors + [target] + other_selected + low_distractors
            for rank, alternative in enumerate(ranking):
                utilities[theta, agent, alternative] = domain.n_alternatives - 1 - rank
    return utilities


def _negative_utilities_and_target(
    rng: np.random.Generator,
    domain: Domain,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    honest_mapping = _random_target_mapping(rng, domain)
    utilities = _utilities_with_implementable_targets(rng, domain, honest_mapping)
    target_mapping = np.roll(honest_mapping, 1)

    if np.array_equal(target_mapping, honest_mapping):
        target_mapping = honest_mapping.copy()
        theta = int(rng.integers(domain.n_states))
        choices = [a for a in range(domain.n_alternatives) if a != int(honest_mapping[theta])]
        target_mapping[theta] = int(rng.choice(choices))
    return utilities, _target_from_mapping(domain, target_mapping), target_mapping


def canonical_outcome_table(domain: Domain, target_mapping: np.ndarray) -> np.ndarray:
    """Teacher table for the fixed report/challenge scaffold.

    If any player sends a challenge, the lowest-index challenger controls the
    challenged state. Otherwise unanimous state reports are honored. Nonunanimous
    report profiles fall back to player 0's reported state.
    """

    sizes = message_sizes(domain)
    table = np.zeros(sizes, dtype=np.int64)
    state_count = domain.n_states

    for profile in product(*(range(size) for size in sizes)):
        chosen_state = None
        for agent_message in profile:
            if agent_message >= state_count:
                chosen_state = agent_message - state_count
                break

        if chosen_state is None:
            if all(message == profile[0] for message in profile):
                chosen_state = profile[0]
            else:
                chosen_state = profile[0]

        table[profile] = int(target_mapping[int(chosen_state)])
    return table


def _make_positive_scr(
    rng: np.random.Generator,
    domain: Domain,
    split: str,
    index: int,
) -> SocialChoiceRule:
    mapping = _random_target_mapping(rng, domain)
    utilities = _utilities_with_implementable_targets(rng, domain, mapping)
    target = _target_from_mapping(domain, mapping)
    teacher = canonical_outcome_table(domain, mapping)

    return SocialChoiceRule(
        scr_id=f"{split}_{index:05d}",
        domain=domain,
        utilities=utilities,
        target_mask=target,
        split=split,
        label=-1,
        kind="selected_top_challenge",
        metadata={
            "target_mapping": mapping.astype(int).tolist(),
            "message_scaffold": "report_or_challenge_state",
        },
        teacher_outcome_table=teacher,
    )


def _make_negative_scr(
    rng: np.random.Generator,
    domain: Domain,
    split: str,
    index: int,
) -> SocialChoiceRule:
    utilities, target, mapping = _negative_utilities_and_target(rng, domain)

    return SocialChoiceRule(
        scr_id=f"{split}_{index:05d}",
        domain=domain,
        utilities=utilities,
        target_mask=target,
        split=split,
        label=-1,
        kind="negative_non_unanimous_target",
        metadata={
            "target_mapping": mapping.astype(int).tolist(),
            "nonimplementable_control": True,
            "message_scaffold": "report_or_challenge_state",
        },
        teacher_outcome_table=None,
    )


def make_dataset(config: dict[str, Any]) -> list[SocialChoiceRule]:
    dataset_cfg = config.get("dataset", {})
    split = str(dataset_cfg.get("split", "train"))
    num_scrs = int(dataset_cfg.get("num_scrs", 128))
    only_nonimplementable = bool(dataset_cfg.get("only_nonimplementable", False))
    domain = Domain.from_config(config)
    rng = _rng_for_split(_config_seed(config), split)

    if only_nonimplementable or split == "negative":
        return [_make_negative_scr(rng, domain, split, i) for i in range(num_scrs)]
    return [_make_positive_scr(rng, domain, split, i) for i in range(num_scrs)]
