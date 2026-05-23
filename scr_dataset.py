"""Fixed SCR generator for mechanism-synthesis autoresearch.

The positive split deliberately contains implementable finite SCR families:
- dictator SCRs: F(theta) is the top alternative of one decisive agent
- constant SCRs: F(theta) is the same alternative at every state

Negative controls are singleton SCRs that are neither constant nor any
agent-top rule on the sampled finite domain.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from envs import Domain, SocialChoiceRule


SPLIT_OFFSETS = {
    "train": 0,
    "holdout": 100_000,
    "negative": 200_000,
}


def _config_seed(config: dict[str, Any]) -> int:
    dataset = config.get("dataset", {})
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    return int(dataset.get("seed", training.get("seed", evaluation.get("seed", 0))))


def _rng_for_split(seed: int, split: str) -> np.random.Generator:
    return np.random.default_rng(seed + SPLIT_OFFSETS.get(split, 300_000))


def _random_ordinal_utilities(rng: np.random.Generator, domain: Domain) -> np.ndarray:
    utilities = np.zeros(
        (domain.n_states, domain.n_agents, domain.n_alternatives),
        dtype=np.float32,
    )
    for theta in range(domain.n_states):
        for agent in range(domain.n_agents):
            ranking = rng.permutation(domain.n_alternatives)
            for rank, alternative in enumerate(ranking):
                utilities[theta, agent, alternative] = domain.n_alternatives - 1 - rank
    return utilities


def _dictator_target(utilities: np.ndarray, domain: Domain, dictator: int) -> np.ndarray:
    target = np.zeros((domain.n_states, domain.n_alternatives), dtype=bool)
    for theta in range(domain.n_states):
        target[theta, int(np.argmax(utilities[theta, dictator]))] = True
    return target


def _constant_target(domain: Domain, alternative: int) -> np.ndarray:
    target = np.zeros((domain.n_states, domain.n_alternatives), dtype=bool)
    target[:, alternative] = True
    return target


def _singleton_target_from_mapping(domain: Domain, mapping: np.ndarray) -> np.ndarray:
    target = np.zeros((domain.n_states, domain.n_alternatives), dtype=bool)
    for theta, alternative in enumerate(mapping.astype(int).tolist()):
        target[theta, alternative] = True
    return target


def _is_constant(target: np.ndarray) -> bool:
    winners = np.argmax(target.astype(int), axis=1)
    return bool(np.all(winners == winners[0]))


def _matches_any_agent_top(target: np.ndarray, utilities: np.ndarray) -> bool:
    winners = np.argmax(target.astype(int), axis=1)
    for agent in range(utilities.shape[1]):
        tops = np.argmax(utilities[:, agent, :], axis=1)
        if np.array_equal(winners, tops):
            return True
    return False


def _make_positive_scr(
    rng: np.random.Generator,
    domain: Domain,
    split: str,
    index: int,
) -> SocialChoiceRule:
    utilities = _random_ordinal_utilities(rng, domain)

    # Keep the initial benchmark learnable but not completely one-template.
    if index % 5 == 0:
        alternative = int(rng.integers(domain.n_alternatives))
        target = _constant_target(domain, alternative)
        label = domain.n_agents + alternative
        kind = "constant"
        metadata = {"constant_alternative": alternative}
    else:
        dictator = int(rng.integers(domain.n_agents))
        target = _dictator_target(utilities, domain, dictator)
        label = dictator
        kind = "dictator"
        metadata = {"dictator": dictator}

    return SocialChoiceRule(
        scr_id=f"{split}_{index:04d}",
        domain=domain,
        utilities=utilities,
        target_mask=target,
        split=split,
        label=label,
        kind=kind,
        metadata=metadata,
    )


def _make_negative_scr(
    rng: np.random.Generator,
    domain: Domain,
    split: str,
    index: int,
) -> SocialChoiceRule:
    utilities = _random_ordinal_utilities(rng, domain)
    for _ in range(10_000):
        if index % 2 == 0:
            agent = int(rng.integers(domain.n_agents))
            mapping = np.argmin(utilities[:, agent, :], axis=1)
        else:
            mapping = rng.integers(domain.n_alternatives, size=domain.n_states)
        target = _singleton_target_from_mapping(domain, mapping)
        if not _is_constant(target) and not _matches_any_agent_top(target, utilities):
            break
    else:
        raise RuntimeError("Could not generate a negative-control SCR.")

    return SocialChoiceRule(
        scr_id=f"{split}_{index:04d}",
        domain=domain,
        utilities=utilities,
        target_mask=target,
        split=split,
        label=-1,
        kind="negative_control",
        metadata={"nonimplementable_control": True},
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
