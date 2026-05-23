"""Fixed generator for minimal finite-mechanism synthesis.

Positive SCRs are generated from varied small latent mechanisms, not from a
state-report construction. For each sampled latent mechanism H and each state
theta, we sample ordinal utilities and define F(theta) to be exactly the set of
outcomes attained by pure Nash equilibria of H at theta. H therefore implements
F by construction.

The benchmark target is not "recover H exactly". The verifier accepts any
generated mechanism that implements F. The aggregate score then rewards smaller
mechanisms by comparing generated complexity to H's quotient complexity.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from envs import Domain, Mechanism, SocialChoiceRule
from equilibrium import verify_generated_mechanism


SPLIT_OFFSETS = {
    "train": 0,
    "holdout": 100_000,
    "negative": 200_000,
}


def max_message_size(config: dict[str, Any]) -> int:
    mechanism = config.get("mechanism", {})
    return int(mechanism.get("max_messages_per_agent", 3))


def max_message_profiles(domain: Domain, max_messages: int) -> int:
    return max_messages ** domain.n_agents


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


def _mechanism_complexity(message_sizes: tuple[int, ...], table: np.ndarray) -> float:
    profiles = int(np.prod(np.array(message_sizes, dtype=np.int64)))
    outcomes = len(set(table.reshape(-1).astype(int).tolist()))
    return float(profiles + outcomes)


def _sample_latent_mechanism(
    rng: np.random.Generator,
    domain: Domain,
    max_messages: int,
) -> Mechanism:
    message_sizes = tuple(
        int(rng.integers(2, max_messages + 1))
        for _ in range(domain.n_agents)
    )
    table = rng.integers(domain.n_alternatives, size=message_sizes, dtype=np.int64)

    # Make degenerate constant mechanisms rare; they make the minimality task
    # uninteresting.
    if len(set(table.reshape(-1).astype(int).tolist())) < 2:
        table[(0,) * domain.n_agents] = int((table[(0,) * domain.n_agents] + 1) % domain.n_alternatives)

    return Mechanism(
        domain=domain,
        message_sizes=message_sizes,
        outcome_table=table,
        template_name="latent_random_game_form",
    )


def _target_from_ne_outcomes(domain: Domain, mechanism: Mechanism, utilities: np.ndarray) -> np.ndarray | None:
    target = np.zeros((domain.n_states, domain.n_alternatives), dtype=bool)
    probe_scr = SocialChoiceRule(
        scr_id="probe",
        domain=domain,
        utilities=utilities,
        target_mask=np.ones((domain.n_states, domain.n_alternatives), dtype=bool),
        split="probe",
    )

    for theta in range(domain.n_states):
        outcomes = set()
        for profile in mechanism.all_message_profiles():
            if _is_pure_nash(mechanism, probe_scr, theta, profile):
                outcomes.add(mechanism.outcome(theta, profile))
        if not outcomes:
            return None
        for outcome in outcomes:
            target[theta, outcome] = True
    return target


def _is_pure_nash(
    mechanism: Mechanism,
    scr: SocialChoiceRule,
    theta: int,
    profile: tuple[int, ...],
) -> bool:
    for agent in range(scr.n_agents):
        current = mechanism.outcome(theta, profile)
        current_utility = scr.utility(theta, agent, current)
        for message in range(mechanism.message_sizes[agent]):
            if message == profile[agent]:
                continue
            deviated = list(profile)
            deviated[agent] = message
            deviated_outcome = mechanism.outcome(theta, tuple(deviated))
            if scr.utility(theta, agent, deviated_outcome) > current_utility + 1e-9:
                return False
    return True


def _positive_candidate(
    rng: np.random.Generator,
    domain: Domain,
    max_messages: int,
) -> tuple[Mechanism, np.ndarray, np.ndarray] | None:
    mechanism = _sample_latent_mechanism(rng, domain, max_messages)
    utilities = _random_ordinal_utilities(rng, domain)
    target = _target_from_ne_outcomes(domain, mechanism, utilities)
    if target is None:
        return None

    target_sizes = target.sum(axis=1)
    if np.any(target_sizes == 0):
        return None
    if np.mean(target_sizes) > max(1.0, domain.n_alternatives / 2):
        return None
    return mechanism, utilities, target


def _pad_teacher_table(table: np.ndarray, message_sizes: tuple[int, ...], max_messages: int) -> np.ndarray:
    padded = np.zeros(tuple([max_messages] * len(message_sizes)), dtype=np.int64)
    slices = tuple(slice(0, size) for size in message_sizes)
    padded[slices] = table
    return padded


def _make_positive_scr(
    rng: np.random.Generator,
    domain: Domain,
    split: str,
    index: int,
    max_messages: int,
) -> SocialChoiceRule:
    for _ in range(10_000):
        candidate = _positive_candidate(rng, domain, max_messages)
        if candidate is None:
            continue
        mechanism, utilities, target = candidate
        oracle_complexity = _mechanism_complexity(mechanism.message_sizes, mechanism.outcome_table)

        scr = SocialChoiceRule(
            scr_id=f"{split}_{index:05d}",
            domain=domain,
            utilities=utilities,
            target_mask=target,
            split=split,
            label=-1,
            kind="latent_mechanism_ne_outcomes",
            metadata={
                "teacher_message_sizes": list(mechanism.message_sizes),
                "oracle_complexity": oracle_complexity,
                "message_scaffold": "variable_latent_game_form",
            },
            teacher_outcome_table=_pad_teacher_table(
                mechanism.outcome_table,
                mechanism.message_sizes,
                max_messages,
            ),
            teacher_message_sizes=mechanism.message_sizes,
        )

        # Guard the generator invariant using the same fixed verifier used at
        # evaluation time.
        if verify_generated_mechanism(mechanism, scr)["verified"]:
            return scr

    raise RuntimeError("Could not generate a positive implementable SCR.")


def _make_negative_scr(
    rng: np.random.Generator,
    domain: Domain,
    split: str,
    index: int,
    max_messages: int,
) -> SocialChoiceRule:
    positive = _make_positive_scr(rng, domain, split, index, max_messages)
    target = positive.target_mask.copy()
    for theta in range(domain.n_states):
        if rng.random() < 0.75:
            choices = [a for a in range(domain.n_alternatives) if not target[theta, a]]
            if choices:
                target[theta, :] = False
                target[theta, int(rng.choice(choices))] = True

    return SocialChoiceRule(
        scr_id=f"{split}_{index:05d}",
        domain=domain,
        utilities=positive.utilities,
        target_mask=target,
        split=split,
        label=-1,
        kind="negative_perturbed_ne_outcomes",
        metadata={
            "nonimplementable_control": True,
            "source_teacher_message_sizes": positive.metadata["teacher_message_sizes"],
            "oracle_complexity": positive.metadata["oracle_complexity"],
        },
        teacher_outcome_table=None,
        teacher_message_sizes=None,
    )


def make_dataset(config: dict[str, Any]) -> list[SocialChoiceRule]:
    dataset_cfg = config.get("dataset", {})
    split = str(dataset_cfg.get("split", "train"))
    num_scrs = int(dataset_cfg.get("num_scrs", 128))
    only_nonimplementable = bool(dataset_cfg.get("only_nonimplementable", False))
    domain = Domain.from_config(config)
    max_messages = max_message_size(config)
    rng = _rng_for_split(_config_seed(config), split)

    if only_nonimplementable or split == "negative":
        return [_make_negative_scr(rng, domain, split, i, max_messages) for i in range(num_scrs)]
    return [_make_positive_scr(rng, domain, split, i, max_messages) for i in range(num_scrs)]
