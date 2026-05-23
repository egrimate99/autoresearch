"""Fixed finite SCR and mechanism data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


@dataclass(frozen=True)
class Domain:
    n_agents: int
    n_states: int
    n_alternatives: int
    preference_type: str = "ordinal"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Domain":
        domain = config.get("domain", {})
        return cls(
            n_agents=int(domain.get("n_agents", 3)),
            n_states=int(domain.get("n_states", 4)),
            n_alternatives=int(domain.get("n_alternatives", 4)),
            preference_type=str(domain.get("preference_type", "ordinal")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SocialChoiceRule:
    """A finite social choice rule/correspondence.

    utilities has shape [state, agent, alternative].
    target_mask has shape [state, alternative] and may contain multiple
    acceptable outcomes per state.
    """

    scr_id: str
    domain: Domain
    utilities: np.ndarray
    target_mask: np.ndarray
    split: str
    label: int = -1
    kind: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    teacher_outcome_table: np.ndarray | None = None

    @property
    def n_agents(self) -> int:
        return self.domain.n_agents

    @property
    def n_states(self) -> int:
        return self.domain.n_states

    @property
    def n_alternatives(self) -> int:
        return self.domain.n_alternatives

    def target_outcomes(self, theta: int) -> set[int]:
        return set(np.flatnonzero(self.target_mask[theta]).astype(int).tolist())

    def utility(self, theta: int, agent: int, alternative: int) -> float:
        return float(self.utilities[theta, agent, alternative])

    def top_alternative(self, theta: int, agent: int) -> int:
        return int(np.argmax(self.utilities[theta, agent]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scr_id": self.scr_id,
            "domain": self.domain.to_dict(),
            "utilities": self.utilities.astype(float).tolist(),
            "target_mask": self.target_mask.astype(int).tolist(),
            "split": self.split,
            "label": int(self.label),
            "kind": self.kind,
            "metadata": self.metadata,
            "has_teacher_outcome_table": self.teacher_outcome_table is not None,
        }


@dataclass
class Mechanism:
    """Finite-message mechanism/game form with state-independent outcomes."""

    domain: Domain
    message_sizes: tuple[int, ...]
    outcome_table: np.ndarray
    template_name: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def all_message_profiles(self) -> Iterable[tuple[int, ...]]:
        return product(*(range(size) for size in self.message_sizes))

    def outcome(self, theta: int | tuple[int, ...], profile: tuple[int, ...] | None = None) -> int:
        # theta is accepted for verifier ergonomics but intentionally ignored:
        # a mechanism outcome function cannot observe the true state.
        if profile is None:
            profile = theta  # type: ignore[assignment]
        return int(self.outcome_table[tuple(profile)])

    @property
    def num_message_profiles(self) -> int:
        total = 1
        for size in self.message_sizes:
            total *= size
        return total

    @property
    def complexity(self) -> float:
        unique_outcomes = len(set(self.outcome_table.reshape(-1).astype(int).tolist()))
        return float(self.num_message_profiles + unique_outcomes)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping.")
    return data


def dump_jsonable_array(value: np.ndarray) -> list[Any]:
    return value.tolist()
