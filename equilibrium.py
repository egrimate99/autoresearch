"""Fixed exact pure-Nash verifier for generated finite mechanisms."""

from __future__ import annotations

from typing import Any

from envs import Mechanism, SocialChoiceRule


def is_pure_nash(
    mechanism: Mechanism,
    scr: SocialChoiceRule,
    theta: int,
    profile: tuple[int, ...],
    tol: float = 1e-9,
) -> bool:
    for agent in range(scr.n_agents):
        current_outcome = mechanism.outcome(theta, profile)
        current_utility = scr.utility(theta, agent, current_outcome)
        for message in range(mechanism.message_sizes[agent]):
            if message == profile[agent]:
                continue
            deviated = list(profile)
            deviated[agent] = message
            deviated_outcome = mechanism.outcome(theta, tuple(deviated))
            deviated_utility = scr.utility(theta, agent, deviated_outcome)
            if deviated_utility > current_utility + tol:
                return False
    return True


def verify_generated_mechanism(mechanism: Mechanism, scr: SocialChoiceRule) -> dict[str, Any]:
    state_results = []

    for theta in range(scr.n_states):
        target_outcomes = scr.target_outcomes(theta)
        ne_profiles: list[tuple[int, ...]] = []
        good_eq = 0
        bad_eq = 0

        for profile in mechanism.all_message_profiles():
            if is_pure_nash(mechanism, scr, theta, profile):
                outcome = mechanism.outcome(theta, profile)
                ne_profiles.append(profile)
                if outcome in target_outcomes:
                    good_eq += 1
                else:
                    bad_eq += 1

        state_results.append(
            {
                "theta": theta,
                "num_ne": len(ne_profiles),
                "good_eq": good_eq,
                "bad_eq": bad_eq,
                "missing_good_equilibrium": int(good_eq == 0),
                "has_bad_equilibrium": int(bad_eq > 0),
                "ne_profiles": [list(profile) for profile in ne_profiles[:32]],
            }
        )

    bad_equilibrium_error = sum(item["has_bad_equilibrium"] for item in state_results) / scr.n_states
    missing_good_equilibrium_error = (
        sum(item["missing_good_equilibrium"] for item in state_results) / scr.n_states
    )

    return {
        "scr_id": scr.scr_id,
        "kind": scr.kind,
        "template_name": mechanism.template_name,
        "verified": bool(bad_equilibrium_error == 0.0 and missing_good_equilibrium_error == 0.0),
        "bad_equilibrium_error": bad_equilibrium_error,
        "missing_good_equilibrium_error": missing_good_equilibrium_error,
        "avg_num_ne": sum(item["num_ne"] for item in state_results) / scr.n_states,
        "mechanism_complexity": mechanism.complexity,
        "states": state_results,
    }
