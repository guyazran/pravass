"""Pure protocol helpers for the closed-loop experiment (Experiment 6).

This module is independent of the Experiment 1--3 runners.  It contains the
common constrained state ordering, fixed-mass prefixes, the AMS prefix search,
and the paired bootstrap used by Experiment 6.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from .planning import PlanningResult
from .search import ranked_states
from .states import State, unconstrained_probability

PROB_EPS = 1e-6


@dataclass(frozen=True)
class OrderedBelief:
    states: tuple[State, ...]
    state_unconstrained_masses: tuple[float, ...]
    cumulative_constrained_masses: tuple[float, ...]
    normalizer: float
    status: str
    target_mass: float
    states_popped: int
    states_scored: int
    invalid_states_skipped: int
    runtime_seconds: float


def build_common_state_ordering(
    probabilities: Sequence[float],
    constraint_groups: Sequence[Sequence[int]],
    normalizer: float,
    *,
    max_belief_states: int = 16,
    target_mass: float = 0.99,
    timeout_seconds: float | None = 30.0,
) -> OrderedBelief:
    """Generate complete valid states in non-increasing task-belief mass."""
    if max_belief_states <= 0:
        raise ValueError("max_belief_states must be positive")
    result = ranked_states(
        tuple(float(q) for q in probabilities),
        tuple(tuple(group) for group in constraint_groups),
        float(normalizer),
        theta=float(target_mass),
        timeout_seconds=timeout_seconds,
        max_selected_states=max_belief_states,
    )
    states = tuple(result.state_set.states)
    masses = tuple(
        unconstrained_probability(state, probabilities) for state in states
    )
    cumulative: list[float] = []
    running = 0.0
    for mass in masses:
        running += mass / normalizer
        cumulative.append(running)

    status = result.state_set.status
    if (
        status == "ok"
        and len(states) >= max_belief_states
        and (not cumulative or cumulative[-1] + 1e-12 < target_mass)
    ):
        status = "capped"
    return OrderedBelief(
        states=states,
        state_unconstrained_masses=masses,
        cumulative_constrained_masses=tuple(cumulative),
        normalizer=float(normalizer),
        status=status,
        target_mass=float(target_mass),
        states_popped=result.stats.states_popped,
        states_scored=result.stats.states_scored,
        invalid_states_skipped=result.stats.invalid_states_skipped,
        runtime_seconds=result.stats.runtime_seconds,
    )


@dataclass(frozen=True)
class FixedMassPrefix:
    theta: float
    states: tuple[State, ...]
    constrained_mass: float
    status: str

    @property
    def size(self) -> int:
        return len(self.states)


def fixed_mass_prefix(
    ordering: OrderedBelief,
    theta: float,
) -> FixedMassPrefix:
    """Return the smallest available common prefix reaching ``theta``."""
    theta = float(theta)
    if not 0.0 <= theta <= 1.0:
        raise ValueError(f"theta outside [0, 1]: {theta}")
    size = 0
    mass = 0.0
    for size, mass in enumerate(ordering.cumulative_constrained_masses, start=1):
        if mass + 1e-12 >= theta:
            return FixedMassPrefix(theta, ordering.states[:size], mass, "ok")
    if ordering.states:
        size = len(ordering.states)
        mass = ordering.cumulative_constrained_masses[-1]
    return FixedMassPrefix(theta, ordering.states[:size], mass, ordering.status)


def constraint_valid_map(ordering: OrderedBelief) -> State:
    if not ordering.states:
        raise ValueError("cannot extract MAP from an empty state ordering")
    return ordering.states[0]


@dataclass(frozen=True)
class AMSResult:
    planning_result: PlanningResult
    prefix_size: int
    attempts: tuple[tuple[int, PlanningResult], ...]


def planning_result_found(result: PlanningResult) -> bool:
    return result.status == "ok"


def search_ams_prefixes(
    states: Sequence[State],
    plan_prefix: Callable[[tuple[State, ...]], PlanningResult],
    *,
    singleton_result: PlanningResult | None = None,
) -> AMSResult:
    """Search the largest established-plannable prefix with MAP fallback.

    ``singleton_result`` is the exact object already used for MAP.  It is never
    overwritten, even if every larger planner call times out or crashes.
    """
    ordered = tuple(states)
    if not ordered:
        raise ValueError("AMS requires at least one ordered state")
    cached_singleton = (
        singleton_result if singleton_result is not None else plan_prefix(ordered[:1])
    )
    attempts: dict[int, PlanningResult] = {1: cached_singleton}
    if not planning_result_found(cached_singleton) or len(ordered) == 1:
        return AMSResult(cached_singleton, 1, tuple(attempts.items()))

    best_size = 1
    best_result = cached_singleton
    full_size = len(ordered)
    full_result = plan_prefix(ordered)
    attempts[full_size] = full_result
    if planning_result_found(full_result):
        return AMSResult(full_result, full_size, tuple(sorted(attempts.items())))

    lo, hi = 1, full_size
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        result = plan_prefix(ordered[:mid])
        attempts[mid] = result
        if planning_result_found(result):
            lo = mid
            best_size = mid
            best_result = result
        else:
            hi = mid
    return AMSResult(best_result, best_size, tuple(sorted(attempts.items())))


def planning_result_dict(result: PlanningResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "planner_status": result.planner_status,
        "runtime_seconds": float(result.runtime_seconds),
        "plan_found": planning_result_found(result),
        "success": bool(result.success),
        "plan_length": result.plan_length,
        "actions": list(result.actions),
        "unsafe_first": bool(result.unsafe_first),
        "precondition_failure": bool(result.pre_fail),
        "no_plan_fn": bool(result.no_plan_fn),
        "error_message": result.error_message,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_mean_ci(
    values_by_sample: Mapping[str, float],
    *,
    seed: int = 0,
    repetitions: int = 1000,
) -> dict[str, float | int | None]:
    """Bootstrap sample IDs, preserving all within-sample pairing."""
    sample_ids = tuple(sorted(values_by_sample))
    if not sample_ids:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    values = tuple(float(values_by_sample[sample_id]) for sample_id in sample_ids)
    rng = random.Random(seed)
    draws = [
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(max(1, repetitions))
    ]
    return {
        "n": len(values),
        "mean": mean(values),
        "ci_low": _percentile(draws, 0.025),
        "ci_high": _percentile(draws, 0.975),
    }


