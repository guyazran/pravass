from __future__ import annotations

import math
import random
import time
from typing import Sequence

from .search import SearchResult, SearchStats, ranked_states
from .states import (
    State,
    StateSet,
    all_states,
    conditional_at_most_one_normalizer,
    is_valid_state,
    log_unconstrained_probability,
    unconstrained_probability,
)

THETA_GRID = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99)
MASS_EPSILON = 1e-12


def _valid_normalizer(normalizer: float) -> bool:
    return math.isfinite(normalizer) and normalizer > 0.0


def _error_state_set() -> StateSet:
    return StateSet((), 0.0, 0.0, 0, status="error")


def _state_from_assignment_index(index: int, feature_count: int) -> State:
    return tuple(
        bool(index & (1 << (feature_count - 1 - pos)))
        for pos in range(feature_count)
    )


def mbs_first(
    probabilities: Sequence[float],
    selected_groups: Sequence[Sequence[int]],
    normalizer: float,
    *,
    timeout_seconds: float | None = None,
    exactly_one: bool = False,
) -> StateSet:
    """Return PRaVaSS's first valid state: the most probable valid state.

    This has the same reduced-state selection as ``valid_map``; callers can use
    method-specific metadata to interpret omitted/full-feature uncertainty
    differently downstream.
    """
    if not _valid_normalizer(normalizer):
        return _error_state_set()
    result = ranked_states(
        probabilities,
        selected_groups,
        normalizer,
        max_selected_states=1,
        timeout_seconds=timeout_seconds,
        exactly_one=exactly_one,
    )
    return result.state_set


def valid_map(
    probabilities: Sequence[float],
    selected_groups: Sequence[Sequence[int]],
    normalizer: float,
    *,
    timeout_seconds: float | None = None,
    exactly_one: bool = False,
) -> StateSet:
    return mbs_first(
        probabilities,
        selected_groups,
        normalizer,
        timeout_seconds=timeout_seconds,
        exactly_one=exactly_one,
    )


def map_baseline(probabilities: Sequence[float]) -> StateSet:
    """The most likely state of the unconstrained belief (a feature is true iff q > 0.5)."""
    state = tuple(float(q) > 0.5 for q in probabilities)
    p = unconstrained_probability(state, probabilities)
    return StateSet(
        states=(state,),
        unconstrained_mass=p,
        constrained_mass=p,
        total_states_considered=1,
        status="ok",
    )


def committed_literals(probabilities: Sequence[float], tau: float) -> dict[int, bool]:
    commits: dict[int, bool] = {}
    for idx, q in enumerate(probabilities):
        if q >= tau:
            commits[idx] = True
        elif q <= 1.0 - tau:
            commits[idx] = False
    return commits


def fluent_abstention_unconstrained(
    probabilities: Sequence[float],
    tau: float,
    normalizer: float,
    *,
    selected_groups: Sequence[Sequence[int]] = (),
    timeout_seconds: float | None = None,  # accepted for API parity; computation is O(n)
) -> StateSet:
    """Unconstrained fluent abstention.

    Pins τ-committed literals; uncommitted positions are wildcards. The result
    represents *all* 2^k completions of the committed template implicitly via
    ``StateSet.committed_template`` — nothing is enumerated. Mass is reported
    over the *valid* (constraint-satisfying) members of those completions: the
    unconstrained mass is Σ_{valid s consistent with commits} β̃(s) and the
    constrained mass divides that by ``normalizer``. The output set itself is
    still all 2^k completions — the method is constraint-blind in construction
    — but evaluating its mass against the task belief excludes invalid states,
    matching MBSθ's θ-mass semantics so that constrained mass stays in [0, 1].
    """
    del timeout_seconds  # this branch never iterates, so a timeout has nothing to enforce
    if not _valid_normalizer(normalizer):
        return _error_state_set()
    n = len(probabilities)
    commits = committed_literals(probabilities, tau)
    template: tuple[bool | None, ...] = tuple(
        commits.get(i) for i in range(n)
    )
    unconstrained_mass = conditional_at_most_one_normalizer(
        probabilities, selected_groups, commits
    )
    return StateSet(
        states=(),
        unconstrained_mass=unconstrained_mass,
        constrained_mass=unconstrained_mass / normalizer,
        total_states_considered=0,
        status="ok",
        committed_template=template,
    )


def random_theta_confident(
    probabilities: Sequence[float],
    selected_groups: Sequence[Sequence[int]],
    theta: float,
    normalizer: float,
    *,
    seed: int,
    timeout_seconds: float | None = None,
) -> StateSet:
    """Return a deterministic pseudo-random valid-state prefix.

    The affine permutation scans assignment indices without materializing all
    states; filtering that bijection visits each valid state once in an induced
    pseudo-random order. ``total_states_considered`` reports assignment indices
    visited by this algorithm, not the total number of valid states.
    """
    if not _valid_normalizer(normalizer):
        return _error_state_set()
    if theta <= 0.0:
        return StateSet((), 0.0, 0.0, 0, status="ok")
    start = time.perf_counter()
    rng = random.Random(seed)
    feature_count = len(probabilities)
    total_assignments = 1 << feature_count
    if total_assignments == 1:
        multiplier = 0
        offset = 0
    else:
        multiplier = rng.randrange(1, total_assignments, 2)
        offset = rng.randrange(total_assignments)
    selected: list[State] = []
    unconstrained_mass = 0.0
    constrained_mass = 0.0
    assignments_considered = 0
    for step in range(total_assignments):
        if timeout_seconds is not None and time.perf_counter() - start >= timeout_seconds:
            return StateSet(
                states=tuple(selected),
                unconstrained_mass=unconstrained_mass,
                constrained_mass=constrained_mass,
                total_states_considered=assignments_considered,
                status="timeout",
            )
        assignments_considered = step + 1
        index = (multiplier * step + offset) % total_assignments
        state = _state_from_assignment_index(index, feature_count)
        if not is_valid_state(state, selected_groups):
            continue
        p = unconstrained_probability(state, probabilities)
        selected.append(state)
        unconstrained_mass += p
        constrained_mass += p / normalizer
        if constrained_mass >= theta:
            break
    return StateSet(
        states=tuple(selected),
        unconstrained_mass=unconstrained_mass,
        constrained_mass=constrained_mass,
        total_states_considered=assignments_considered,
        status="ok",
    )


def mbs_prevas(
    probabilities: Sequence[float],
    selected_groups: Sequence[Sequence[int]],
    theta: float,
    normalizer: float,
    timeout_seconds: float | None = None,
    exactly_one: bool = False,
    exact: bool = True,
) -> SearchResult:
    """MBSθ by PRaVaSS (the pravass package).

    ``normalizer`` is ``Z`` for ``selected_groups``; ``exact=False`` marks it as
    an upper bound (a relaxed normalizer), which PRaVaSS tightens as it searches.
    """
    if not _valid_normalizer(normalizer):
        return SearchResult(
            state_set=_error_state_set(),
            stats=SearchStats(
                states_popped=0,
                states_scored=0,
                valid_states_selected=0,
                invalid_states_skipped=0,
                cumulative_mass=0.0,
                runtime_seconds=0.0,
                timeout=False,
            ),
        )
    return ranked_states(
        probabilities,
        selected_groups,
        normalizer,
        theta=theta,
        exact=exact,
        timeout_seconds=timeout_seconds,
        exactly_one=exactly_one,
    )


def unconstrained_top_mass_baseline(
    probabilities: Sequence[float],
    theta: float,
    *,
    timeout_seconds: float | None = None,
) -> StateSet:
    """MBSθ of the unconstrained belief, by PRaVaSS without constraints."""
    return ranked_states(probabilities, (), 1.0, theta=theta, timeout_seconds=timeout_seconds).state_set


def exhaustive_mbs(
    probabilities: Sequence[float],
    selected_groups: Sequence[Sequence[int]],
    theta: float,
    normalizer: float,
    *,
    timeout_seconds: float | None = None,
    exactly_one: bool = False,
) -> StateSet:
    """MBSθ by enumerating and sorting every state (the Experiment 2 reference)."""
    if not math.isfinite(theta) or theta < 0.0 or theta > 1.0:
        raise ValueError(f"theta outside [0, 1]: {theta}")
    if not _valid_normalizer(normalizer):
        raise ValueError(f"normalizer must be positive, got {normalizer}")
    if theta <= 0.0:
        return StateSet(
            states=(),
            unconstrained_mass=0.0,
            constrained_mass=0.0,
            total_states_considered=0,
            status="ok",
        )

    start = time.perf_counter()
    scored: list[tuple[float, State, float]] = []
    considered = 0
    for state in all_states(len(probabilities)):
        if timeout_seconds is not None and time.perf_counter() - start >= timeout_seconds:
            return StateSet(
                states=(),
                unconstrained_mass=0.0,
                constrained_mass=0.0,
                total_states_considered=considered,
                status="timeout",
            )
        considered += 1
        if not is_valid_state(state, selected_groups, exactly_one=exactly_one):
            continue
        logp = log_unconstrained_probability(state, probabilities)
        p = unconstrained_probability(state, probabilities)
        scored.append((-round(logp, 12), state, p))
    scored.sort(key=lambda item: (item[0], item[1]))
    selected: list[State] = []
    unconstrained_mass = 0.0
    constrained_mass = 0.0
    for _priority, state, p in scored:
        selected.append(state)
        unconstrained_mass += p
        constrained_mass += p / normalizer
        if constrained_mass + MASS_EPSILON >= theta:
            break
    return StateSet(
        states=tuple(selected),
        unconstrained_mass=unconstrained_mass,
        constrained_mass=constrained_mass,
        total_states_considered=len(scored),
        status="ok",
    )


def fluent_prediction_marginals(probabilities: Sequence[float]) -> tuple[float, ...]:
    """Sorted unique fluent uncertainty marginals min(q, 1-q)."""
    marginals: set[float] = set()
    for q_raw in probabilities:
        q = float(q_raw)
        if not math.isfinite(q) or q < 0.0 or q > 1.0:
            raise ValueError(f"Probability outside [0, 1]: {q}")
        marginals.add(1.0 - q if q > 0.5 else q)
    return tuple(sorted(marginals))


def adaptive_fluent_abstention_tau_candidates(
    marginals: Sequence[float],
) -> tuple[float, ...]:
    """Tau candidates ordered from smallest to largest abstention belief set.

    Fluent abstention commits a fluent when its confidence max(q, 1-q) is at
    least τ.  The committed set only changes when τ crosses one of those
    confidence values, so midpoints between adjacent confidences represent the
    strict in-between cases. Candidates are clamped to [0.5, 1.0] and de-duped.
    """
    if not marginals:
        return (0.5,)
    confidences = tuple(sorted({1.0 - marginal for marginal in marginals}))
    raw_candidates: list[float] = [confidences[0]]
    for lower, upper in zip(confidences, confidences[1:]):
        raw_candidates.append((lower + upper) / 2.0)
    raw_candidates.append(1.0)

    candidates: list[float] = []
    for tau in sorted(raw_candidates):
        tau = min(1.0, max(0.5, float(tau)))
        if not candidates or tau != candidates[-1]:
            candidates.append(tau)
    return tuple(candidates)


def per_sample_tau_star(
    probabilities: Sequence[float],
    selected_groups: Sequence[Sequence[int]],
    normalizer: float,
    theta: float,
) -> float | None:
    """Smallest τ candidate whose fluent-abstention constrained mass ≥ θ.

    Iterates τ candidates from lowest (most commits, smallest set, lowest mass)
    upward and returns the first one whose unconstrained-variant constrained
    mass meets θ. Returns ``None`` if no candidate qualifies (only possible at
    pathological θ, since τ → 1 yields constrained mass → 1).
    """
    if not _valid_normalizer(normalizer):
        return None
    marginals = fluent_prediction_marginals(probabilities)
    candidates = adaptive_fluent_abstention_tau_candidates(marginals)
    for tau in candidates:
        result = fluent_abstention_unconstrained(
            probabilities,
            tau,
            normalizer,
            selected_groups=selected_groups,
        )
        if result.status != "ok":
            continue
        if result.constrained_mass + 1e-12 >= theta:
            return tau
    return None
