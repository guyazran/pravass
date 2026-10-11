"""PRaVaSS for the experiments, run by the ``pravass`` package.

The experiments describe a state as a tuple of booleans over indexed features
and a constraint as groups of feature indices (at most one true per group, or
exactly one). This module translates both to the package's types, runs
:func:`pravass.iter_pravass`, and translates the states back. The normalizer
``Z`` is computed by the experiment code (see ``normalizers.py``) and handed to
the package as a :class:`pravass.Normalization`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from pravass import (
    FunctionConstraint,
    InfeasibleConstraintError,
    MutexGroups,
    Normalization,
    Normalizer,
    TaskBelief,
    TimeLimitError,
    UnconstrainedBelief,
    UnitNormalizer,
    iter_pravass,
)

from .states import State, StateSet, unconstrained_probability


@dataclass(frozen=True)
class SearchStats:
    states_popped: int  # states taken from the search frontier, valid or not
    states_scored: int  # states inserted into the search
    valid_states_selected: int
    invalid_states_skipped: int
    cumulative_mass: float
    runtime_seconds: float
    timeout: bool


@dataclass(frozen=True)
class SearchResult:
    state_set: StateSet
    stats: SearchStats


class GivenNormalizer(Normalizer):
    """A normalizer the experiment code computed: exact, or an upper bound on ``Z``."""

    def __init__(self, value: float, exact: bool):
        self._normalization = Normalization(float(value), exact=exact)

    def __call__(self, base, constraint) -> Normalization:
        return self._normalization


def feature_names(count: int) -> tuple[str, ...]:
    """Names for indexed features; their sorted order is the index order."""
    return tuple(f"f{i:05d}" for i in range(count))


def group_constraint(
    names: Sequence[str],
    groups: Sequence[Sequence[int]],
    *,
    exactly_one: bool = False,
):
    """The package constraint for index groups, or ``None`` when there are none."""
    named = [frozenset(names[i] for i in group) for group in groups]
    if exactly_one:
        if not named:
            return None
        return FunctionConstraint(lambda state: all(len(group & state) == 1 for group in named))
    named = [group for group in named if len(group) >= 2]
    return MutexGroups(named) if named else None


def task_belief(
    probabilities: Sequence[float],
    groups: Sequence[Sequence[int]],
    normalizer: "float | Normalizer",
    *,
    exact: bool = True,
    exactly_one: bool = False,
    names: Sequence[str] | None = None,
):
    """The package belief for index-based inputs.

    ``normalizer`` is ``Z`` for ``groups`` (``exact`` says whether it is exact or
    an upper bound), or a package :class:`~pravass.Normalizer`.
    """
    names = tuple(names) if names is not None else feature_names(len(probabilities))
    base = UnconstrainedBelief(dict(zip(names, (float(q) for q in probabilities))))
    constraint = group_constraint(names, groups, exactly_one=exactly_one)
    if constraint is None:
        return base
    if not isinstance(normalizer, Normalizer):
        normalizer = GivenNormalizer(normalizer, exact)
    return TaskBelief(base, constraint, normalizer=normalizer)


def _to_tuple(state: frozenset[str], index: dict[str, int], count: int) -> State:
    values = [False] * count
    for name in state:
        values[index[name]] = True
    return tuple(values)


def ranked_states(
    probabilities: Sequence[float],
    groups: Sequence[Sequence[int]],
    normalizer: float,
    *,
    theta: float | None = None,
    exact: bool = True,
    exactly_one: bool = False,
    timeout_seconds: float | None = None,
    max_selected_states: int | None = None,
) -> SearchResult:
    """Run PRaVaSS: valid states in non-increasing probability.

    Stops once the states reach mass ``theta`` (``None``: every valid state),
    after ``max_selected_states`` states, or at ``timeout_seconds``, whichever
    comes first. A timeout keeps the states found so far (status ``"timeout"``).
    """
    start = time.perf_counter()
    count = len(probabilities)
    names = feature_names(count)
    index = {name: i for i, name in enumerate(names)}
    states: list[State] = []
    status = "ok"
    examined = explored = 0
    if theta != 0.0 or max_selected_states is not None:
        belief = task_belief(probabilities, groups, normalizer, exact=exact, exactly_one=exactly_one, names=names)
        search = iter_pravass(
            belief,
            None if theta == 0.0 else theta,  # θ = 0 with a cap: the first states
            max_states=None,
            time_limit=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
        try:
            for state in search:
                states.append(_to_tuple(state, index, count))
                if max_selected_states is not None and len(states) >= max_selected_states:
                    break
        except TimeLimitError:
            status = "timeout"
        except InfeasibleConstraintError:
            pass  # no valid state: an empty selection
        examined, explored = search.examined, search.explored

    unconstrained_mass = sum(unconstrained_probability(state, probabilities) for state in states)
    constrained_mass = unconstrained_mass / normalizer
    return SearchResult(
        state_set=StateSet(
            states=tuple(states),
            unconstrained_mass=unconstrained_mass,
            constrained_mass=constrained_mass,
            total_states_considered=explored,
            status=status,
        ),
        stats=SearchStats(
            states_popped=examined,
            states_scored=explored,
            valid_states_selected=len(states),
            invalid_states_skipped=examined - len(states),
            cumulative_mass=constrained_mass,
            runtime_seconds=time.perf_counter() - start,
            timeout=status == "timeout",
        ),
    )


def iter_ranked_states(
    probabilities: Sequence[float],
    groups: Sequence[Sequence[int]],
    *,
    exactly_one: bool = False,
    names: Sequence[str] | None = None,
) -> Iterator[tuple[State, float]]:
    """Every valid state with its unconstrained probability, most probable first."""
    count = len(probabilities)
    names = tuple(names) if names is not None else feature_names(count)
    index = {name: i for i, name in enumerate(names)}
    belief = task_belief(probabilities, groups, UnitNormalizer(), exactly_one=exactly_one, names=names)
    for state in iter_pravass(belief, max_states=None):
        values = _to_tuple(state, index, count)
        yield values, unconstrained_probability(values, probabilities)
