"""PRaVaSS: Probability-Ranked Valid State Search (paper Sec. 3, Algorithm 1).

PRaVaSS computes a θ-confidence Minimal Belief Set (MBS_θ): a smallest set of
valid states whose total task-belief probability reaches ``θ``, made of the most
probable valid states. It runs a best-first search from the most likely state,
where neighbouring states differ in one feature, ordered by the unconstrained
belief. Valid states then come out in non-increasing task-belief order
(Theorem 1), so the search can stop as soon as it has collected enough mass,
without enumerating the state space.
"""

import heapq
import math
import time

from ._validation import DEFAULT_MAX_STATES, check_max_states, check_theta, check_time_limit
from .base import BeliefSet, InfeasibleConstraintError, State, StateLimitError, TimeLimitError
from .normalizers import Normalization
from .task import TaskBelief
from .unconstrained import UnconstrainedBelief

TOLERANCE = 1e-9
"""Absolute tolerance on θ: PRaVaSS guarantees coverage of at least ``θ - TOLERANCE``."""


class _CompensatedSum:
    """Running sum with Neumaier compensation (keeps tiny terms accurate)."""

    def __init__(self) -> None:
        self._sum = 0.0
        self._correction = 0.0

    def add(self, value: float) -> None:
        total = self._sum + value
        if abs(self._sum) >= abs(value):
            self._correction += (self._sum - total) + value
        else:
            self._correction += (value - total) + self._sum
        self._sum = total

    @property
    def value(self) -> float:
        return self._sum + self._correction


def pravass(
    belief: "UnconstrainedBelief | TaskBelief",
    theta: float,
    *,
    max_states: "int | float | None" = DEFAULT_MAX_STATES,
) -> BeliefSet:
    """Compute a θ-confidence minimal belief set with PRaVaSS.

    Args:
        belief: A :class:`~pravass.TaskBelief` whose base is an
            :class:`~pravass.UnconstrainedBelief`, or an ``UnconstrainedBelief``
            on its own (no constraints).
        theta: The mass threshold ``θ`` in [0, 1]. ``θ = 0`` returns an empty set.
        max_states: Stop with :class:`~pravass.StateLimitError` instead of
            inserting more states than this into the search (this bounds
            memory; each explored state inserts up to one neighbour per
            feature). ``None`` or ``math.inf`` means no limit, which guarantees
            an answer but can take time exponential in the number of features.

    Returns:
        The most probable valid states, whose total probability under the task
        belief is at least ``θ - TOLERANCE`` (up to floating-point rounding). If ``belief.normalization`` is
        exact, the result is a θ-confidence minimal belief set (one of several
        if probabilities tie). If not, it is correctly ordered and contains a
        minimal belief set, but may be larger. Its total probability is then
        still at least ``θ - TOLERANCE``, but ``sum(belief(s) for s in result)``
        computes only a lower bound on it, which can be below ``θ``: the search
        certifies the mass with a tighter bound on the normalizer that it learns
        along the way.

    Raises:
        TypeError: If ``belief`` is not one of the types above.
        InfeasibleConstraintError: If no valid state has nonzero probability.
        StateLimitError: If the search would exceed ``max_states``.
        ValueError: If ``theta`` is outside [0, 1], or the belief is so diffuse
            that the most likely state's probability underflows.

    To get the states one at a time, most probable first, or to bound the
    search time, use :func:`iter_pravass`.

    Example:
        >>> from pravass import MutexGroups, TaskBelief, UnconstrainedBelief
        >>> unconstrained = UnconstrainedBelief({"a": 0.4, "b": 0.7, "c": 0.9})
        >>> belief = TaskBelief(unconstrained, MutexGroups([["a", "b"]]))
        >>> sorted(sorted(s) for s in pravass(belief, theta=0.8))
        [['a', 'c'], ['b', 'c'], ['c']]
    """
    return frozenset(PravassIterator(belief, check_theta(theta), max_states=max_states, time_limit=None))


def iter_pravass(
    belief: "UnconstrainedBelief | TaskBelief",
    theta: "float | None" = None,
    *,
    max_states: "int | float | None" = DEFAULT_MAX_STATES,
    time_limit: "float | None" = None,
) -> "PravassIterator":
    """Run PRaVaSS lazily, yielding valid states from most to least probable.

    ``frozenset(iter_pravass(belief, theta))`` equals ``pravass(belief, theta)``.
    Iterating instead gives the states in non-increasing task-belief
    probability, so a caller can stop early, for example at the largest prefix
    a planner can still handle (the paper's AMS), and can read how much search
    it took.

    Equally probable states come out in a fixed order: features with
    probability exactly 0.5 start false, and among tied states the search has
    found, it takes the one that is false at the first feature (in sorted
    order) where they differ.

    Args:
        belief: As for :func:`pravass`.
        theta: Stop once the yielded states provably carry probability
            ``θ - TOLERANCE``. ``None`` yields every valid state with nonzero
            probability.
        max_states: As for :func:`pravass`.
        time_limit: Seconds after which the next step raises
            :class:`~pravass.TimeLimitError`. The states yielded until then are
            still the most probable valid states. ``None`` means no limit.

    Returns:
        A :class:`PravassIterator`. Besides yielding states, it reports
        :attr:`~PravassIterator.mass`, :attr:`~PravassIterator.explored` and
        :attr:`~PravassIterator.examined`.

    Raises:
        The errors of :func:`pravass`, when the iterator is created (``TypeError``,
        ``ValueError``) or while iterating (``StateLimitError``,
        ``InfeasibleConstraintError``), and ``TimeLimitError`` while iterating.

    Example:
        >>> from pravass import MutexGroups, TaskBelief, UnconstrainedBelief
        >>> unconstrained = UnconstrainedBelief({"a": 0.4, "b": 0.7, "c": 0.9})
        >>> belief = TaskBelief(unconstrained, MutexGroups([["a", "b"]]))
        >>> [sorted(s) for s in iter_pravass(belief)]
        [['b', 'c'], ['c'], ['a', 'c'], ['b'], [], ['a']]
        >>> search = iter_pravass(belief, theta=0.7)
        >>> [sorted(s) for s in search], round(search.mass, 3)
        ([['b', 'c'], ['c']], 0.75)
    """
    return PravassIterator(
        belief,
        None if theta is None else check_theta(theta),
        max_states=max_states,
        time_limit=time_limit,
    )


class PravassIterator:
    """The valid states found by PRaVaSS, most probable first.

    Created by :func:`iter_pravass`; see there for the arguments and errors.
    The search runs only as far as the states are consumed.
    """

    def __init__(
        self,
        belief: "UnconstrainedBelief | TaskBelief",
        theta: "float | None",
        *,
        max_states: "int | float | None",
        time_limit: "float | None",
    ):
        self._limit = check_max_states(max_states)
        self._time_limit = check_time_limit(time_limit)
        self._deadline = None if self._time_limit is None else time.perf_counter() + self._time_limit
        if type(belief) is UnconstrainedBelief:
            unconstrained, self._constraint = belief, None
            normalization = Normalization(1.0, exact=True)
        elif isinstance(belief, TaskBelief) and type(belief.base) is UnconstrainedBelief:
            unconstrained, self._constraint = belief.base, belief.constraint
            normalization = belief.normalization
        else:
            raise TypeError(
                "pravass needs an UnconstrainedBelief or a TaskBelief whose base is an "
                f"UnconstrainedBelief, got {belief!r}; to combine several constraints, "
                "condition once on a single constraint that checks all of them"
            )
        self._theta = theta
        self._normalizer, self._exact = normalization.value, normalization.exact
        self._valid_mass = _CompensatedSum()  # V: unconstrained mass of valid popped states
        self._popped_mass = _CompensatedSum()  # E: unconstrained mass of all popped states
        self._visited: set[int] = set()
        self._examined = 0
        if theta == 0.0:
            self._steps = iter(())
            return

        # Bit i holds the i-th feature in reverse sorted order, so the frontier's
        # (cost, mask) order breaks ties by the first sorted feature where two
        # states differ, false before true. Features at exactly 0.5 start false.
        self._features = sorted(unconstrained.features, reverse=True)
        self._probabilities = [unconstrained.factored_belief[f] for f in self._features]
        self._start = sum(1 << i for i, p in enumerate(self._probabilities) if p > 0.5)
        self._costs = [_flip_cost(p) for p in self._probabilities]
        self._log_start_mass = math.fsum(math.log(max(p, 1.0 - p)) for p in self._probabilities)
        if math.exp(self._log_start_mass) == 0.0:
            raise ValueError(
                "the belief is too diffuse: the most likely state's probability "
                "underflows, so any minimal belief set would be astronomically large"
            )
        self._slack = (len(self._features) + 2) ** 2 * 2.0**-52  # floating-point slack for 1 - E
        self._visited.add(self._start)
        self._steps = self._search()

    def __iter__(self) -> "PravassIterator":
        return self

    def __next__(self) -> State:
        return next(self._steps)

    @property
    def mass(self) -> float:
        """A certified lower bound on the task-belief probability of the states yielded so far."""
        valid = self._valid_mass.value
        return valid / self._bound(valid) if valid > 0.0 else 0.0

    @property
    def explored(self) -> int:
        """States inserted into the search so far (the quantity ``max_states`` limits)."""
        return len(self._visited)

    @property
    def examined(self) -> int:
        """States taken from the search frontier so far, valid or not."""
        return self._examined

    def _bound(self, valid: float) -> float:
        # With an exact normalizer Z, the yielded states carry V / Z. Otherwise Z
        # is only an upper bound, and Z_true <= V + (mass not yet popped) gives a
        # bound that tightens as the search proceeds.
        if self._exact:
            return self._normalizer
        return min(self._normalizer, valid + max(0.0, 1.0 - self._popped_mass.value) + self._slack)

    def _search(self):
        features, costs, start, constraint = self._features, self._costs, self._start, self._constraint
        target = None if self._theta is None else self._theta - TOLERANCE
        heap: list[tuple[float, int]] = [(0.0, start)]
        visited = self._visited

        while heap:
            if self._deadline is not None and time.perf_counter() >= self._deadline:
                raise TimeLimitError(
                    f"PRaVaSS reached its time limit of {self._time_limit} s after examining "
                    f"{self._examined} states; the states yielded so far are the most "
                    f"probable valid states, with probability at least {self.mass:.6g}"
                )
            cost, mask = heapq.heappop(heap)
            mass = math.exp(self._log_start_mass - cost)
            self._popped_mass.add(mass)
            self._examined += 1
            state = frozenset(f for i, f in enumerate(features) if mask >> i & 1)
            valid = constraint is None or constraint(state)
            if valid:
                self._valid_mass.add(mass)

            # Stop once the yielded states provably carry mass >= theta. This is
            # checked after every popped state, valid or not, because invalid
            # states tighten the bound on an inexact normalizer.
            v = self._valid_mass.value
            done = target is not None and v > 0.0 and v >= target * self._bound(v)
            if valid:
                yield state
            if done:
                return

            for i, step in enumerate(costs):
                if step == math.inf:
                    continue
                neighbor = mask ^ (1 << i)
                if neighbor in visited:
                    continue
                if len(visited) >= self._limit:
                    raise StateLimitError(
                        _limit_message(self._limit, self._theta, self._probabilities, self._valid_mass.value)
                    )
                visited.add(neighbor)
                # Flipping away from the most likely value adds the cost; flipping back removes it.
                moved_away = (neighbor >> i & 1) != (start >> i & 1)
                heapq.heappush(heap, (cost + step if moved_away else cost - step, neighbor))

        if self._valid_mass.value == 0.0:
            raise InfeasibleConstraintError(
                "no state that satisfies the constraint has nonzero probability"
            )
        # Every state with nonzero probability was popped: all the mass was yielded.


def _flip_cost(p: float) -> float:
    """Cost of moving a feature away from its most likely value.

    log(max(p, 1 - p)) - log(min(p, 1 - p)) >= 0, written with log1p to stay
    accurate near p = 0.5; infinite for hard evidence (p = 0 or 1).
    """
    low, high = min(p, 1.0 - p), max(p, 1.0 - p)
    if low == 0.0:
        return math.inf
    ratio = (high - low) / low
    return math.log(high) - math.log(low) if math.isinf(ratio) else math.log1p(ratio)


def _limit_message(limit: float, theta: "float | None", probabilities: list[float], valid_mass: float) -> str:
    goal = "enumerating the valid states" if theta is None else f"collecting mass theta={theta}"
    message = (
        f"PRaVaSS inserted max_states={limit} states into its search before {goal}. "
        "Each explored state inserts up to one neighbour per "
        f"feature ({len(probabilities)} here). Raise max_states, or pass "
        "max_states=None for no limit."
    )
    if valid_mass == 0.0:
        message += " No valid state was found yet; check whether the constraint can be satisfied."
    ties = sum(1 for p in probabilities if p == 0.5)
    if ties:
        message += (
            f" Note: {ties} features have probability exactly 0.5 (e.g. unanswered "
            f"queries), which makes 2^{ties} states equally likely."
        )
    return message
