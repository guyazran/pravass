"""Normalizers for conditioning a belief on task constraints.

Conditioning a belief ``β`` on a constraint ``C`` gives the task belief

    β_C(s) = β(s) · 1[C(s)] / Z,    Z = Σ_{s ⊨ C} β(s),

so the normalizer ``Z`` is the probability that a state drawn from ``β``
satisfies ``C``. This is specific to constraint conditioning. It is not the
evidence term ``P(o)`` that normalizes a Bayesian filter update: beliefs in this
package are already conditioned on the observation and have no update step.

Computing ``Z`` exactly can require summing over exponentially many states, so a
normalizer may instead return an upper bound (a *relaxed* normalizer, paper
Sec. 2.3). An upper bound never makes PRaVaSS unsafe: state probabilities become
lower bounds and the returned belief set can only grow. A value below the true
``Z`` would be unsafe, so normalizers must never return one.
"""

import math
import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product

from ._validation import DEFAULT_MAX_STATES, check_max_states
from .base import Belief, StateLimitError
from .constraints import Constraint


@dataclass(frozen=True)
class Normalization:
    """The result of a normalizer.

    Attributes:
        value: ``Z``, or an upper bound on it when ``exact`` is false.
        exact: Whether ``value`` equals ``Z`` (up to floating-point rounding).
    """

    value: float
    exact: bool

    def __post_init__(self):
        if not isinstance(self.exact, bool):
            raise TypeError(f"exact must be a bool, got {self.exact!r}")
        if isinstance(self.value, bool) or not isinstance(self.value, numbers.Real):
            raise TypeError(f"value must be a real number, got {self.value!r}")
        object.__setattr__(self, "value", float(self.value))
        if not (math.isfinite(self.value) and self.value >= 0.0):
            raise ValueError(f"a normalizer must be finite and non-negative, got {self.value!r}")
        if self.exact and self.value > 1.0 + 1e-9:
            raise ValueError(f"an exact normalizer is a probability, at most 1, got {self.value!r}")


class Normalizer(ABC):
    """Computes the normalizer ``Z`` for conditioning a belief on a constraint.

    Implementations return ``Z`` exactly or an upper bound on it, never less.
    """

    @abstractmethod
    def __call__(self, base: Belief, constraint: Constraint) -> Normalization:
        """Return the normalizer for conditioning ``base`` on ``constraint``."""


class EnumerationNormalizer(Normalizer):
    """The exact normalizer, computed by summing over states.

    This is the default, and the only part of the package that enumerates a
    state space. It works for any belief and any constraint. When the
    constraint declares a :attr:`~pravass.Constraint.support` and the belief
    supports :meth:`~pravass.Belief.marginal`, only the support's ``2^|support|``
    states are enumerated; otherwise all ``2^|F|`` states are.

    Without a declared support, the cost is comparable to the exhaustive
    baseline in the paper, so the speed of PRaVaSS then comes from the other
    normalizers.

    Args:
        max_states: Refuse (with :class:`~pravass.StateLimitError`) to enumerate
            more states than this. ``None`` or ``math.inf`` means no limit.
    """

    def __init__(self, max_states: "int | float | None" = DEFAULT_MAX_STATES):
        self._max_states = check_max_states(max_states)

    def __call__(self, base: Belief, constraint: Constraint) -> Normalization:
        belief = base
        outside: frozenset[str] = frozenset()
        if constraint.support is not None:
            try:
                belief = base.marginal(frozenset(constraint.support) & base.features)
                outside = base.features - belief.features
            except NotImplementedError:
                pass
        features = sorted(belief.features)
        count = 2 ** len(features)
        if count > self._max_states:
            raise StateLimitError(
                f"EnumerationNormalizer would enumerate 2^{len(features)} = {count} "
                f"states, more than max_states={self._max_states}. Use "
                "MutexNormalizer for mutex-group constraints, UnitNormalizer for a "
                "cheap upper bound, declare the constraint's support, or pass "
                "max_states=None to enumerate anyway."
            )
        states = (
            frozenset(f for f, value in zip(features, values) if value)
            for values in product((False, True), repeat=len(features))
        )
        z = math.fsum(belief(state) for state in states if _satisfies(constraint, state, outside))
        return Normalization(z, exact=True)

    def __repr__(self) -> str:
        return f"EnumerationNormalizer(max_states={self._max_states})"


def _satisfies(constraint: Constraint, state: frozenset[str], outside: frozenset[str]) -> bool:
    """Evaluate the constraint on a state over its support.

    The features outside the support are absent (false) in ``state``. Setting
    them all to true must not change the answer; if it does, the constraint
    reads features outside its declared support.
    """
    result = constraint(state)
    if outside and constraint(state | outside) != result:
        raise ValueError(
            "the constraint depends on features outside its declared support; "
            "declare every feature it reads"
        )
    return result


class UnitNormalizer(Normalizer):
    """The relaxed normalizer ``Z = 1`` (no normalization; paper Sec. 2.3).

    Always a valid upper bound, at no cost. Task-belief probabilities become
    lower bounds, and PRaVaSS returns a correctly ordered belief set that may
    be larger than the minimal one.
    """

    def __call__(self, base: Belief, constraint: Constraint) -> Normalization:
        return Normalization(1.0, exact=False)

    def __repr__(self) -> str:
        return "UnitNormalizer()"
