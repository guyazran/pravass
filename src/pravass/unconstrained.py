"""The unconstrained belief: independent features (paper Eq. 1).

The *factored belief* ``β_F`` gives each state feature its own probability of
being true. Assuming the features are independent turns it into a distribution
over whole states, the *unconstrained belief*

    β̃(s) = Π_{f ∈ s} β_F(f) · Π_{f ∉ s} (1 − β_F(f)).
"""

import math
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import final

from ._validation import check_feature, check_features, check_probability, check_state
from .base import Belief, State

FactoredBelief = Mapping[str, float]
"""The factored belief ``β_F``: the probability that each state feature is true.

With s3e, ``results.probabilities()`` is a factored belief.
"""


@final
class UnconstrainedBelief(Belief):
    """The unconstrained belief ``β̃`` induced by a factored belief (Eq. 1).

    Args:
        factored_belief: Maps each state feature to the probability that it is
            true. Probabilities of exactly 0 or 1 are hard evidence: states that
            contradict them get probability 0.

    Example:
        >>> belief = UnconstrainedBelief({"clear(top-shelf)": 0.6, "open(fridge)": 0.9})
        >>> round(belief(frozenset({"open(fridge)"})), 2)
        0.36
    """

    def __init__(self, factored_belief: FactoredBelief):
        if not isinstance(factored_belief, Mapping):
            raise TypeError(
                "factored_belief must be a mapping from feature names to "
                f"probabilities, got {type(factored_belief).__name__}"
            )
        validated = {
            check_feature(feature): check_probability(p, what=f"probability of {feature!r}")
            for feature, p in factored_belief.items()
        }
        # Sorted by feature, so products (and results) do not depend on input order.
        self._factored_belief: Mapping[str, float] = MappingProxyType(dict(sorted(validated.items())))
        self._features = frozenset(validated)

    @property
    def features(self) -> frozenset[str]:
        return self._features

    @property
    def factored_belief(self) -> Mapping[str, float]:
        """A read-only view of the factored belief ``β_F``."""
        return self._factored_belief

    def __call__(self, state: State) -> float:
        state = check_state(state, self._features)
        return math.prod(
            p if feature in state else 1.0 - p
            for feature, p in self._factored_belief.items()
        )

    def most_likely_state(self) -> State:
        """The state that sets every feature to its most probable value.

        Features with probability exactly 0.5 are set to false, so this is the
        first state :func:`~pravass.iter_pravass` examines. (s3e's
        ``PredictionSet.to_state(0.5)`` sets them to true.)
        """
        return frozenset(f for f, p in self._factored_belief.items() if p > 0.5)

    def marginal(self, features: Iterable[str]) -> "UnconstrainedBelief":
        """The unconstrained belief over a subset of the features."""
        subset = check_features(features, what="features")
        unknown = subset - self._features
        if unknown:
            raise ValueError(f"unknown features: {sorted(unknown)}")
        return UnconstrainedBelief({f: self._factored_belief[f] for f in subset})

    def __repr__(self) -> str:
        return f"UnconstrainedBelief({dict(self._factored_belief)!r})"
