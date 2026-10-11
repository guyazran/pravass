"""Core vocabulary: states, belief sets, beliefs, and errors.

A *state* is the set of state features that are true in it (the paper's
``s ∈ 2^F``); every feature not in the set is false. A *belief* is a
probability distribution over the states of a fixed feature set ``F``, and a
*belief set* is a set of states.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable

State = frozenset[str]
"""The features that are true in a state; all other features are false."""

BeliefSet = frozenset[State]
"""A set of states, e.g. a θ-confidence minimal belief set returned by PRaVaSS."""


class InfeasibleConstraintError(ValueError):
    """No state that satisfies the constraints has nonzero probability."""


class StateLimitError(ValueError):
    """A computation would visit more states than its ``max_states`` limit."""


class TimeLimitError(RuntimeError):
    """A search ran past its ``time_limit``."""


class Belief(ABC):
    """A probability distribution over the states of a fixed feature set.

    Beliefs in this package are observation-conditioned: they describe the
    state given a single observation, and have no update or predict steps.
    """

    @property
    @abstractmethod
    def features(self) -> frozenset[str]:
        """The state features ``F`` this belief is defined over."""

    @abstractmethod
    def __call__(self, state: State) -> float:
        """Return the probability of ``state``.

        Raises:
            TypeError: If ``state`` is not a set or frozenset.
            ValueError: If ``state`` contains features outside :attr:`features`.
        """

    def marginal(self, features: Iterable[str]) -> "Belief":
        """Return the marginal belief over a subset of :attr:`features`.

        Used to compute normalizers over only the features a constraint reads.
        Beliefs that cannot marginalize cheaply raise ``NotImplementedError``
        (the default).
        """
        raise NotImplementedError(f"{type(self).__name__} does not support marginal()")
