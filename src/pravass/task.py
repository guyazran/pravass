"""The task belief: a belief conditioned on task constraints (paper Eq. 2)."""

import sys

from ._validation import check_state
from .base import Belief, InfeasibleConstraintError, State
from .constraints import Constraint
from .normalizers import EnumerationNormalizer, Normalization, Normalizer


class TaskBelief(Belief):
    """The task belief ``β_C``: a belief conditioned on a constraint (Eq. 2).

        β_C(s) = β(s) / Z  if s satisfies the constraint, else 0.

    Args:
        base: The belief to condition, typically an
            :class:`~pravass.UnconstrainedBelief`.
        constraint: The task constraint (see :class:`~pravass.Constraint`).
        normalizer: How to compute ``Z`` (see :mod:`pravass.normalizers`).
            Defaults to the exact :class:`~pravass.EnumerationNormalizer`.

    Raises:
        InfeasibleConstraintError: If no state that satisfies the constraint has
            nonzero probability (``Z = 0``).

    When :attr:`normalization` is not exact, the probabilities returned are
    lower bounds on the true task-belief probabilities.
    """

    def __init__(
        self,
        base: Belief,
        constraint: Constraint,
        *,
        normalizer: "Normalizer | None" = None,
    ):
        if not isinstance(base, Belief):
            raise TypeError(f"base must be a Belief, got {type(base).__name__}")
        if not isinstance(constraint, Constraint):
            raise TypeError(
                f"constraint must be a Constraint, got {type(constraint).__name__}; "
                "wrap plain functions in FunctionConstraint"
            )
        if normalizer is None:
            normalizer = EnumerationNormalizer()
        if not isinstance(normalizer, Normalizer):
            raise TypeError(f"normalizer must be a Normalizer, got {type(normalizer).__name__}")
        normalization = normalizer(base, constraint)
        if not isinstance(normalization, Normalization):
            raise TypeError(
                f"{type(normalizer).__name__} returned {normalization!r}, not a Normalization"
            )
        if normalization.value == 0.0:
            raise InfeasibleConstraintError(
                "no state that satisfies the constraint has nonzero probability "
                "(the normalizer is 0); check for features with probability "
                "exactly 0 or 1 that contradict the constraint, or for "
                "probabilities so small that their products underflow"
            )
        if normalization.value < sys.float_info.min:
            raise ValueError(
                f"the normalizer {normalization.value!r} is below the floating-point "
                "range; probabilities this small cannot be represented accurately. "
                "Clamp near-zero marginals (e.g. to 1e-12) or treat them as 0."
            )
        self._base = base
        self._constraint = constraint
        self._normalization = normalization

    @property
    def features(self) -> frozenset[str]:
        return self._base.features

    @property
    def base(self) -> Belief:
        """The belief being conditioned."""
        return self._base

    @property
    def constraint(self) -> Constraint:
        """The task constraint."""
        return self._constraint

    @property
    def normalization(self) -> Normalization:
        """The normalizer ``Z`` used, and whether it is exact."""
        return self._normalization

    def __call__(self, state: State) -> float:
        state = check_state(state, self.features)
        if not self._constraint(state):
            return 0.0
        return self._base(state) / self._normalization.value

    def __repr__(self) -> str:
        return (
            f"TaskBelief({self._base!r}, {self._constraint!r}, "
            f"normalization={self._normalization!r})"
        )
