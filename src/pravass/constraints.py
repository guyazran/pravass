"""Constraints: functions from states to true/false.

A constraint ``C : 2^F → {0, 1}`` marks the states that are valid for a task,
e.g. "an object cannot be in two places at once". The paper extracts such
task-level constraints from a planning domain, but any deterministic function
of the state works.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

from ._validation import check_features
from .base import State


class Constraint(ABC):
    """A task constraint: a function from states to ``True`` (valid) or ``False``.

    Implementations must be deterministic and free of side effects.
    """

    @abstractmethod
    def __call__(self, state: State) -> bool:
        """Return whether ``state`` satisfies the constraint."""

    @property
    def support(self) -> "frozenset[str] | None":
        """The features this constraint reads, or ``None`` if unknown.

        Declaring a support lets :class:`~pravass.EnumerationNormalizer`
        enumerate only these features instead of the whole state space. A
        declared support must include every feature the constraint reads;
        otherwise the normalizer is wrong.
        """
        return None


class FunctionConstraint(Constraint):
    """A constraint defined by a plain function of the state.

    Args:
        fn: Called with a state (the frozenset of true features); returns
            whether the state is valid.
        support: Optional features ``fn`` reads (see :attr:`Constraint.support`).

    Example:
        >>> holds_one = FunctionConstraint(
        ...     lambda s: not {"holding(a)", "holding(b)"} <= s,
        ...     support={"holding(a)", "holding(b)"},
        ... )
    """

    def __init__(
        self,
        fn: Callable[[State], bool],
        *,
        support: "Iterable[str] | None" = None,
    ):
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {fn!r}")
        self._fn = fn
        self._support = None if support is None else check_features(support, what="support")

    def __call__(self, state: State) -> bool:
        return bool(self._fn(state))

    @property
    def support(self) -> "frozenset[str] | None":
        return self._support

    def __repr__(self) -> str:
        support = "" if self._support is None else f", support={sorted(self._support)!r}"
        return f"FunctionConstraint({self._fn!r}{support})"
