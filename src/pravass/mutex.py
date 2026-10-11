"""Mutex groups: at-most-one constraints and their closed-form normalizer.

A mutex group is a set of features of which at most one can be true in a valid
state, e.g. ``{"on(milk,top-shelf)", "on(milk,bottom-shelf)", "on(milk,counter)"}``.
Planners such as Fast Downward extract mutex groups from a PDDL task
automatically (paper Sec. 2.2 and App. G).

Under an unconstrained belief, the probability that a single group holds has a
closed form, and disjoint groups are independent, so the normalizer of a family
of disjoint groups is exact and costs ``O(|F|)``. Overlapping groups are relaxed
to a disjoint family, which gives an upper bound (App. G.5).
"""

import math
from collections.abc import Iterable
from typing import final

from ._validation import check_features
from .base import Belief, State
from .constraints import Constraint
from .normalizers import Normalization, Normalizer
from .unconstrained import UnconstrainedBelief

Groups = frozenset[frozenset[str]]


def _canonical_groups(groups: Iterable[Iterable[str]], *, what: str) -> Groups:
    """Freeze groups, drop groups with fewer than two features, drop subsumed groups."""
    if isinstance(groups, str):
        raise TypeError(f"{what} must be an iterable of groups, not a string")
    frozen = {check_features(group, what=f"each group in {what}") for group in groups}
    return _drop_subsumed(g for g in frozen if len(g) >= 2)


def _drop_subsumed(groups: Iterable[frozenset[str]]) -> Groups:
    """Remove groups contained in a larger group (implied by it)."""
    by_size = sorted(set(groups), key=len, reverse=True)
    kept: list[frozenset[str]] = []
    for group in by_size:
        if not any(group <= other for other in kept):
            kept.append(group)
    return frozenset(kept)


def _pairwise_disjoint(groups: Iterable[frozenset[str]]) -> bool:
    seen: set[str] = set()
    for group in groups:
        if seen & group:
            return False
        seen |= group
    return True


def _group_mass(group: frozenset[str], probabilities) -> float:
    """P(at most one feature of ``group`` is true) under independence.

    A two-state recursion over "none true so far" and "exactly one true so far"
    (paper App. G.4); it never divides, so probabilities of 0 and 1 are safe.
    """
    none, one = 1.0, 0.0
    for feature in sorted(group):
        p = probabilities[feature]
        none, one = none * (1.0 - p), one * (1.0 - p) + none * p
    return none + one


def _greedy_disjoint_cover(groups: Groups) -> Groups:
    """Shrink overlapping groups to a disjoint family, largest groups first.

    Each kept group is a subset of a constraint group, so the constraint
    implies the cover and the cover's normalizer is an upper bound.
    """
    covered: set[str] = set()
    cover: list[frozenset[str]] = []
    for group in sorted(groups, key=lambda g: (-len(g), sorted(g))):
        remainder = group - covered
        if len(remainder) >= 2:
            cover.append(remainder)
            covered |= remainder
    return frozenset(cover)


@final
class MutexGroups(Constraint):
    """At most one feature of each group is true.

    Args:
        groups: An iterable of groups, each an iterable of feature names. Groups
            with fewer than two features are vacuous and dropped, as are
            duplicate groups and groups contained in another group; this does
            not change which states are valid.

    Features named in a group but absent from a belief are never true in that
    belief's states, so the same groups can be reused when estimating only a
    subset of the features.

    Example:
        >>> constraint = MutexGroups([["in(eggs,fridge)", "on(eggs,counter)"]])
        >>> constraint(frozenset({"in(eggs,fridge)"}))
        True
        >>> constraint(frozenset({"in(eggs,fridge)", "on(eggs,counter)"}))
        False
    """

    def __init__(self, groups: Iterable[Iterable[str]]):
        self._groups = _canonical_groups(groups, what="groups")

    @property
    def groups(self) -> Groups:
        """The canonical groups."""
        return self._groups

    @property
    def support(self) -> frozenset[str]:
        return frozenset().union(*self._groups)

    def __call__(self, state: State) -> bool:
        return all(len(group & state) <= 1 for group in self._groups)

    def __repr__(self) -> str:
        groups = sorted(sorted(g) for g in self._groups)
        return f"MutexGroups({groups!r})"


@final
class MutexNormalizer(Normalizer):
    """Closed-form normalizer for :class:`MutexGroups` under an
    :class:`~pravass.UnconstrainedBelief` (paper App. G).

    The normalizer is exact when the groups are pairwise disjoint, after
    removing features with probability 0 (they are never true). Overlapping
    groups are relaxed to a disjoint family: ``selected_groups`` if given, or
    otherwise a greedy cover that keeps the largest groups first. The relaxed
    normalizer is an upper bound and is reported as inexact.

    Args:
        selected_groups: Optional disjoint family to use for overlapping groups,
            such as the disjoint cover a planner computes alongside its mutex
            groups (the paper's ``C_Z``). Each selected group must be a clique of
            the constraint's conflicts: every pair of its features must share a
            constraint group. This is checked whenever the normalizer is called.

    Raises:
        TypeError: When called with anything other than an
            ``UnconstrainedBelief`` and ``MutexGroups``.
        ValueError: If ``selected_groups`` is invalid for the constraint, or if
            the normalizer underflows to 0 for a satisfiable constraint.
    """

    def __init__(self, selected_groups: "Iterable[Iterable[str]] | None" = None):
        if selected_groups is None:
            self._selected: "Groups | None" = None
        else:
            if isinstance(selected_groups, str):
                raise TypeError("selected_groups must be an iterable of groups, not a string")
            selected = frozenset(
                g
                for g in (check_features(group, what="each selected group") for group in selected_groups)
                if len(g) >= 2
            )
            if not _pairwise_disjoint(selected):
                raise ValueError("selected_groups must be pairwise disjoint")
            self._selected = selected

    def __call__(self, base: Belief, constraint: Constraint) -> Normalization:
        if type(base) is not UnconstrainedBelief or type(constraint) is not MutexGroups:
            raise TypeError(
                "MutexNormalizer needs an UnconstrainedBelief and MutexGroups, got "
                f"{type(base).__name__} and {type(constraint).__name__}; use "
                "EnumerationNormalizer or UnitNormalizer for other combinations"
            )
        probabilities = base.factored_belief
        if self._selected is not None:
            self._check_selected(constraint.groups)

        def restrict(groups: Groups) -> Groups:
            return _drop_subsumed(
                r
                for r in (frozenset(f for f in g if probabilities.get(f, 0.0) > 0.0) for g in groups)
                if len(r) >= 2
            )

        groups = restrict(constraint.groups)
        for group in groups:
            if sum(1 for f in group if probabilities[f] == 1.0) >= 2:
                return Normalization(0.0, exact=True)

        if _pairwise_disjoint(groups):
            cover, exact = groups, True
        else:
            cover = restrict(self._selected) if self._selected is not None else _greedy_disjoint_cover(groups)
            exact = all(any(g <= c for c in cover) for g in groups)
        z = math.prod(_group_mass(g, probabilities) for g in sorted(cover, key=sorted))
        if z == 0.0:
            raise ValueError(
                "the normalizer underflows to 0 although some valid state has "
                "nonzero probability; the belief is too concentrated on invalid states"
            )
        return Normalization(z, exact=exact)

    def _check_selected(self, groups: Groups) -> None:
        """Every selected group must be a clique of the constraint's conflict graph."""
        memberships: dict[str, set[int]] = {}
        for index, group in enumerate(groups):
            for feature in group:
                memberships.setdefault(feature, set()).add(index)
        for selected in self._selected:
            members = sorted(selected)
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    if not memberships.get(a, set()) & memberships.get(b, set()):
                        raise ValueError(
                            f"selected group {members} is not implied by the "
                            f"constraint: {a!r} and {b!r} share no mutex group"
                        )

    def __repr__(self) -> str:
        if self._selected is None:
            return "MutexNormalizer()"
        return f"MutexNormalizer(selected_groups={sorted(sorted(g) for g in self._selected)!r})"
