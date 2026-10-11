"""TaskBelief: Eq. 2, normalizers, and infeasibility."""

import math

import pytest

from pravass import (
    EnumerationNormalizer,
    FunctionConstraint,
    InfeasibleConstraintError,
    MutexGroups,
    Normalizer,
    TaskBelief,
    UnconstrainedBelief,
    UnitNormalizer,
)

from oracle import all_states


def test_eq2_by_hand(app_a_unconstrained, app_a_constraint):
    belief = TaskBelief(app_a_unconstrained, app_a_constraint)
    assert belief(frozenset({"f2", "f3"})) == pytest.approx(0.525)
    assert belief(frozenset({"f3"})) == pytest.approx(0.225)
    assert belief(frozenset({"f1", "f2"})) == 0.0


def test_default_normalizer_is_exact_enumeration(app_a_unconstrained, app_a_constraint):
    belief = TaskBelief(app_a_unconstrained, app_a_constraint)
    assert belief.normalization.exact
    assert belief.normalization.value == pytest.approx(0.72)


def test_exact_normalizer_sums_to_one(app_a_unconstrained, app_a_constraint):
    belief = TaskBelief(app_a_unconstrained, app_a_constraint)
    assert math.fsum(belief(s) for s in all_states(belief.features)) == pytest.approx(1.0)


def test_unit_normalizer_gives_lower_bounds(app_a_unconstrained, app_a_constraint):
    exact = TaskBelief(app_a_unconstrained, app_a_constraint)
    relaxed = TaskBelief(app_a_unconstrained, app_a_constraint, normalizer=UnitNormalizer())
    assert relaxed.normalization.exact is False
    for state in all_states(exact.features):
        assert relaxed(state) <= exact(state)
    assert math.fsum(relaxed(s) for s in all_states(relaxed.features)) == pytest.approx(0.72)


def test_features_and_accessors(app_a_unconstrained, app_a_constraint):
    belief = TaskBelief(app_a_unconstrained, app_a_constraint)
    assert belief.features == app_a_unconstrained.features
    assert belief.base is app_a_unconstrained
    assert belief.constraint is app_a_constraint


def test_infeasible_constraint():
    unconstrained = UnconstrainedBelief({"a": 1.0, "b": 1.0})
    with pytest.raises(InfeasibleConstraintError):
        TaskBelief(unconstrained, MutexGroups([["a", "b"]]))


def test_nested_conditioning_is_allowed():
    unconstrained = UnconstrainedBelief({"a": 0.5, "b": 0.5, "c": 0.5})
    first = TaskBelief(unconstrained, MutexGroups([["a", "b"]]))
    second = TaskBelief(first, FunctionConstraint(lambda s: "c" in s))
    assert second(frozenset({"c"})) == pytest.approx(1 / 3)


class TestTypeChecks:
    def test_rejects_plain_function(self, app_a_unconstrained):
        with pytest.raises(TypeError, match="FunctionConstraint"):
            TaskBelief(app_a_unconstrained, lambda s: True)

    def test_rejects_non_belief(self, app_a_constraint):
        with pytest.raises(TypeError):
            TaskBelief({"f1": 0.5}, app_a_constraint)

    def test_rejects_non_normalizer(self, app_a_unconstrained, app_a_constraint):
        with pytest.raises(TypeError):
            TaskBelief(app_a_unconstrained, app_a_constraint, normalizer=lambda b, c: 1.0)

    def test_rejects_normalizer_returning_float(self, app_a_unconstrained, app_a_constraint):
        class Bad(Normalizer):
            def __call__(self, base, constraint):
                return 1.0

        with pytest.raises(TypeError):
            TaskBelief(app_a_unconstrained, app_a_constraint, normalizer=Bad())


def test_explicit_enumeration_normalizer(app_a_unconstrained, app_a_constraint):
    belief = TaskBelief(app_a_unconstrained, app_a_constraint, normalizer=EnumerationNormalizer(max_states=8))
    assert belief.normalization.exact


def test_normalizer_below_float_range_raises():
    unconstrained = UnconstrainedBelief({"a": 1e-300, "b": 1e-20, "c": 0.5})
    constraint = FunctionConstraint(lambda s: {"a", "b"} <= s, support={"a", "b"})
    with pytest.raises(ValueError, match="floating-point range"):
        TaskBelief(unconstrained, constraint)
