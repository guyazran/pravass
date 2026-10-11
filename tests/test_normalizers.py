"""Normalization, EnumerationNormalizer, and UnitNormalizer."""

import math
import time

import pytest

from pravass import (
    EnumerationNormalizer,
    FunctionConstraint,
    MutexGroups,
    Normalization,
    StateLimitError,
    UnconstrainedBelief,
    UnitNormalizer,
)


class TestNormalization:
    @pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
    def test_rejects_bad_values(self, value):
        with pytest.raises(ValueError):
            Normalization(value, exact=True)

    def test_coerces_real_numbers(self):
        assert Normalization(1, exact=False).value == 1.0

    def test_exact_value_is_a_probability(self):
        with pytest.raises(ValueError):
            Normalization(1.5, exact=True)
        assert Normalization(1.5, exact=False).value == 1.5

    def test_rejects_non_bool_exact(self):
        with pytest.raises(TypeError):
            Normalization(0.5, exact=1)


class TestEnumerationNormalizer:
    def test_app_a(self, app_a_unconstrained, app_a_constraint):
        result = EnumerationNormalizer()(app_a_unconstrained, app_a_constraint)
        assert result.exact and result.value == pytest.approx(0.72)

    def test_support_does_not_change_z(self, app_a_unconstrained):
        # A feature outside the support (f3) must integrate out, not count as false.
        rule = lambda s: not {"f1", "f2"} <= s  # noqa: E731
        with_support = EnumerationNormalizer()(
            app_a_unconstrained, FunctionConstraint(rule, support={"f1", "f2"})
        )
        without = EnumerationNormalizer()(app_a_unconstrained, FunctionConstraint(rule))
        assert with_support.value == pytest.approx(0.72)
        assert without.value == pytest.approx(0.72)

    def test_too_small_support_is_detected(self, app_a_unconstrained):
        wrong = FunctionConstraint(lambda s: "f1" in s or "f2" in s, support={"f1"})
        with pytest.raises(ValueError, match="outside its declared support"):
            EnumerationNormalizer()(app_a_unconstrained, wrong)

    def test_support_as_list_from_a_subclass(self, app_a_unconstrained):
        from pravass import Constraint

        class Rule(Constraint):
            def __call__(self, state):
                return not {"f1", "f2"} <= state

            @property
            def support(self):
                return ["f1", "f2"]

        assert EnumerationNormalizer()(app_a_unconstrained, Rule()).value == pytest.approx(0.72)

    def test_support_avoids_enumerating_everything(self):
        belief = UnconstrainedBelief({f"x{i}": 0.3 for i in range(40)})
        constraint = MutexGroups([["x0", "x1", "x2"]])  # support = 3 features
        start = time.perf_counter()
        result = EnumerationNormalizer()(belief, constraint)
        assert time.perf_counter() - start < 1.0
        assert result.value == pytest.approx(0.7**3 + 3 * 0.3 * 0.7**2)

    def test_limit(self):
        belief = UnconstrainedBelief({f"x{i}": 0.3 for i in range(5)})
        constraint = FunctionConstraint(lambda s: True)
        with pytest.raises(StateLimitError, match="MutexNormalizer"):
            EnumerationNormalizer(max_states=31)(belief, constraint)
        assert EnumerationNormalizer(max_states=32)(belief, constraint).value == pytest.approx(1.0)
        assert EnumerationNormalizer(max_states=None)(belief, constraint).exact
        assert EnumerationNormalizer(max_states=float("inf"))(belief, constraint).exact

    @pytest.mark.parametrize("max_states", [0, -3])
    def test_rejects_small_limits(self, max_states):
        with pytest.raises(ValueError):
            EnumerationNormalizer(max_states=max_states)

    @pytest.mark.parametrize("max_states", [1e6, 2.5, True, "10"])
    def test_rejects_non_int_limits(self, max_states):
        with pytest.raises(TypeError):
            EnumerationNormalizer(max_states=max_states)


def test_unit_normalizer(app_a_unconstrained, app_a_constraint):
    assert UnitNormalizer()(app_a_unconstrained, app_a_constraint) == Normalization(1.0, exact=False)
