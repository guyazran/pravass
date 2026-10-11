"""UnconstrainedBelief: Eq. 1, validation, and the most likely state."""

import math

import numpy as np
import pytest

from pravass import UnconstrainedBelief

from oracle import all_states


class TestProbabilities:
    def test_eq1_by_hand(self, app_a_unconstrained):
        assert app_a_unconstrained(frozenset({"f2", "f3"})) == pytest.approx(0.6 * 0.7 * 0.9)
        assert app_a_unconstrained(frozenset()) == pytest.approx(0.6 * 0.3 * 0.1)

    def test_sums_to_one(self):
        belief = UnconstrainedBelief({"a": 0.2, "b": 0.5, "c": 0.99, "d": 0.0})
        total = math.fsum(belief(s) for s in all_states(belief.features))
        assert total == pytest.approx(1.0)

    def test_hard_evidence(self):
        belief = UnconstrainedBelief({"a": 1.0, "b": 0.3})
        assert belief(frozenset({"b"})) == 0.0
        assert belief(frozenset({"a"})) == pytest.approx(0.7)

    def test_empty_feature_set(self):
        belief = UnconstrainedBelief({})
        assert belief.features == frozenset()
        assert belief(frozenset()) == 1.0

    def test_accepts_set_and_numpy_values(self):
        belief = UnconstrainedBelief({"a": np.float32(0.25), "b": np.int64(1)})
        assert belief({"a", "b"}) == pytest.approx(0.25)


class TestValidation:
    @pytest.mark.parametrize("value", [True, False, np.bool_(True), "0.5", None])
    def test_rejects_non_numbers(self, value):
        with pytest.raises(TypeError):
            UnconstrainedBelief({"a": value})

    @pytest.mark.parametrize("value", [-0.1, 1.5, math.nan, math.inf])
    def test_rejects_out_of_range(self, value):
        with pytest.raises(ValueError):
            UnconstrainedBelief({"a": value})

    @pytest.mark.parametrize("key, error", [(1, TypeError), ("", ValueError)])
    def test_rejects_bad_feature_names(self, key, error):
        with pytest.raises(error):
            UnconstrainedBelief({key: 0.5})

    def test_rejects_non_mapping(self):
        with pytest.raises(TypeError):
            UnconstrainedBelief([("a", 0.5)])

    def test_rejects_dict_state(self, app_a_unconstrained):
        with pytest.raises(TypeError, match="frozenset"):
            app_a_unconstrained({"f1": True, "f2": False, "f3": True})

    @pytest.mark.parametrize("state", [["f1"], ("f1",), "f1"])
    def test_rejects_other_state_types(self, app_a_unconstrained, state):
        with pytest.raises(TypeError):
            app_a_unconstrained(state)

    def test_rejects_unknown_features(self, app_a_unconstrained):
        with pytest.raises(ValueError, match="unknown"):
            app_a_unconstrained(frozenset({"nope"}))


class TestImmutability:
    def test_input_mapping_is_copied(self):
        factored = {"a": 0.2}
        belief = UnconstrainedBelief(factored)
        factored["a"] = 0.9
        assert belief.factored_belief["a"] == 0.2

    def test_factored_belief_is_read_only(self, app_a_unconstrained):
        with pytest.raises(TypeError):
            app_a_unconstrained.factored_belief["f1"] = 0.5


class TestMostLikelyState:
    def test_most_likely_state(self, app_a_unconstrained):
        assert app_a_unconstrained.most_likely_state() == frozenset({"f2", "f3"})

    def test_ties_go_to_false(self):
        assert UnconstrainedBelief({"a": 0.5, "b": 0.51}).most_likely_state() == frozenset({"b"})


def test_products_do_not_depend_on_input_order():
    factored = {f"x{i}": 0.1 + 0.08 * i for i in range(10)}
    forward = UnconstrainedBelief(factored)
    backward = UnconstrainedBelief(dict(reversed(list(factored.items()))))
    state = frozenset({"x1", "x4", "x7"})
    assert forward(state) == backward(state)


class TestMarginal:
    def test_marginal_keeps_subset(self, app_a_unconstrained):
        marginal = app_a_unconstrained.marginal({"f1", "f3"})
        assert marginal.features == frozenset({"f1", "f3"})
        assert marginal(frozenset({"f3"})) == pytest.approx(0.6 * 0.9)

    def test_marginal_rejects_unknown(self, app_a_unconstrained):
        with pytest.raises(ValueError):
            app_a_unconstrained.marginal({"nope"})
