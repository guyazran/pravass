"""MutexGroups and MutexNormalizer."""

import random
from fractions import Fraction

import pytest

from pravass import (
    EnumerationNormalizer,
    FunctionConstraint,
    MutexGroups,
    MutexNormalizer,
    Normalization,
    UnconstrainedBelief,
)

from oracle import exact_z


class TestMutexGroups:
    def test_validity(self, app_a_constraint):
        assert app_a_constraint(frozenset({"f1", "f3"}))
        assert not app_a_constraint(frozenset({"f1", "f2"}))

    def test_canonicalization(self):
        constraint = MutexGroups([["a", "b"], ("b", "a"), {"a", "b", "c"}, ["d"], []])
        assert constraint.groups == frozenset({frozenset({"a", "b", "c"})})

    def test_accepts_generators(self):
        constraint = MutexGroups(g for g in (iter(["a", "b"]), iter(["c", "d"])))
        assert constraint.groups == frozenset({frozenset("ab"), frozenset("cd")})

    def test_rejects_string_group(self):
        with pytest.raises(TypeError, match="string"):
            MutexGroups(["on(a,b)"])

    def test_rejects_string_groups(self):
        with pytest.raises(TypeError):
            MutexGroups("ab")

    def test_rejects_non_string_features(self):
        with pytest.raises(TypeError):
            MutexGroups([["a", 1]])

    def test_support(self):
        assert MutexGroups([["a", "b"], ["c", "d"]]).support == frozenset("abcd")

    def test_features_outside_state_are_ignored(self):
        assert MutexGroups([["a", "b", "zzz"]])(frozenset({"a"}))


class TestMutexNormalizer:
    def test_app_a(self, app_a_unconstrained, app_a_constraint):
        result = MutexNormalizer()(app_a_unconstrained, app_a_constraint)
        assert result.exact and result.value == pytest.approx(0.72)

    def test_disjoint_groups_match_enumeration(self):
        rng = random.Random(0)
        constraint = MutexGroups([["x0", "x1", "x2"], ["x3", "x4"], ["x5", "x6", "x7"]])
        for _ in range(50):
            belief = UnconstrainedBelief({f"x{i}": rng.random() for i in range(9)})
            closed_form = MutexNormalizer()(belief, constraint)
            assert closed_form.exact
            assert closed_form.value == pytest.approx(EnumerationNormalizer()(belief, constraint).value)

    def test_overlapping_groups_are_an_upper_bound(self):
        # Paper App. G.5: all marginals 0.5, groups {a,b} and {b,c}: exact Z = 5/8.
        belief = UnconstrainedBelief({"a": 0.5, "b": 0.5, "c": 0.5})
        result = MutexNormalizer()(belief, MutexGroups([["a", "b"], ["b", "c"]]))
        assert result.exact is False
        assert result.value == pytest.approx(0.75)

    def test_random_groups_never_underestimate(self):
        rng = random.Random(1)
        features = [f"x{i}" for i in range(7)]
        for _ in range(300):
            probabilities = {f: rng.choice([0.0, 0.5, 1.0, rng.random()]) for f in features}
            groups = [rng.sample(features, rng.randint(2, 4)) for _ in range(rng.randint(1, 4))]
            constraint = MutexGroups(groups)
            truth = exact_z(probabilities, constraint)
            result = MutexNormalizer()(UnconstrainedBelief(probabilities), constraint)
            assert Fraction(result.value) >= truth * (1 - Fraction(1, 10**12))
            if result.exact:
                assert result.value == pytest.approx(float(truth), rel=1e-12, abs=1e-15)

    def test_zero_probability_features_can_make_groups_disjoint(self):
        belief = UnconstrainedBelief({"a": 0.3, "b": 0.0, "c": 0.4})
        assert MutexNormalizer()(belief, MutexGroups([["a", "b"], ["b", "c"]])).exact

    def test_infeasible(self):
        belief = UnconstrainedBelief({"a": 1.0, "b": 1.0, "c": 0.5})
        assert MutexNormalizer()(belief, MutexGroups([["a", "b", "c"]])) == Normalization(0.0, exact=True)

    def test_subsumed_groups_stay_exact(self):
        belief = UnconstrainedBelief({"a": 0.3, "b": 0.6, "c": 0.2})
        assert MutexNormalizer()(belief, MutexGroups([["a", "b"], ["a", "b", "c"]])).exact

    def test_features_outside_belief_are_ignored(self):
        belief = UnconstrainedBelief({"a": 0.3, "b": 0.6})
        result = MutexNormalizer()(belief, MutexGroups([["a", "b", "zzz"]]))
        assert result.exact and result.value == pytest.approx(1 - 0.3 * 0.6)

    def test_selected_groups_clique(self):
        # Triangle of pairwise mutexes: the cover {a, b, c} is exact (Z = 0.5),
        # while the greedy cover only gives 0.75.
        belief = UnconstrainedBelief({"a": 0.5, "b": 0.5, "c": 0.5})
        constraint = MutexGroups([["a", "b"], ["b", "c"], ["a", "c"]])
        assert MutexNormalizer()(belief, constraint).value == pytest.approx(0.75)
        result = MutexNormalizer(selected_groups=[["a", "b", "c"]])(belief, constraint)
        assert result.exact and result.value == pytest.approx(0.5)

    def test_selected_groups_with_features_outside_belief(self):
        belief = UnconstrainedBelief({"a": 0.5, "b": 0.5, "c": 0.5})
        constraint = MutexGroups([["a", "b", "zzz"], ["b", "c", "zzz"], ["a", "c"]])
        result = MutexNormalizer(selected_groups=[["a", "b", "c"], ["zzz", "yyy"]])
        with pytest.raises(ValueError):  # yyy shares no group with zzz
            result(belief, constraint)
        normalization = MutexNormalizer(selected_groups=[["a", "b", "c"]])(belief, constraint)
        assert normalization.exact and normalization.value == pytest.approx(0.5)

    def test_independent_of_hash_seed(self):
        import subprocess
        import sys

        code = (
            "from pravass import *\n"
            "groups = [[f'g{i}x', f'g{i}y', f'g{i}z'] for i in range(20)]\n"
            "p = {f: 0.05 + 0.01 * (hash(f) % 1 + k % 7) for k, g in enumerate(groups) for f in g}\n"
            "print(repr(MutexNormalizer()(UnconstrainedBelief(p), MutexGroups(groups)).value))"
        )
        values = {
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True,
                env={"PYTHONHASHSEED": str(seed)},
            ).stdout
            for seed in range(5)
        }
        assert len(values) == 1

    def test_selected_groups_must_be_implied(self):
        belief = UnconstrainedBelief({"a": 0.5, "b": 0.5, "c": 0.5})
        constraint = MutexGroups([["a", "b"], ["b", "c"]])
        with pytest.raises(ValueError, match="share no mutex group"):
            MutexNormalizer(selected_groups=[["a", "b", "c"]])(belief, constraint)

    def test_selected_groups_must_be_disjoint(self):
        with pytest.raises(ValueError, match="disjoint"):
            MutexNormalizer(selected_groups=[["a", "b"], ["b", "c"]])

    def test_selected_groups_singletons_dropped(self):
        belief = UnconstrainedBelief({"a": 0.5, "b": 0.5, "c": 0.5})
        normalizer = MutexNormalizer(selected_groups=[["a"], ["b", "c"]])
        assert normalizer(belief, MutexGroups([["b", "c"]])).exact

    def test_wrong_types(self, app_a_unconstrained):
        with pytest.raises(TypeError, match="UnconstrainedBelief and MutexGroups"):
            MutexNormalizer()(app_a_unconstrained, FunctionConstraint(lambda s: True))

    def test_underflow(self):
        p = 1 - 1e-6
        groups = [[f"g{i}a", f"g{i}b"] for i in range(60)]
        belief = UnconstrainedBelief({f: p for group in groups for f in group})
        with pytest.raises(ValueError, match="underflows"):
            MutexNormalizer()(belief, MutexGroups(groups))
