"""pravass(): the paper's worked example, guarantees against an exact oracle, and edge cases."""

import math
import random
from fractions import Fraction

import pytest

from pravass import (
    TOLERANCE,
    FunctionConstraint,
    InfeasibleConstraintError,
    MutexGroups,
    MutexNormalizer,
    StateLimitError,
    TaskBelief,
    TimeLimitError,
    UnconstrainedBelief,
    UnitNormalizer,
    iter_pravass,
    pravass,
)

from oracle import exact_z, minimal_size, ranked_valid, task_probability


def as_sorted(belief_set):
    return sorted(sorted(state) for state in belief_set)


class TestPaperExample:
    """App. A: β_F = [0.4, 0.7, 0.9], f1 and f2 mutually exclusive, Z = 0.72."""

    def test_theta_08(self, app_a_unconstrained, app_a_constraint):
        belief = TaskBelief(app_a_unconstrained, app_a_constraint)
        assert as_sorted(pravass(belief, 0.8)) == [["f1", "f3"], ["f2", "f3"], ["f3"]]

    @pytest.mark.parametrize(
        "theta, exact_size, unit_size",
        [(0.525, 1, 2), (0.75, 2, 3), (0.8, 3, 3), (0.9, 3, 4), (1.0, 6, 6)],
    )
    def test_sizes(self, app_a_unconstrained, app_a_constraint, theta, exact_size, unit_size):
        exact = pravass(TaskBelief(app_a_unconstrained, app_a_constraint), theta)
        closed_form = pravass(
            TaskBelief(app_a_unconstrained, app_a_constraint, normalizer=MutexNormalizer()), theta
        )
        relaxed = pravass(
            TaskBelief(app_a_unconstrained, app_a_constraint, normalizer=UnitNormalizer()), theta
        )
        assert len(exact) == len(closed_form) == exact_size
        assert len(relaxed) == unit_size
        assert exact <= relaxed


def _random_instance(rng, n):
    features = [f"x{i}" for i in range(n)]
    probabilities = {
        f: rng.choice([0.0, 1.0, 0.5, 1e-12, 1 - 1e-12, rng.random(), rng.random()])
        for f in features
    }
    if rng.random() < 0.5:
        groups = [rng.sample(features, rng.randint(2, min(4, n))) for _ in range(rng.randint(1, 3))]
        constraint = MutexGroups(groups)
    else:
        banned = {frozenset(rng.sample(features, 2)) for _ in range(rng.randint(1, 3))}
        constraint = FunctionConstraint(lambda s, banned=banned: not any(b <= s for b in banned))
    return probabilities, constraint


@pytest.mark.parametrize("normalizer", [None, MutexNormalizer(), UnitNormalizer()], ids=["exact", "mutex", "unit"])
def test_against_exact_oracle(normalizer):
    """Coverage >= θ - TOLERANCE (exact arithmetic), top-ness, and minimality when exact."""
    rng = random.Random(7)
    checked = 0
    for _ in range(400):
        probabilities, constraint = _random_instance(rng, rng.randint(2, 7))
        if isinstance(normalizer, MutexNormalizer) and not isinstance(constraint, MutexGroups):
            continue
        ranked = ranked_valid(probabilities, constraint)
        unconstrained = UnconstrainedBelief(probabilities)
        if not ranked:
            with pytest.raises(InfeasibleConstraintError):
                belief = TaskBelief(unconstrained, constraint, normalizer=normalizer)
                pravass(belief, 0.5)
            continue
        belief = TaskBelief(unconstrained, constraint, normalizer=normalizer)
        theta = rng.choice([1e-12, 0.3, 0.5, 0.9, 0.99, 1 - 1e-12, 1.0, rng.random()])
        result = pravass(belief, theta)

        z = exact_z(probabilities, constraint)
        selected = [task_probability(probabilities, constraint, s, z) for s in result]
        assert sum(selected, Fraction(0)) >= Fraction(theta) - Fraction(TOLERANCE)
        assert all(constraint(s) for s in result)
        # Top-ness: nothing left out is more probable than anything selected.
        excluded = [p for p, s in ranked if s not in result]
        if selected and excluded:
            assert max(excluded) <= min(selected) * (1 + Fraction(1, 10**9))
        if belief.normalization.exact:
            assert len(result) <= minimal_size(probabilities, constraint, theta)
            assert len(result) >= minimal_size(probabilities, constraint, max(0.0, theta - TOLERANCE))
        checked += 1
    assert checked > 100


def test_near_deterministic_marginals_keep_coverage():
    # A confidently wrong mutex group: the exact normalizer is tiny (3e-24).
    p = 1 - 1e-12
    probabilities = {"a": p, "b": p, "c": p}
    constraint = MutexGroups([["a", "b", "c"]])
    for normalizer in (None, MutexNormalizer(), UnitNormalizer()):
        belief = TaskBelief(UnconstrainedBelief(probabilities), constraint, normalizer=normalizer)
        result = pravass(belief, 0.9)
        coverage = sum((task_probability(probabilities, constraint, s) for s in result), Fraction(0))
        assert coverage >= Fraction(9, 10) - Fraction(TOLERANCE)


def test_unconstrained_belief_alone():
    belief = UnconstrainedBelief({"a": 0.9, "b": 0.8})
    assert as_sorted(pravass(belief, 0.7)) == [["a", "b"]]
    assert len(pravass(belief, 1.0)) == 4


def test_theta_zero_is_empty(app_a_unconstrained):
    assert pravass(app_a_unconstrained, 0.0) == frozenset()


@pytest.mark.parametrize("theta, error", [(-0.1, ValueError), (1.1, ValueError), (math.nan, ValueError), (True, TypeError)])
def test_invalid_theta(app_a_unconstrained, theta, error):
    with pytest.raises(error):
        pravass(app_a_unconstrained, theta)


def test_infeasible_with_unit_normalizer():
    belief = TaskBelief(
        UnconstrainedBelief({"a": 0.5, "b": 0.5}),
        FunctionConstraint(lambda s: False),
        normalizer=UnitNormalizer(),
    )
    for theta in (1e-12, 0.5, 1.0):
        with pytest.raises(InfeasibleConstraintError):
            pravass(belief, theta)


def test_infeasible_with_hard_evidence_and_invalid_start():
    belief = TaskBelief(
        UnconstrainedBelief({"a": 1.0, "b": 1.0}),
        FunctionConstraint(lambda s: not {"a", "b"} <= s),
        normalizer=UnitNormalizer(),
    )
    with pytest.raises(InfeasibleConstraintError):
        pravass(belief, 0.5)


class TestStateLimit:
    def test_boundary(self):
        # Two states answer theta = 0.85; the start state and its first neighbour fit in 3 insertions.
        belief = UnconstrainedBelief({"a": 0.9, "b": 0.9})
        with pytest.raises(StateLimitError):
            pravass(belief, 0.85, max_states=2)
        assert len(pravass(belief, 0.85, max_states=3)) == 2

    def test_mentions_possible_infeasibility(self):
        belief = TaskBelief(
            UnconstrainedBelief({"a": 0.5, "b": 0.5, "c": 0.5}),
            FunctionConstraint(lambda s: False),
            normalizer=UnitNormalizer(),
        )
        with pytest.raises(StateLimitError, match="constraint can be satisfied"):
            pravass(belief, 0.5, max_states=2)

    def test_one_state_suffices(self):
        belief = UnconstrainedBelief({"a": 0.99, "b": 0.99})
        assert len(pravass(belief, 0.9, max_states=1)) == 1

    def test_limit_reached(self):
        belief = UnconstrainedBelief({"a": 0.6, "b": 0.6})
        with pytest.raises(StateLimitError):
            pravass(belief, 0.9, max_states=2)
        assert len(pravass(belief, 0.9, max_states=None)) == 4
        assert len(pravass(belief, 0.9, max_states=math.inf)) == 4

    def test_message_explains_ties(self):
        belief = UnconstrainedBelief({f"x{i}": 0.5 for i in range(12)})
        with pytest.raises(StateLimitError, match="12 features have probability exactly 0.5"):
            pravass(belief, 0.9, max_states=1000)
        assert len(pravass(belief, 0.9)) == math.ceil(0.9 * 4096)

    def test_message_explains_neighbour_insertions(self):
        belief = UnconstrainedBelief({"a": 0.6, "b": 0.6})
        with pytest.raises(StateLimitError, match="one neighbour per feature"):
            pravass(belief, 0.9, max_states=2)

    def test_message_omits_ties_when_none(self):
        belief = UnconstrainedBelief({"a": 0.6, "b": 0.6})
        with pytest.raises(StateLimitError) as info:
            pravass(belief, 0.9, max_states=2)
        assert "0.5" not in str(info.value)


def test_rejects_unsupported_beliefs(app_a_unconstrained, app_a_constraint):
    nested = TaskBelief(TaskBelief(app_a_unconstrained, app_a_constraint), FunctionConstraint(lambda s: True))
    with pytest.raises(TypeError, match="single constraint"):
        pravass(nested, 0.5)
    with pytest.raises(TypeError):
        pravass({"f1": 0.5}, 0.5)


def test_repeat_calls_are_identical(app_a_unconstrained, app_a_constraint):
    belief = TaskBelief(app_a_unconstrained, app_a_constraint)
    assert pravass(belief, 0.9) == pravass(belief, 0.9)


def test_too_diffuse_belief():
    belief = UnconstrainedBelief({f"x{i}": 0.5 for i in range(1100)})
    with pytest.raises(ValueError, match="diffuse"):
        pravass(belief, 0.5)


def test_flip_costs_are_accurate_near_one_half():
    from pravass.search import _flip_cost

    p = 0.5 + 1e-12
    assert _flip_cost(p) == pytest.approx(math.log(p / (1 - p)), rel=1e-9)
    assert _flip_cost(0.0) == _flip_cost(1.0) == math.inf


class TestRunningBound:
    """Regression instances for the inexact-normalizer stopping rule."""

    def test_bound_tightens_on_invalid_states(self):
        # Stopping is checked after every popped state, valid or not.
        probabilities = {"a": 0.3, "b": 0.9, "c": 0.7, "d": 0.2}
        constraint = MutexGroups([["d", "a"], ["d", "b"]])
        belief = TaskBelief(UnconstrainedBelief(probabilities), constraint, normalizer=UnitNormalizer())
        result = pravass(belief, 0.9)
        assert len(result) == 5
        assert sum(task_probability(probabilities, constraint, s) for s in result) >= Fraction(9, 10)

    def test_bound_never_exceeds_the_given_normalizer(self):
        probabilities = {"a": 0.3, "b": 0.6, "c": 0.7, "d": 0.1}
        constraint = MutexGroups([["d", "b"], ["b", "c"]])
        belief = TaskBelief(UnconstrainedBelief(probabilities), constraint, normalizer=MutexNormalizer())
        assert not belief.normalization.exact
        result = pravass(belief, 0.8)
        assert len(result) == 5
        assert sum(task_probability(probabilities, constraint, s) for s in result) >= Fraction(8, 10)


def test_against_exact_oracle_at_mass_boundaries():
    """θ placed exactly on cumulative masses, with supports and selected covers."""
    rng = random.Random(11)
    checked = 0
    for _ in range(150):
        n = rng.randint(2, 6)
        features = [f"x{i}" for i in range(n)]
        probabilities = {f: rng.choice([0.0, 1.0, 0.5, rng.random(), rng.random()]) for f in features}
        if rng.random() < 0.5:
            pair = rng.sample(features, 2)
            constraint = FunctionConstraint(lambda s, pair=frozenset(pair): not pair <= s, support=pair)
            normalizer = None
        else:
            triangle = rng.sample(features, min(3, n))
            groups = [[a, b] for i, a in enumerate(triangle) for b in triangle[i + 1:]] + [["outside", features[0]]]
            constraint = MutexGroups(groups)
            normalizer = MutexNormalizer(selected_groups=[triangle] if len(triangle) == 3 else None)
        ranked = ranked_valid(probabilities, constraint)
        if not ranked:
            continue
        belief = TaskBelief(UnconstrainedBelief(probabilities), constraint, normalizer=normalizer)
        cumulative = Fraction(0)
        for p, _ in ranked:
            cumulative += p
            theta = float(cumulative)
            result = pravass(belief, theta)
            z = exact_z(probabilities, constraint)
            mass = sum((task_probability(probabilities, constraint, s, z) for s in result), Fraction(0))
            assert mass >= Fraction(theta) - Fraction(TOLERANCE)
            if belief.normalization.exact:
                assert len(result) <= minimal_size(probabilities, constraint, theta)
            checked += 1
    assert checked > 100


class TestIterPravass:
    """iter_pravass(): the same search, one state at a time."""

    @pytest.mark.parametrize("normalizer", [None, MutexNormalizer(), UnitNormalizer()], ids=["exact", "mutex", "unit"])
    def test_matches_pravass_and_ranks_states(self, normalizer):
        rng = random.Random(3)
        checked = 0
        for _ in range(300):
            probabilities, constraint = _random_instance(rng, rng.randint(2, 7))
            if isinstance(normalizer, MutexNormalizer) and not isinstance(constraint, MutexGroups):
                continue
            ranked = ranked_valid(probabilities, constraint)
            if not ranked:
                continue
            belief = TaskBelief(UnconstrainedBelief(probabilities), constraint, normalizer=normalizer)
            theta = rng.choice([0.3, 0.5, 0.9, 0.99, 1.0, rng.random()])
            assert frozenset(iter_pravass(belief, theta)) == pravass(belief, theta)

            states = list(iter_pravass(belief))
            assert set(states) == {s for _, s in ranked}
            z = exact_z(probabilities, constraint)
            masses = [task_probability(probabilities, constraint, s, z) for s in states]
            assert all(a >= b * (1 - Fraction(1, 10**9)) for a, b in zip(masses, masses[1:]))
            checked += 1
        assert checked > 50

    def test_mass_and_counts(self, app_a_unconstrained, app_a_constraint):
        belief = TaskBelief(app_a_unconstrained, app_a_constraint)
        search = iter_pravass(belief)
        assert search.mass == 0.0 and search.examined == 0 and search.explored == 1
        first = next(search)
        assert first == frozenset({"f2", "f3"})
        assert search.mass == pytest.approx(0.525)
        rest = list(search)
        assert len(rest) == 5
        assert search.mass == pytest.approx(1.0)
        assert search.examined == search.explored == 8  # every state of three features

    def test_theta_stops_at_the_minimal_set(self, app_a_unconstrained, app_a_constraint):
        belief = TaskBelief(app_a_unconstrained, app_a_constraint)
        search = iter_pravass(belief, theta=0.7)
        assert [sorted(s) for s in search] == [["f2", "f3"], ["f3"]]
        assert search.mass == pytest.approx(0.75)

    def test_unconstrained_belief(self):
        belief = UnconstrainedBelief({"a": 0.9, "b": 0.2})
        assert [sorted(s) for s in iter_pravass(belief)] == [["a"], ["a", "b"], [], ["b"]]

    def test_theta_zero_yields_nothing(self, app_a_unconstrained):
        assert list(iter_pravass(app_a_unconstrained, 0.0)) == []

    def test_infeasible(self):
        belief = TaskBelief(
            UnconstrainedBelief({"a": 1.0, "b": 1.0}), MutexGroups([["a", "b"]]), normalizer=UnitNormalizer()
        )
        with pytest.raises(InfeasibleConstraintError):
            list(iter_pravass(belief))

    def test_state_limit(self):
        belief = UnconstrainedBelief({f"x{i}": 0.5 for i in range(20)})
        with pytest.raises(StateLimitError, match="enumerating the valid states"):
            list(iter_pravass(belief, max_states=10))

    def test_time_limit_keeps_the_states_found(self):
        belief = UnconstrainedBelief({f"x{i}": 0.5 + i / 100 for i in range(30)})
        search = iter_pravass(belief, time_limit=1e-9)
        with pytest.raises(TimeLimitError, match="time limit"):
            next(search)
        found = []
        search = iter_pravass(belief, time_limit=0.05)
        with pytest.raises(TimeLimitError):
            for state in search:
                found.append(state)
        assert found and found[0] == belief.most_likely_state()
        assert search.mass > 0.0

    def test_ties_start_false_and_break_on_the_first_feature(self):
        belief = UnconstrainedBelief({"a": 0.5, "b": 0.5, "c": 0.5})
        assert [sorted(s) for s in iter_pravass(belief)] == [
            [], ["c"], ["b"], ["b", "c"], ["a"], ["a", "c"], ["a", "b"], ["a", "b", "c"]
        ]

    def test_tied_alternatives_come_last_feature_first(self):
        belief = TaskBelief(
            UnconstrainedBelief({"a": 0.5, "b": 0.5, "c": 0.5}),
            FunctionConstraint(lambda state: len(state) == 1),
            normalizer=UnitNormalizer(),
        )
        assert [sorted(s) for s in iter_pravass(belief)] == [["c"], ["b"], ["a"]]

    @pytest.mark.parametrize("time_limit, error", [(0, ValueError), (-1.0, ValueError), (True, TypeError), ("1", TypeError)])
    def test_invalid_time_limit(self, app_a_unconstrained, time_limit, error):
        with pytest.raises(error):
            iter_pravass(app_a_unconstrained, time_limit=time_limit)
