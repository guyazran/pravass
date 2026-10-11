from __future__ import annotations

import unittest

from mbs_experiments.states import (
    conditional_at_most_one_normalizer,
    is_valid_state,
    selected_groups_normalizer,
    unconstrained_probability,
)


def brute_force_conditional_mass(
    probabilities, selected_groups, commits
):
    n = len(probabilities)
    total = 0.0
    for assignment in range(1 << n):
        state = tuple(bool((assignment >> i) & 1) for i in range(n))
        if any(state[i] != value for i, value in commits.items()):
            continue
        if not is_valid_state(state, selected_groups):
            continue
        total += unconstrained_probability(state, probabilities)
    return total


class ConditionalAtMostOneNormalizerTests(unittest.TestCase):
    def test_no_commits_matches_selected_groups_normalizer(self) -> None:
        probabilities = (0.7, 0.4, 0.3)
        groups = ((0, 1),)
        z = conditional_at_most_one_normalizer(probabilities, groups, {})
        expected = selected_groups_normalizer(probabilities, groups)
        self.assertAlmostEqual(z, expected, places=12)
        self.assertAlmostEqual(
            z, brute_force_conditional_mass(probabilities, groups, {}), places=12
        )

    def test_commit_grouped_true_forces_others_false(self) -> None:
        probabilities = (0.7, 0.4, 0.3)
        groups = ((0, 1),)
        z = conditional_at_most_one_normalizer(probabilities, groups, {0: True})
        # Group {0,1}: 0=True forces 1=False → 0.7 × 0.6 = 0.42
        # Lonely feature 2 integrates out → factor 1
        self.assertAlmostEqual(z, 0.42, places=12)
        self.assertAlmostEqual(
            z,
            brute_force_conditional_mass(probabilities, groups, {0: True}),
            places=12,
        )

    def test_commit_grouped_false_removes_from_at_most_one(self) -> None:
        probabilities = (0.7, 0.4, 0.3)
        groups = ((0, 1),)
        z = conditional_at_most_one_normalizer(probabilities, groups, {1: False})
        # Group {0,1} with 1=False: factor (1-0.4) × at_most_one({0}) = 0.6 × 1 = 0.6
        self.assertAlmostEqual(z, 0.6, places=12)
        self.assertAlmostEqual(
            z,
            brute_force_conditional_mass(probabilities, groups, {1: False}),
            places=12,
        )

    def test_two_committed_true_in_same_group_returns_zero(self) -> None:
        probabilities = (0.7, 0.4, 0.3)
        groups = ((0, 1),)
        z = conditional_at_most_one_normalizer(
            probabilities, groups, {0: True, 1: True}
        )
        self.assertEqual(z, 0.0)

    def test_lonely_committed_feature_contributes_its_factor(self) -> None:
        probabilities = (0.7, 0.4, 0.3)
        groups = ((0, 1),)
        z_true = conditional_at_most_one_normalizer(probabilities, groups, {2: True})
        z_false = conditional_at_most_one_normalizer(
            probabilities, groups, {2: False}
        )
        # at_most_one({0,1}) = 0.72; feature 2 contributes q_2 or 1-q_2.
        self.assertAlmostEqual(z_true, 0.72 * 0.3, places=12)
        self.assertAlmostEqual(z_false, 0.72 * 0.7, places=12)

    def test_two_disjoint_groups_factorize(self) -> None:
        probabilities = (0.7, 0.4, 0.3, 0.5)
        groups = ((0, 1), (2, 3))
        z = conditional_at_most_one_normalizer(probabilities, groups, {})
        self.assertAlmostEqual(
            z, brute_force_conditional_mass(probabilities, groups, {}), places=12
        )
        # Cross-checked: 0.72 × 0.85 = 0.612.
        self.assertAlmostEqual(z, 0.612, places=12)

    def test_commit_in_one_group_does_not_affect_other(self) -> None:
        probabilities = (0.7, 0.4, 0.3, 0.5)
        groups = ((0, 1), (2, 3))
        z = conditional_at_most_one_normalizer(
            probabilities, groups, {0: True, 3: False}
        )
        # Group {0,1}: 0.7 × 0.6 = 0.42. Group {2,3}: (1-0.5) × at_most_one({2}) = 0.5 × 1 = 0.5.
        self.assertAlmostEqual(z, 0.42 * 0.5, places=12)
        self.assertAlmostEqual(
            z,
            brute_force_conditional_mass(
                probabilities, groups, {0: True, 3: False}
            ),
            places=12,
        )

    def test_matches_brute_force_on_size_3_group(self) -> None:
        probabilities = (0.6, 0.3, 0.8, 0.2)
        groups = ((0, 1, 2),)
        for commits in (
            {},
            {0: False},
            {2: True},
            {1: False, 2: False},
            {3: True},
        ):
            with self.subTest(commits=commits):
                z = conditional_at_most_one_normalizer(probabilities, groups, commits)
                expected = brute_force_conditional_mass(probabilities, groups, commits)
                self.assertAlmostEqual(z, expected, places=12)

    def test_empty_groups_just_use_committed_factors(self) -> None:
        probabilities = (0.7, 0.4, 0.3)
        z = conditional_at_most_one_normalizer(
            probabilities, (), {0: True, 2: False}
        )
        # No constraints; uncommitted feature 1 integrates out (factor 1).
        self.assertAlmostEqual(z, 0.7 * 0.7, places=12)


if __name__ == "__main__":
    unittest.main()
