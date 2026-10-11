from __future__ import annotations

import unittest

from mbs_experiments.baselines import fluent_abstention_unconstrained
from mbs_experiments.states import (
    is_valid_state,
    selected_groups_normalizer,
    unconstrained_probability,
)


def brute_force_valid_mass(probabilities, selected_groups, commits):
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


def expected_commits(probabilities, tau):
    commits = {}
    for i, q in enumerate(probabilities):
        if q >= tau:
            commits[i] = True
        elif q <= 1.0 - tau:
            commits[i] = False
    return commits


class FluentAbstentionUnconstrainedMassTests(unittest.TestCase):
    def test_mass_filters_invalid_completions(self) -> None:
        probabilities = (0.95, 0.85, 0.4)  # both first two committed at tau=0.8
        groups = ((0, 1),)  # mutex
        z = selected_groups_normalizer(probabilities, groups)
        result = fluent_abstention_unconstrained(
            probabilities, tau=0.8, normalizer=z, selected_groups=groups
        )
        # tau=0.8 commits 0=True and 1=True; that violates the mutex, so the
        # valid intersection is empty → mass = 0.
        commits = expected_commits(probabilities, 0.8)
        self.assertEqual(commits, {0: True, 1: True})
        expected_mass = brute_force_valid_mass(probabilities, groups, commits)
        self.assertEqual(expected_mass, 0.0)
        self.assertAlmostEqual(result.unconstrained_mass, 0.0, places=12)
        self.assertAlmostEqual(result.constrained_mass, 0.0, places=12)

    def test_constrained_mass_is_at_most_one(self) -> None:
        # The previous bug: ignoring validity could make constrained_mass > 1.
        probabilities = (0.6, 0.4, 0.55)
        groups = ((0, 1, 2),)
        z = selected_groups_normalizer(probabilities, groups)
        for tau in (0.55, 0.7, 0.9, 0.99):
            with self.subTest(tau=tau):
                result = fluent_abstention_unconstrained(
                    probabilities, tau=tau, normalizer=z, selected_groups=groups
                )
                self.assertGreaterEqual(result.constrained_mass, 0.0)
                self.assertLessEqual(result.constrained_mass, 1.0 + 1e-12)

    def test_mass_matches_brute_force_across_taus(self) -> None:
        probabilities = (0.7, 0.4, 0.3, 0.85)
        groups = ((0, 1), (2, 3))
        z = selected_groups_normalizer(probabilities, groups)
        for tau in (0.55, 0.6, 0.7, 0.85, 0.99):
            with self.subTest(tau=tau):
                commits = expected_commits(probabilities, tau)
                expected = brute_force_valid_mass(probabilities, groups, commits)
                result = fluent_abstention_unconstrained(
                    probabilities, tau=tau, normalizer=z, selected_groups=groups
                )
                self.assertAlmostEqual(result.unconstrained_mass, expected, places=12)
                self.assertAlmostEqual(result.constrained_mass, expected / z, places=12)

    def test_no_groups_recovers_old_behavior(self) -> None:
        # With no constraints every completion is valid, so the new mass should
        # equal the product-of-committed-factors used previously.
        probabilities = (0.7, 0.4, 0.3)
        z = 1.0
        for tau in (0.55, 0.7, 0.85):
            with self.subTest(tau=tau):
                commits = expected_commits(probabilities, tau)
                product = 1.0
                for i, value in commits.items():
                    q = probabilities[i]
                    product *= q if value else 1.0 - q
                result = fluent_abstention_unconstrained(
                    probabilities, tau=tau, normalizer=z, selected_groups=()
                )
                self.assertAlmostEqual(result.unconstrained_mass, product, places=12)

    def test_committed_template_still_reflects_all_completions(self) -> None:
        # The output set semantics (size = 2^k) are unchanged — only mass
        # accounting changed. Validate the committed_template is preserved.
        probabilities = (0.95, 0.4, 0.3)
        groups = ((0, 1),)
        z = selected_groups_normalizer(probabilities, groups)
        result = fluent_abstention_unconstrained(
            probabilities, tau=0.8, normalizer=z, selected_groups=groups
        )
        self.assertEqual(result.committed_template, (True, None, None))


if __name__ == "__main__":
    unittest.main()
