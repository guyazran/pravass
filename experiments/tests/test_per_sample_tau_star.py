from __future__ import annotations

import unittest

from mbs_experiments.baselines import (
    adaptive_fluent_abstention_tau_candidates,
    fluent_abstention_unconstrained,
    fluent_prediction_marginals,
    per_sample_tau_star,
)
from mbs_experiments.states import selected_groups_normalizer


class PerSampleTauStarTests(unittest.TestCase):
    def test_returns_smallest_candidate_meeting_theta(self) -> None:
        probabilities = (0.95, 0.7, 0.55)
        groups = ((0, 1),)
        z = selected_groups_normalizer(probabilities, groups)
        theta = 0.5

        tau_star = per_sample_tau_star(probabilities, groups, z, theta)
        self.assertIsNotNone(tau_star)

        # Mass at tau_star meets theta...
        mass = fluent_abstention_unconstrained(
            probabilities, tau_star, z, selected_groups=groups
        ).constrained_mass
        self.assertGreaterEqual(mass + 1e-12, theta)

        # ...and no smaller candidate does.
        candidates = adaptive_fluent_abstention_tau_candidates(
            fluent_prediction_marginals(probabilities)
        )
        smaller = [tau for tau in candidates if tau < tau_star]
        for tau in smaller:
            with self.subTest(tau=tau):
                smaller_mass = fluent_abstention_unconstrained(
                    probabilities, tau, z, selected_groups=groups
                ).constrained_mass
                self.assertLess(smaller_mass, theta)

    def test_high_theta_drives_tau_to_one(self) -> None:
        probabilities = (0.7, 0.4, 0.55)
        groups = ((0, 1),)
        z = selected_groups_normalizer(probabilities, groups)
        # θ = 0.99 forces τ to (almost) the top candidate, which yields mass = 1.
        tau_star = per_sample_tau_star(probabilities, groups, z, 0.99)
        self.assertIsNotNone(tau_star)
        self.assertEqual(tau_star, 1.0)

    def test_low_theta_picks_smallest_candidate_when_already_satisfied(self) -> None:
        # If the lowest candidate already exceeds θ, pick it.
        probabilities = (0.95, 0.05)  # very confident
        groups = ()
        z = 1.0
        tau_star = per_sample_tau_star(probabilities, groups, z, 0.5)
        candidates = adaptive_fluent_abstention_tau_candidates(
            fluent_prediction_marginals(probabilities)
        )
        self.assertEqual(tau_star, candidates[0])

    def test_invalid_normalizer_returns_none(self) -> None:
        self.assertIsNone(
            per_sample_tau_star((0.5, 0.5), (), normalizer=0.0, theta=0.9)
        )

    def test_candidates_for_uniform_marginals_collapse(self) -> None:
        # Identical confidences should produce a small, sorted, unique candidate set.
        candidates = adaptive_fluent_abstention_tau_candidates(
            fluent_prediction_marginals((0.7, 0.7, 0.7))
        )
        self.assertEqual(candidates, (0.7, 1.0))


if __name__ == "__main__":
    unittest.main()
