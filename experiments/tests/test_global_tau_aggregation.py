from __future__ import annotations

import unittest

from mbs_experiments.abstention import _aggregate_global_tau
from mbs_experiments.baselines import (
    adaptive_fluent_abstention_tau_candidates,
    fluent_abstention_unconstrained,
    fluent_prediction_marginals,
)
from mbs_experiments.states import selected_groups_normalizer


def _sample_payload(probabilities, selected_groups=()):
    z = selected_groups_normalizer(probabilities, selected_groups)
    candidates = adaptive_fluent_abstention_tau_candidates(
        fluent_prediction_marginals(probabilities)
    )
    return {
        "probabilities": tuple(probabilities),
        "selected_groups": tuple(tuple(g) for g in selected_groups),
        "normalizer": z,
        "tau_candidates": tuple(candidates),
    }


def _mean_mass_at(samples, tau):
    masses = []
    for sample in samples:
        result = fluent_abstention_unconstrained(
            sample["probabilities"],
            tau,
            sample["normalizer"],
            selected_groups=sample["selected_groups"],
        )
        masses.append(result.constrained_mass)
    return sum(masses) / len(masses) if masses else 0.0


class GlobalTauAggregationTests(unittest.TestCase):
    def test_returns_smallest_global_candidate_where_mean_mass_meets_theta(self) -> None:
        s1 = _sample_payload((0.95, 0.7, 0.55))
        s2 = _sample_payload((0.6, 0.85, 0.4))
        samples = [s1, s2]
        theta = 0.5

        selected, mean_at_selected, history = _aggregate_global_tau(samples, theta)

        self.assertIsNotNone(selected)
        # Selected τ must satisfy θ.
        self.assertGreaterEqual(mean_at_selected + 1e-12, theta)
        # And no smaller global candidate does.
        global_candidates = sorted(
            {tau for s in samples for tau in s["tau_candidates"]}
        )
        for tau in global_candidates:
            if tau >= selected:
                break
            self.assertLess(_mean_mass_at(samples, tau), theta)

    def test_unreachable_theta_returns_none(self) -> None:
        # At τ = 1.0 (top candidate) constrained mass = 1 for both samples.
        # θ = 1.0 + ε is unreachable.
        s1 = _sample_payload((0.7, 0.4))
        s2 = _sample_payload((0.95, 0.05))
        selected, mean_at_selected, _ = _aggregate_global_tau([s1, s2], 1.0 + 1e-3)
        self.assertIsNone(selected)
        self.assertIsNone(mean_at_selected)

    def test_all_samples_satisfy_at_smallest_candidate(self) -> None:
        # Each sample's smallest candidate (= its smallest confidence) already
        # gives mass ≥ θ, so the returned τ should be the smallest one in the
        # global union.
        s1 = _sample_payload((0.95, 0.95))  # very confident
        s2 = _sample_payload((0.99, 0.99))
        theta = 0.0
        selected, mean_at_selected, _ = _aggregate_global_tau([s1, s2], theta)
        global_smallest = min(
            tau for s in (s1, s2) for tau in s["tau_candidates"]
        )
        self.assertEqual(selected, global_smallest)

    def test_mean_mass_curve_is_monotone_nondecreasing(self) -> None:
        s1 = _sample_payload((0.7, 0.4, 0.85), selected_groups=((0, 1),))
        s2 = _sample_payload((0.55, 0.95, 0.3), selected_groups=((0, 2),))
        _, _, history = _aggregate_global_tau([s1, s2], theta=0.0)
        masses = [row["mean_mass"] for row in history]
        for prev, curr in zip(masses, masses[1:]):
            self.assertLessEqual(prev - 1e-12, curr)

    def test_empty_samples_returns_none(self) -> None:
        selected, mean_at_selected, history = _aggregate_global_tau([], 0.5)
        self.assertIsNone(selected)
        self.assertIsNone(mean_at_selected)
        self.assertEqual(history, [])

    def test_single_sample_matches_per_sample_tau_star(self) -> None:
        # With one sample, the global mean is the sample's mass, so the
        # smallest-mean-≥-θ τ equals the sample's per-sample τ*.
        from mbs_experiments.baselines import per_sample_tau_star

        probs = (0.7, 0.55, 0.85)
        groups = ()
        z = selected_groups_normalizer(probs, groups)
        theta = 0.4
        sample_tau_star = per_sample_tau_star(probs, groups, z, theta)
        selected, _, _ = _aggregate_global_tau([_sample_payload(probs, groups)], theta)
        self.assertEqual(selected, sample_tau_star)


if __name__ == "__main__":
    unittest.main()
