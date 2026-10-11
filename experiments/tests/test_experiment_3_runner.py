from __future__ import annotations

import unittest
from types import SimpleNamespace

from mbs_experiments.planning import PlanningResult
from mbs_experiments.search import SearchResult, SearchStats
from mbs_experiments.states import StateSet


class Experiment3RunnerBaselineTests(unittest.TestCase):
    def test_stale_singleton_records_are_recomputed_when_they_omitted_features(self) -> None:
        from scripts.run_experiment_3_planning_utility import stale_singleton_baseline_keys

        stale_key = ("prb", "h01_n03", "sample_000001", "global", "map", None)
        fresh_key = ("gw", "h01", "sample_000001", "global", "valid_map", None)
        theta_key = ("prb", "h01_n03", "sample_000001", "global", "mbs_prevas", 0.9)
        records = {
            stale_key: {"method": "map", "omitted_feature_count": 12},
            fresh_key: {"method": "valid_map", "omitted_feature_count": 0},
            theta_key: {"method": "mbs_prevas", "omitted_feature_count": 12},
        }

        self.assertEqual(stale_singleton_baseline_keys(records), {stale_key})

    def test_singleton_baselines_use_all_features_including_none_features(self) -> None:
        from scripts.run_experiment_3_planning_utility import singleton_baseline_inputs

        bundle = SimpleNamespace(
            feature_names=("visible-a", "none-b", "visible-c"),
            probabilities=(0.9, 0.5, 0.2),
            none_is_max=(False, True, False),
            selected_groups=((0, 1, 2),),
        )

        feature_names, probabilities, groups, normalizer, omitted_feature_names = (
            singleton_baseline_inputs(bundle)
        )

        self.assertEqual(feature_names, bundle.feature_names)
        self.assertEqual(probabilities, bundle.probabilities)
        self.assertEqual(groups, bundle.selected_groups)
        self.assertEqual(omitted_feature_names, ())
        self.assertGreater(normalizer, 0.0)

    def test_failure_pruning_does_not_censor_random_sweeps(self) -> None:
        from scripts.run_experiment_3_planning_utility import failure_pruning_enabled

        self.assertTrue(failure_pruning_enabled("mbs_prevas"))
        self.assertFalse(failure_pruning_enabled("random_theta"))

    def test_prevas_prefix_search_finds_largest_plannable_prefix(self) -> None:
        import scripts.run_experiment_3_planning_utility as runner

        states = tuple(
            tuple(bool((i >> bit) & 1) for bit in range(4))
            for i in range(10)
        )
        attempts: list[int] = []

        def fake_mbs_prevas(*args, **kwargs):
            return SearchResult(
                state_set=StateSet(
                    states=states,
                    unconstrained_mass=1.0,
                    constrained_mass=1.0,
                    total_states_considered=10,
                    status="ok",
                ),
                stats=SearchStats(10, 10, 10, 0, 1.0, 0.0, False),
            )

        def fake_run_belief_planning(*args, **kwargs):
            belief_states = args[3]
            size = len(belief_states)
            attempts.append(size)
            if 0 < size <= 6:
                return PlanningResult(
                    "ok", "FAKE", 0.1, 1, ("act",), True, False, False, False
                )
            return PlanningResult(
                "no_plan", None, 0.1, None, (), False, False, False, True
            )

        original_mbs_prevas = runner.mbs_prevas
        original_run_belief_planning = runner.run_belief_planning
        runner.mbs_prevas = fake_mbs_prevas
        runner.run_belief_planning = fake_run_belief_planning
        probabilities = (0.6, 0.6, 0.6, 0.6)
        try:
            outcome = runner.run_mbs_prevas_prefix_search_planning(
                SimpleNamespace(conditional_domain_path="domain", problem_path="problem"),
                probabilities,
                (),
                1.0,
                ("a", "b", "c", "d"),
                (),
                method_timeout=30.0,
                planning_timeout=30.0,
                oracle_success=True,
            )
        finally:
            runner.mbs_prevas = original_mbs_prevas
            runner.run_belief_planning = original_run_belief_planning

        self.assertEqual(attempts, [10, 5, 7, 6])
        expected_mass = sum(
            runner.unconstrained_probability(state, probabilities)
            for state in states[:6]
        )
        self.assertEqual(outcome.belief_size, 6)
        self.assertEqual(outcome.result.status, "ok")
        self.assertTrue(outcome.extra["prevas_prefix_search_plan_found"])
        self.assertEqual(outcome.extra["final_plan_set_size"], 6)
        self.assertAlmostEqual(
            outcome.extra["final_plan_set_unconstrained_mass"],
            expected_mass,
        )
        self.assertAlmostEqual(
            outcome.extra["final_plan_set_constrained_mass"],
            expected_mass,
        )

    def test_prevas_prefix_search_accepts_prevas_timeout_prefix(self) -> None:
        import scripts.run_experiment_3_planning_utility as runner

        states = (
            (False, False),
            (True, False),
            (False, True),
        )
        attempts: list[int] = []

        def fake_mbs_prevas(*args, **kwargs):
            return SearchResult(
                state_set=StateSet(
                    states=states,
                    unconstrained_mass=0.5,
                    constrained_mass=0.5,
                    total_states_considered=3,
                    status="timeout",
                ),
                stats=SearchStats(3, 3, 3, 0, 0.5, 30.0, True),
            )

        def fake_run_belief_planning(*args, **kwargs):
            attempts.append(len(args[3]))
            return PlanningResult(
                "ok", "FAKE", 0.1, 1, ("act",), True, False, False, False
            )

        original_mbs_prevas = runner.mbs_prevas
        original_run_belief_planning = runner.run_belief_planning
        runner.mbs_prevas = fake_mbs_prevas
        runner.run_belief_planning = fake_run_belief_planning
        try:
            outcome = runner.run_mbs_prevas_prefix_search_planning(
                SimpleNamespace(conditional_domain_path="domain", problem_path="problem"),
                (0.6, 0.4),
                (),
                1.0,
                ("a", "b"),
                (),
                method_timeout=30.0,
                planning_timeout=30.0,
                oracle_success=True,
            )
        finally:
            runner.mbs_prevas = original_mbs_prevas
            runner.run_belief_planning = original_run_belief_planning

        self.assertEqual(attempts, [3])
        self.assertEqual(outcome.belief_size, 3)
        self.assertEqual(outcome.result.status, "ok")
        self.assertEqual(outcome.extra["final_plan_set_size"], 3)
        self.assertEqual(
            outcome.extra["prevas_prefix_search_prevas_status"],
            "timeout",
        )


if __name__ == "__main__":
    unittest.main()
