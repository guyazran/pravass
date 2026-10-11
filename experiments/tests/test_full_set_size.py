from __future__ import annotations

import unittest
from types import SimpleNamespace

from mbs_experiments.states import StateSet, constraint_satisfying_full_state_count


class FullSetSizeTests(unittest.TestCase):
    def test_counts_valid_null_extensions_for_explicit_reduced_states(self) -> None:
        state_set = StateSet(
            states=(
                (True, False),
                (False, True),
                (False, False),
            ),
            unconstrained_mass=0.0,
            constrained_mass=0.0,
            total_states_considered=3,
        )

        size = constraint_satisfying_full_state_count(
            state_set,
            full_feature_count=5,
            selected_groups=((0, 2, 3), (1, 4)),
            search_indices=(0, 1),
        )

        # (T,F): 1 extension for group (0,2,3), 2 for (1,4)
        # (F,T): 3 extensions for group (0,2,3), 1 for (1,4)
        # (F,F): 3 extensions for group (0,2,3), 2 for (1,4)
        self.assertEqual(size, 2 + 3 + 6)

    def test_template_count_can_filter_constraints_without_materializing(self) -> None:
        state_set = StateSet(
            states=(),
            unconstrained_mass=0.0,
            constrained_mass=0.0,
            total_states_considered=0,
            committed_template=(True, None),
        )

        size = constraint_satisfying_full_state_count(
            state_set,
            full_feature_count=5,
            selected_groups=((0, 2, 3), (1, 4)),
            search_indices=(0, 1),
        )

        # Feature 0 is committed true, so 2 and 3 must be false. Feature 1 and
        # null feature 4 are jointly at-most-one, giving {}, {1}, or {4}.
        self.assertEqual(size, 3)

    def test_committed_full_valid_count_matches_template_semantics(self) -> None:
        from mbs_experiments.abstention import count_committed_full_valid_states

        size = count_committed_full_valid_states(
            full_feature_count=5,
            selected_groups=((0, 2, 3), (1, 4)),
            search_indices=(0, 1),
            search_feature_count=2,
            commits={0: True},
        )

        self.assertEqual(size, 3)

    def test_tau_worker_returns_sample_payload_for_global_aggregation(self) -> None:
        from unittest.mock import patch

        from mbs_experiments import abstention

        sid = SimpleNamespace(split_relative_path="dom/cfg/sample", domain="dom")
        bundle = SimpleNamespace(
            feature_names=("a", "b"),
            probabilities=(0.95, 0.05),
            none_is_max=(False, False),
            selected_groups=(),
        )

        with patch.object(abstention, "load_json", return_value={}), patch.object(
            abstention, "available_calibrations", return_value=("global",)
        ), patch.object(
            abstention, "load_calibration_bundle", return_value=bundle
        ):
            with patch.object(abstention, "prediction_path") as pred_path_mock:
                pred_path_mock.return_value.exists.return_value = True
                result = abstention._tau_sample_worker(
                    (
                        sid,
                        "global",
                        "test-model",
                        "validation",
                        "/tmp",
                        "/tmp",
                        "/tmp",
                    )
                )

        kind, path, reason, payload = result
        self.assertEqual((kind, path, reason), ("used", "dom/cfg/sample", None))
        self.assertEqual(payload["sample"], "dom/cfg/sample")
        self.assertEqual(payload["probabilities"], (0.95, 0.05))
        self.assertEqual(payload["selected_groups"], ())
        self.assertGreater(payload["normalizer"], 0.0)
        self.assertGreater(len(payload["tau_candidates"]), 0)

    def test_unconstrained_top_mass_full_size_ignores_constraints(self) -> None:
        from scripts.run_experiment_1_static_quality import downstream_full_set_size

        bundle = SimpleNamespace(
            feature_names=("a", "b", "c"),
            selected_groups=((0, 1),),
        )
        state_set = StateSet(
            states=((True,),),
            unconstrained_mass=0.0,
            constrained_mass=0.0,
            total_states_considered=1,
        )

        constrained_size = downstream_full_set_size(
            bundle,
            "mbs_prevas",
            state_set,
            compact_set_size=1,
            omitted_feature_count=2,
            search_indices=(0,),
        )
        unconstrained_size = downstream_full_set_size(
            bundle,
            "unconstrained_top_mass",
            state_set,
            compact_set_size=1,
            omitted_feature_count=2,
            search_indices=(0,),
        )

        self.assertEqual(constrained_size, 2)
        self.assertEqual(unconstrained_size, 4)

    def test_invalid_visible_assignment_has_no_valid_full_extensions(self) -> None:
        state_set = StateSet(
            states=((True, True),),
            unconstrained_mass=0.0,
            constrained_mass=0.0,
            total_states_considered=1,
        )

        size = constraint_satisfying_full_state_count(
            state_set,
            full_feature_count=3,
            selected_groups=((0, 1, 2),),
            search_indices=(0, 1),
        )

        self.assertEqual(size, 0)


if __name__ == "__main__":
    unittest.main()
