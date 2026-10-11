from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass

from mbs_experiments.states import (
    exact_at_most_one_normalizer,
    project_groups,
    selected_groups_normalizer,
    visible_indices,
)


@dataclass(frozen=True)
class FakeBundle:
    probabilities: tuple[float, ...]
    none_is_max: tuple[bool, ...]
    selected_groups: tuple[tuple[int, ...], ...]


class RuntimeNormalizerTests(unittest.TestCase):
    def test_visible_at_most_one_matches_old_reduced_computation(self) -> None:
        from mbs_experiments.normalizers import selected_groups_runtime_normalizer

        bundle = FakeBundle(
            probabilities=(0.7, 0.4, 0.2),
            none_is_max=(False, True, False),
            selected_groups=((0, 1, 2),),
        )
        idx = visible_indices(bundle.none_is_max)
        reduced_probs = tuple(bundle.probabilities[i] for i in idx)
        reduced_groups = project_groups(bundle.selected_groups, idx)
        expected = selected_groups_normalizer(reduced_probs, reduced_groups)

        got = selected_groups_runtime_normalizer(bundle)

        self.assertAlmostEqual(got, expected, places=12)

    def test_full_at_most_one_matches_full_selected_groups_computation(self) -> None:
        from mbs_experiments.normalizers import selected_groups_runtime_normalizer

        bundle = FakeBundle(
            probabilities=(0.7, 0.4, 0.2),
            none_is_max=(False, True, False),
            selected_groups=((0, 1, 2),),
        )
        expected = selected_groups_normalizer(
            bundle.probabilities, bundle.selected_groups
        )

        got = selected_groups_runtime_normalizer(bundle, feature_space="full")

        self.assertAlmostEqual(got, expected, places=12)

    def test_full_at_most_one_rejects_invalid_ungrouped_probability(self) -> None:
        from mbs_experiments.normalizers import selected_groups_runtime_normalizer

        bundle = FakeBundle(
            probabilities=(1.5,),
            none_is_max=(False,),
            selected_groups=(),
        )

        with self.assertRaisesRegex(ValueError, "Probability"):
            selected_groups_runtime_normalizer(bundle, feature_space="full")

    def test_visible_at_most_one_rejects_invalid_visible_probability(self) -> None:
        from mbs_experiments.normalizers import selected_groups_runtime_normalizer

        bundle = FakeBundle(
            probabilities=(1.5,),
            none_is_max=(False,),
            selected_groups=(),
        )

        with self.assertRaisesRegex(ValueError, "Probability"):
            selected_groups_runtime_normalizer(bundle, feature_space="visible")

    def test_exactly_one_factor_differs_from_at_most_one_factor(self) -> None:
        from mbs_experiments.normalizers import selected_groups_product_normalizer

        probabilities = (0.7, 0.4)
        groups = ((0, 1),)

        at_most_one = selected_groups_product_normalizer(
            probabilities, groups, exactly_one=False
        )
        exactly_one = selected_groups_product_normalizer(
            probabilities, groups, exactly_one=True
        )

        self.assertAlmostEqual(at_most_one, 0.72, places=12)
        self.assertAlmostEqual(exactly_one, 0.54, places=12)

    def test_singleton_group_contributes_only_for_exactly_one(self) -> None:
        from mbs_experiments.normalizers import selected_groups_product_normalizer

        probabilities = (0.7, 0.4)

        at_most_one = selected_groups_product_normalizer(
            probabilities, ((0,),), exactly_one=False
        )
        exactly_one = selected_groups_product_normalizer(
            probabilities, ((0,),), exactly_one=True
        )

        self.assertAlmostEqual(at_most_one, 1.0, places=12)
        self.assertAlmostEqual(exactly_one, 0.7, places=12)

    def test_product_at_most_one_rejects_invalid_singleton_probability(self) -> None:
        from mbs_experiments.normalizers import selected_groups_product_normalizer

        with self.assertRaisesRegex(ValueError, "Probability"):
            selected_groups_product_normalizer((1.5,), ((0,),), exactly_one=False)

    def test_product_at_most_one_rejects_invalid_ungrouped_probability(self) -> None:
        from mbs_experiments.normalizers import selected_groups_product_normalizer

        with self.assertRaisesRegex(ValueError, "Probability"):
            selected_groups_product_normalizer((1.5,), (), exactly_one=False)

    def test_empty_group_exactly_one_returns_zero(self) -> None:
        from mbs_experiments.normalizers import selected_groups_product_normalizer

        got = selected_groups_product_normalizer(
            (0.7, 0.4), ((),), exactly_one=True
        )

        self.assertAlmostEqual(got, 0.0, places=12)

    def test_exactly_one_visible_projection_preserves_singleton_groups(self) -> None:
        from mbs_experiments.normalizers import selected_groups_runtime_normalizer

        bundle = FakeBundle(
            probabilities=(0.7, 0.4),
            none_is_max=(False, False),
            selected_groups=((0,),),
        )

        got = selected_groups_runtime_normalizer(
            bundle, feature_space="visible", exactly_one=True
        )

        self.assertAlmostEqual(got, 0.7, places=12)

    def test_exactly_one_visible_projection_refuses_to_drop_hidden_atoms(self) -> None:
        from mbs_experiments.normalizers import selected_groups_runtime_normalizer

        bundle = FakeBundle(
            probabilities=(0.7, 0.4),
            none_is_max=(False, True),
            selected_groups=((0, 1),),
        )

        with self.assertRaisesRegex(ValueError, "exactly_one.*none_is_max"):
            selected_groups_runtime_normalizer(
                bundle, feature_space="visible", exactly_one=True
            )

    def test_implied_selected_groups_bound_the_exact_one_normalizer(self) -> None:
        from itertools import product

        from mbs_experiments.normalizers import (
            implied_selected_groups_normalizer,
            selected_groups_product_normalizer,
        )

        # An object on one of three tables or in the gripper, and a gripper that
        # is empty or holds it: the selected group drops the gripper atom.
        probabilities = (0.1, 0.1, 0.1, 0.9, 0.1)
        constraint_groups = ((0, 1, 2, 3), (3, 4))
        selected_groups = ((0, 1, 2), (3, 4))

        exact = 0.0
        for state in product((0, 1), repeat=5):
            if all(sum(state[i] for i in group) == 1 for group in constraint_groups):
                weight = 1.0
                for bit, q in zip(state, probabilities):
                    weight *= q if bit else 1.0 - q
                exact += weight

        got = implied_selected_groups_normalizer(
            probabilities, selected_groups, constraint_groups
        )
        all_exactly_one = selected_groups_product_normalizer(
            probabilities, selected_groups, exactly_one=True
        )

        self.assertAlmostEqual(got, 0.972 * 0.82, places=12)
        self.assertGreaterEqual(got, exact)
        self.assertLess(all_exactly_one, exact)

    def test_implied_selected_groups_keep_exactly_one_for_whole_groups(self) -> None:
        from mbs_experiments.normalizers import (
            implied_selected_groups_normalizer,
            selected_groups_product_normalizer,
        )

        probabilities = (0.7, 0.4, 0.2, 0.5)
        groups = ((0, 1), (2, 3))

        self.assertEqual(
            implied_selected_groups_normalizer(probabilities, groups, groups),
            selected_groups_product_normalizer(probabilities, groups, exactly_one=True),
        )

    def test_exact_constraints_helper_matches_generic_variable_elimination(self) -> None:
        from mbs_experiments.normalizers import exact_constraints_runtime_normalizer

        probabilities = (0.7, 0.4, 0.3)
        groups = ((0, 1), (1, 2))

        got_z, got_info = exact_constraints_runtime_normalizer(
            probabilities, groups, exactly_one=False
        )
        expected_z, expected_info = exact_at_most_one_normalizer(
            probabilities, groups, exactly_one=False
        )

        self.assertAlmostEqual(got_z, expected_z, places=12)
        self.assertEqual(got_info["method"], expected_info["method"])
        self.assertEqual(
            got_info["constraint_semantics"],
            expected_info["constraint_semantics"],
        )

    def test_exact_constraints_exactly_one_singleton_contributes_probability(
        self,
    ) -> None:
        from mbs_experiments.normalizers import exact_constraints_runtime_normalizer

        z, info = exact_constraints_runtime_normalizer(
            (0.7,), ((0,),), exactly_one=True
        )

        self.assertAlmostEqual(z, 0.7, places=12)
        self.assertEqual(info["constraint_semantics"], "exactly_one")

    def test_exact_constraints_exactly_one_mixes_singleton_and_generic_groups(
        self,
    ) -> None:
        from mbs_experiments.normalizers import exact_constraints_runtime_normalizer

        z, info = exact_constraints_runtime_normalizer(
            (0.7, 0.4, 0.2), ((0,), (1, 2)), exactly_one=True
        )

        self.assertAlmostEqual(z, 0.308, places=12)
        self.assertEqual(info["constraint_semantics"], "exactly_one")

    def test_exact_constraints_exactly_one_singleton_can_overlap_generic_group(
        self,
    ) -> None:
        from mbs_experiments.normalizers import exact_constraints_runtime_normalizer

        z, info = exact_constraints_runtime_normalizer(
            (0.7, 0.4), ((0,), (0, 1)), exactly_one=True
        )

        self.assertAlmostEqual(z, 0.42, places=12)
        self.assertEqual(info["constraint_semantics"], "exactly_one")

    def test_prb_exact_one_helper_uses_matching_dp(self) -> None:
        from mbs_experiments.normalizers import exact_constraints_runtime_normalizer

        feature_names = (
            "is-directly-on-table(a)",
            "shape-on-top-of-block(a,b)",
            "is-clear-on-top(b)",
        )
        probabilities = (0.6, 0.2, 0.8)
        groups = ((0, 1), (2, 1))

        got_z, info = exact_constraints_runtime_normalizer(
            probabilities,
            groups,
            feature_names=feature_names,
            domain="prb",
            exactly_one=True,
        )

        self.assertAlmostEqual(got_z, 0.4, places=12)
        self.assertEqual(info["method"], "prb_matching_dp")
        self.assertEqual(info["constraint_semantics"], "exactly_one")

    def test_prb_exact_one_helper_does_not_import_experiment_script(self) -> None:
        from mbs_experiments.normalizers import exact_constraints_runtime_normalizer

        script_module = "scripts.run_experiment_3_planning_utility"
        sys.modules.pop(script_module, None)
        feature_names = (
            "is-directly-on-table(a)",
            "shape-on-top-of-block(a,b)",
            "is-clear-on-top(b)",
        )

        exact_constraints_runtime_normalizer(
            (0.6, 0.2, 0.8),
            ((0, 1), (2, 1)),
            feature_names=feature_names,
            domain="prb",
            exactly_one=True,
        )

        self.assertNotIn(script_module, sys.modules)


if __name__ == "__main__":
    unittest.main()
