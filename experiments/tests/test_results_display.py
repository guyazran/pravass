from __future__ import annotations

import math
import unittest



class ResultsDisplayTests(unittest.TestCase):
    def test_exp1_summary_groups_theta_methods_and_deduplicates_point_methods(self) -> None:
        from mbs_experiments.results_display import exp1_coverage_size_summary

        records = [
            {
                "calibration": "global",
                "domain": "alf",
                "method": "mbs_prevas",
                "theta": 0.5,
                "status": "ok",
                "split_relative_path": "alf/a",
                "contains_truth": True,
                "compact_set_size": 2,
            },
            {
                "calibration": "global",
                "domain": "alf",
                "method": "mbs_prevas",
                "theta": 0.5,
                "status": "ok",
                "split_relative_path": "alf/b",
                "contains_truth": False,
                "compact_set_size": 4,
            },
            {
                "calibration": "global",
                "domain": "alf",
                "method": "map",
                "theta": 0.5,
                "status": "ok",
                "split_relative_path": "alf/a",
                "contains_truth": True,
                "compact_set_size": 1,
            },
            {
                "calibration": "global",
                "domain": "alf",
                "method": "map",
                "theta": 0.9,
                "status": "ok",
                "split_relative_path": "alf/a",
                "contains_truth": True,
                "compact_set_size": 1,
            },
            {
                "calibration": "global",
                "domain": "alf",
                "method": "map",
                "theta": 0.5,
                "status": "ok",
                "split_relative_path": "alf/b",
                "contains_truth": False,
                "compact_set_size": 1,
            },
            {
                "calibration": "global",
                "domain": "alf",
                "method": "map",
                "theta": 0.9,
                "status": "ok",
                "split_relative_path": "alf/b",
                "contains_truth": False,
                "compact_set_size": 1,
            },
        ]

        summary = exp1_coverage_size_summary(records, bootstrap_resamples=20)

        mbs = summary[summary["Method"].eq("MBSθ (Ours)")].iloc[0]
        self.assertEqual(mbs["Theta"], "0.50")
        self.assertEqual(mbs["N"], 2)
        self.assertAlmostEqual(mbs["Coverage"], 0.5)
        self.assertAlmostEqual(mbs["Average Set Size"], 3.0)

        map_row = summary[summary["Method"].eq("MAP")].iloc[0]
        self.assertEqual(map_row["Theta"], "point")
        self.assertEqual(map_row["N"], 2)
        self.assertAlmostEqual(map_row["Coverage"], 0.5)
        self.assertAlmostEqual(map_row["Average Set Size"], 1.0)

    def test_exp1_theta_methods_use_common_theta_support(self) -> None:
        from mbs_experiments.results_display import exp1_coverage_size_summary

        records = [
            {
                "calibration": "global",
                "domain": "prb",
                "method": "random_theta_confident",
                "theta": 0.5,
                "status": "ok",
                "split_relative_path": "prb/a",
                "contains_truth": True,
                "compact_set_size": 1,
            },
            {
                "calibration": "global",
                "domain": "prb",
                "method": "random_theta_confident",
                "theta": 0.9,
                "status": "ok",
                "split_relative_path": "prb/a",
                "contains_truth": True,
                "compact_set_size": 2,
            },
            {
                "calibration": "global",
                "domain": "prb",
                "method": "random_theta_confident",
                "theta": 0.5,
                "status": "ok",
                "split_relative_path": "prb/hard",
                "contains_truth": True,
                "compact_set_size": 1000,
            },
            {
                "calibration": "global",
                "domain": "prb",
                "method": "random_theta_confident",
                "theta": 0.9,
                "status": "timeout",
                "split_relative_path": "prb/hard",
                "contains_truth": False,
                "compact_set_size": 0,
            },
        ]

        summary = exp1_coverage_size_summary(records, bootstrap_resamples=20)
        sizes = summary.sort_values("Theta Value")["Average Set Size"].tolist()

        self.assertEqual(summary["N"].tolist(), [1, 1])
        self.assertEqual(sizes, [1.0, 2.0])

    def test_exp1_feature_abstention_uses_comparable_method_common_support(self) -> None:
        from mbs_experiments.results_display import exp1_coverage_size_summary

        records = []
        for theta, mbs_size, abstention_size in [(0.5, 1, 10), (0.9, 2, 10)]:
            records.extend(
                [
                    {
                        "calibration": "global",
                        "domain": "alf",
                        "method": "mbs_prevas",
                        "theta": theta,
                        "status": "ok",
                        "split_relative_path": "alf/common",
                        "contains_truth": True,
                        "compact_set_size": mbs_size,
                    },
                    {
                        "calibration": "global",
                        "domain": "alf",
                        "method": "fluent_abstention_unconstrained",
                        "theta": theta,
                        "status": "ok",
                        "split_relative_path": "alf/common",
                        "contains_truth": True,
                        "compact_set_size": abstention_size,
                    },
                    {
                        "calibration": "global",
                        "domain": "alf",
                        "method": "mbs_prevas",
                        "theta": theta,
                        "status": "timeout",
                        "split_relative_path": "alf/hard",
                        "contains_truth": False,
                        "compact_set_size": 0,
                    },
                    {
                        "calibration": "global",
                        "domain": "alf",
                        "method": "fluent_abstention_unconstrained",
                        "theta": theta,
                        "status": "ok",
                        "split_relative_path": "alf/hard",
                        "contains_truth": True,
                        "compact_set_size": 1_000_000,
                    },
                ]
            )

        summary = exp1_coverage_size_summary(records, bootstrap_resamples=20)
        abstention = summary[summary["Method"].eq("Feature Abstention")]

        self.assertEqual(abstention["N"].tolist(), [1, 1])
        self.assertEqual(abstention["Average Set Size"].tolist(), [10.0, 10.0])

    def test_exp1_table_uses_theta_index_and_inline_ci_without_n(self) -> None:
        from mbs_experiments.results_display import (
            exp1_coverage_size_summary,
            exp1_coverage_size_table,
        )

        records = [
            {
                "calibration": "global",
                "domain": "alf",
                "method": "mbs_prevas",
                "theta": theta,
                "status": "ok",
                "split_relative_path": "alf/a",
                "contains_truth": True,
                "compact_set_size": size,
            }
            for theta, size in [(0.5, 1), (0.9, 2)]
        ]

        summary = exp1_coverage_size_summary(records, bootstrap_resamples=20)
        table = exp1_coverage_size_table(summary)

        self.assertEqual(table.index.names, ["Theta", "Method"])
        self.assertEqual(table.index[0][0], "0.50")
        self.assertNotIn("N", table.columns.get_level_values(-1))
        self.assertEqual(table.loc[("0.50", "MBSθ (Ours)"), ("alf", "Coverage")], "1.00 [1.00, 1.00]")
        self.assertEqual(table.loc[("0.90", "MBSθ (Ours)"), ("alf", "Average Set Size")], "2.00 [2.00, 2.00]")

    def test_exp2_summary_deduplicates_exhaustive_and_uses_full_state_count(self) -> None:
        from mbs_experiments.results_display import exp2_performance_summary

        records = [
            {
                "calibration": "global",
                "domain": "alf",
                "configuration": "all",
                "sample": "a",
                "method": "exhaustive",
                "theta": 0.5,
                "status": "ok",
                "runtime_seconds": 8.0,
                "search_feature_count": 3,
            },
            {
                "calibration": "global",
                "domain": "alf",
                "configuration": "all",
                "sample": "a",
                "method": "exhaustive",
                "theta": 0.9,
                "status": "ok",
                "runtime_seconds": 8.0,
                "search_feature_count": 3,
            },
            {
                "calibration": "global",
                "domain": "alf",
                "configuration": "all",
                "sample": "b",
                "method": "exhaustive",
                "theta": 0.5,
                "status": "timeout",
                "runtime_seconds": 10.0,
                "search_feature_count": 4,
            },
            {
                "calibration": "global",
                "domain": "alf",
                "configuration": "all",
                "sample": "a",
                "method": "prevas",
                "theta": 0.5,
                "status": "ok",
                "runtime_seconds": 2.0,
                "states_scored": 5,
                "search_feature_count": 3,
            },
            {
                "calibration": "global",
                "domain": "alf",
                "configuration": "all",
                "sample": "b",
                "method": "prevas",
                "theta": 0.5,
                "status": "timeout",
                "runtime_seconds": 10.0,
                "states_scored": 9,
                "search_feature_count": 4,
            },
        ]

        summary = exp2_performance_summary(records, bootstrap_resamples=20)

        exhaustive = summary[summary["Method"].eq("Exhaustive")].iloc[0]
        self.assertEqual(exhaustive["Theta"], "1.00")
        self.assertEqual(exhaustive["Total"], 2)
        self.assertEqual(exhaustive["Solved"], 1)
        self.assertEqual(exhaustive["Max Search Feature Count Solved"], 3)
        self.assertAlmostEqual(exhaustive["Median Runtime"], 8.0)
        self.assertAlmostEqual(exhaustive["Median States Visited"], 8.0)

        prevas = summary[summary["Method"].eq("PRaVaSS")].iloc[0]
        self.assertEqual(prevas["Theta"], "0.50")
        self.assertEqual(prevas["Total"], 2)
        self.assertEqual(prevas["Solved"], 1)
        self.assertEqual(prevas["Max Search Feature Count Solved"], 3)
        self.assertAlmostEqual(prevas["Median Runtime"], 2.0)
        self.assertAlmostEqual(prevas["Median States Visited"], 5.0)

    def test_exp2_performance_table_uses_inline_ci_columns(self) -> None:
        from mbs_experiments.results_display import (
            exp2_performance_summary,
            exp2_performance_table,
        )

        records = [
            {
                "calibration": "global",
                "domain": "alf",
                "configuration": "all",
                "sample": "a",
                "method": "prevas",
                "theta": 0.5,
                "status": "ok",
                "runtime_seconds": 2.0,
                "states_scored": 8,
                "search_feature_count": 3,
            }
        ]

        summary = exp2_performance_summary(records, bootstrap_resamples=20)
        table = exp2_performance_table(summary)

        self.assertIn("Median Runtime", table.columns)
        self.assertIn("Max Search Feature Count Solved", table.columns)
        self.assertNotIn("Runtime CI Low", table.columns)
        self.assertNotIn("Runtime CI High", table.columns)
        self.assertEqual(table.loc[("PRaVaSS", "0.50"), "Max Search Feature Count Solved"], 3)
        self.assertEqual(table.loc[("PRaVaSS", "0.50"), "Median Runtime"], "2.00 [2.00, 2.00]")
        self.assertEqual(table.loc[("PRaVaSS", "0.50"), "Median States Visited"], "8.00 [8.00, 8.00]")

    def test_plot_helpers_return_standalone_figures(self) -> None:
        from mbs_experiments.results_display import (
            cactus_figures,
            coverage_size_figures,
            exp1_coverage_size_summary,
            exp2_records_for_domains,
        )

        exp1_records = [
            {
                "calibration": "global",
                "domain": "alf",
                "method": "mbs_prevas",
                "theta": theta,
                "status": "ok",
                "split_relative_path": "alf/a",
                "contains_truth": True,
                "compact_set_size": size,
            }
            for theta, size in [(0.5, 2), (0.9, 5)]
        ]
        exp1_records.extend(
            {
                "calibration": "global",
                "domain": "alf",
                "method": "map",
                "theta": theta,
                "status": "ok",
                "split_relative_path": "alf/a",
                "contains_truth": True,
                "compact_set_size": 1,
            }
            for theta in (0.5, 0.9)
        )
        summary = exp1_coverage_size_summary(exp1_records, bootstrap_resamples=20)
        figures = coverage_size_figures(summary)
        # One figure with a panel per paper domain (PRB, GW, ALF).
        self.assertEqual(set(figures), {"all"})
        self.assertEqual(len(figures["all"].axes), 3)

        exp2_records = [
            {
                "calibration": "global",
                "domain": "alf",
                "configuration": "all",
                "sample": "a",
                "method": "exhaustive",
                "theta": theta,
                "status": "ok",
                "runtime_seconds": 4.0,
                "search_feature_count": 2,
            }
            for theta in (0.5, 0.9)
        ]
        exp2_records.extend(
            {
                "calibration": "global",
                "domain": "alf",
                "configuration": "all",
                "sample": "a",
                "method": "prevas",
                "theta": theta,
                "status": "ok",
                "runtime_seconds": runtime,
                "states_scored": states,
                "search_feature_count": 2,
            }
            for theta, runtime, states in [(0.5, 1.0, 2), (0.9, 2.0, 4)]
        )
        filtered = exp2_records_for_domains(exp2_records, ["alf"])
        cactus = cactus_figures(filtered, domain_groups={"alf": ["alf"]})
        self.assertEqual(set(cactus), {"alf"})
        self.assertEqual(len(cactus["alf"].axes), 1)
        self.assertTrue(
            all(math.isfinite(line.get_ydata()[0]) for line in cactus["alf"].axes[0].lines)
        )


if __name__ == "__main__":
    unittest.main()
