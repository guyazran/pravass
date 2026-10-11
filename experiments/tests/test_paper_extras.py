from __future__ import annotations

import math
import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd


def _exp3_record(
    domain: str,
    configuration: str,
    sample: str,
    method: str,
    *,
    theta: float | None = None,
    success: bool = False,
    mass: float | None = None,
    status: str = "ok",
    calibration: str = "global",
) -> dict:
    record = {
        "domain": domain,
        "configuration": configuration,
        "sample": sample,
        "split_relative_path": f"{domain}/{configuration}/{sample}",
        "calibration": calibration,
        "method": method,
        "theta": theta,
        "status": status,
        "success": success,
    }
    if mass is not None:
        record["prevas_prefix_search_final_constrained_mass"] = mass
    return record


STRICT = "mbs_prevas_prefix_search_constraints_exact"
RELAXED = "mbs_prevas_prefix_search_constraints_relaxed"
LOOSE = "mbs_prevas_prefix_search"
ADAPTIVE = "adaptive_fluent_abstention"


def _paired_records() -> list[dict]:
    records: list[dict] = []
    # PRB: four instances. strict succeeds on 0,1,2; map on 1; adaptive on 0;
    # loose has no record for instance 3; relaxed mirrors strict.
    strict = [True, True, True, False]
    map_ = [False, True, False, False]
    adaptive = [True, False, False, False]
    loose = [False, False, True]
    for i in range(4):
        sample = f"sample_{i:06d}"
        records.append(_exp3_record("prb", "h02_n03", sample, "oracle", success=True))
        records.append(_exp3_record("prb", "h02_n03", sample, STRICT, theta=0.99, success=strict[i]))
        records.append(_exp3_record("prb", "h02_n03", sample, RELAXED, theta=0.99, success=strict[i]))
        records.append(_exp3_record("prb", "h02_n03", sample, "map", success=map_[i]))
        records.append(_exp3_record("prb", "h02_n03", sample, ADAPTIVE, success=adaptive[i]))
        # A θ=0.5 strict row must be ignored at the θ=0.99 operating point.
        records.append(_exp3_record("prb", "h02_n03", sample, STRICT, theta=0.5, success=not strict[i]))
        if i < 3:
            records.append(_exp3_record("prb", "h02_n03", sample, LOOSE, theta=0.99, success=loose[i]))
    # GW: two instances, strict beats everything on both.
    for i in range(2):
        sample = f"sample_{i:06d}"
        records.append(_exp3_record("gw", "h01", sample, "oracle", success=True))
        records.append(_exp3_record("gw", "h01", sample, STRICT, theta=0.99, success=True))
        records.append(_exp3_record("gw", "h01", sample, RELAXED, theta=0.99, success=True))
        records.append(_exp3_record("gw", "h01", sample, LOOSE, theta=0.99, success=False))
        records.append(_exp3_record("gw", "h01", sample, "map", success=False))
        records.append(_exp3_record("gw", "h01", sample, ADAPTIVE, success=False))
    # A lifted-calibration oracle row must not enter the global population.
    records.append(_exp3_record("prb", "h02_n03", "sample_000099", "oracle", calibration="lifted"))
    return records


class PairedConstraintAblationTests(unittest.TestCase):
    def test_success_by_instance_matches_methods_on_oracle_population(self) -> None:
        from mbs_experiments.paper_extras import exp3_success_by_instance

        frame = exp3_success_by_instance(
            _paired_records(), domain="prb", methods=[STRICT, "map", LOOSE]
        )
        self.assertEqual(len(frame), 4)
        self.assertEqual(frame[STRICT].tolist(), [1.0, 1.0, 1.0, 0.0])
        self.assertEqual(frame["map"].tolist(), [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(frame[LOOSE].tolist()[:3], [0.0, 0.0, 1.0])
        self.assertTrue(math.isnan(frame[LOOSE].iloc[3]))

    def test_later_duplicate_record_wins(self) -> None:
        from mbs_experiments.paper_extras import exp3_success_by_instance

        records = _paired_records()
        records.append(
            _exp3_record("prb", "h02_n03", "sample_000003", STRICT, theta=0.99, success=True)
        )
        frame = exp3_success_by_instance(records, domain="prb", methods=[STRICT])
        self.assertEqual(frame[STRICT].tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_rows_are_sorted_by_instance_key_regardless_of_file_order(self) -> None:
        from mbs_experiments.paper_extras import exp3_success_by_instance

        records = list(reversed(_paired_records()))
        frame = exp3_success_by_instance(records, domain="prb", methods=[STRICT, "map"])
        self.assertEqual(
            frame.index.get_level_values("sample").tolist(),
            ["sample_000000", "sample_000001", "sample_000002", "sample_000003"],
        )
        self.assertEqual(frame[STRICT].tolist(), [1.0, 1.0, 1.0, 0.0])
        self.assertEqual(frame["map"].tolist(), [0.0, 1.0, 0.0, 0.0])

    def test_contrasts_in_a_domain_share_one_resample_matrix(self) -> None:
        from mbs_experiments.paper_extras import exp3_paired_contrast_summary

        records = _paired_records()
        # A second point method identical to MAP: its contrast must get exactly
        # the same CI as the MAP contrast because both reuse the domain matrix.
        for record in list(records):
            if record["method"] == "map":
                records.append({**record, "method": "valid_map"})
        contrasts = ((STRICT, "map"), (STRICT, "valid_map"))
        summary = exp3_paired_contrast_summary(
            records, contrasts=contrasts, resamples=300, seed=0
        ).set_index(["Domain", "Right"])
        for domain in ("prb", "gw"):
            self.assertEqual(
                summary.loc[(domain, "map"), ["CI Low", "CI High"]].tolist(),
                summary.loc[(domain, "valid_map"), ["CI Low", "CI High"]].tolist(),
            )
        # Results do not depend on the JSONL row order (instances are sorted).
        shuffled = exp3_paired_contrast_summary(
            list(reversed(records)), contrasts=contrasts, resamples=300, seed=0
        ).set_index(["Domain", "Right"])
        pd.testing.assert_frame_equal(summary, shuffled)
        # ... but the resampling itself depends on the seed.
        from mbs_experiments.paper_extras import paired_bootstrap_indices

        self.assertFalse(
            np.array_equal(
                paired_bootstrap_indices(10, resamples=50, rng=0),
                paired_bootstrap_indices(10, resamples=50, rng=1),
            )
        )

    def test_contrast_summary_uses_matched_pairs(self) -> None:
        from mbs_experiments.paper_extras import exp3_paired_contrast_summary

        summary = exp3_paired_contrast_summary(_paired_records(), resamples=200, seed=0)
        self.assertEqual(len(summary), 8)
        prb = summary[summary["Domain"] == "prb"].set_index("Right")
        self.assertEqual(prb.loc["map", "N Pairs"], 4)
        self.assertAlmostEqual(prb.loc["map", "Mean Difference"], 0.5)
        self.assertAlmostEqual(prb.loc[ADAPTIVE, "Mean Difference"], 0.5)
        # Loose has three matched instances: strict [1, 1, 1] vs loose [0, 0, 1].
        self.assertEqual(prb.loc[LOOSE, "N Pairs"], 3)
        self.assertAlmostEqual(prb.loc[LOOSE, "Mean Difference"], 2.0 / 3.0)
        # Exact mirrors strict, so the difference and its interval are zero.
        self.assertEqual(prb.loc[STRICT, "Mean Difference"], 0.0)
        self.assertEqual(prb.loc[STRICT, "CI Low"], 0.0)
        self.assertEqual(prb.loc[STRICT, "CI High"], 0.0)
        gw = summary[summary["Domain"] == "gw"].set_index("Right")
        self.assertEqual(gw.loc["map", "N Pairs"], 2)
        self.assertAlmostEqual(gw.loc["map", "Mean Difference"], 1.0)
        for _, row in summary.iterrows():
            self.assertLessEqual(row["CI Low"], row["Mean Difference"] + 1e-12)
            self.assertGreaterEqual(row["CI High"], row["Mean Difference"] - 1e-12)

    def test_bootstrap_is_seeded_and_degenerates_without_resamples(self) -> None:
        from mbs_experiments.paper_extras import paired_bootstrap_mean_ci

        diffs = np.array([1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        first = paired_bootstrap_mean_ci(diffs, resamples=500, seed=0)
        second = paired_bootstrap_mean_ci(diffs, resamples=500, seed=0)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], diffs.mean())
        self.assertLess(first[1], first[0])
        self.assertGreater(first[2], first[0])
        mean, low, high = paired_bootstrap_mean_ci(diffs, resamples=0)
        self.assertEqual((mean, low, high), (diffs.mean(), diffs.mean(), diffs.mean()))
        self.assertTrue(all(math.isnan(v) for v in paired_bootstrap_mean_ci([])))
        self.assertTrue(all(math.isnan(v) for v in paired_bootstrap_mean_ci([np.nan, np.nan])))

    def test_explicit_indices_and_nan_differences(self) -> None:
        from mbs_experiments.paper_extras import (
            paired_bootstrap_indices,
            paired_bootstrap_mean_ci,
        )

        indices = paired_bootstrap_indices(5, resamples=50, rng=0)
        self.assertEqual(indices.shape, (50, 5))
        self.assertTrue(((indices >= 0) & (indices < 5)).all())
        np.testing.assert_array_equal(
            indices, paired_bootstrap_indices(5, resamples=50, rng=np.random.default_rng(0))
        )
        diffs = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        self.assertEqual(
            paired_bootstrap_mean_ci(diffs, resamples=50, seed=0),
            paired_bootstrap_mean_ci(diffs, indices=indices),
        )
        # An unmatched instance (NaN) is dropped from the point estimate and
        # from every resample it lands in.
        with_nan = np.array([1.0, np.nan, 1.0, 0.0, 1.0])
        mean, low, high = paired_bootstrap_mean_ci(with_nan, indices=indices)
        self.assertAlmostEqual(mean, 0.75)
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)

    def test_formatted_table_layout(self) -> None:
        from mbs_experiments.paper_extras import exp3_paired_constraint_ablation

        table = exp3_paired_constraint_ablation(_paired_records(), resamples=200, seed=0)
        self.assertEqual(
            list(table.columns),
            ["Domain", "Contrast", "Paired difference", "95% paired bootstrap CI", "n pairs"],
        )
        self.assertEqual(table["Domain"].tolist(), ["PRB"] * 4 + ["GW"] * 4)
        self.assertEqual(
            table["Contrast"].tolist()[:4],
            [
                "AMS (Ours) − MAP",
                "AMS (Ours) − Adaptive Abstention",
                "AMS (Ours) − AMS-Loose (Ours)",
                "AMS (Ours, relaxed Z) − AMS (Ours)",
            ],
        )
        self.assertEqual(table["Paired difference"].iloc[0], "+50.0 pp")
        self.assertEqual(table["n pairs"].tolist(), [4, 4, 3, 4, 2, 2, 2, 2])
        # Zero differences never print as "-0.0".
        self.assertEqual(table["Paired difference"].iloc[3], "+0.0 pp")
        self.assertEqual(table["95% paired bootstrap CI"].iloc[3], "[+0.0, +0.0]")
        self.assertEqual(table["Paired difference"].iloc[4], "+100.0 pp")
        header = table.to_csv(index=False).splitlines()[0]
        self.assertEqual(header, "Domain,Contrast,Paired difference,95% paired bootstrap CI,n pairs")


def _fig7_records() -> list[dict]:
    """Five PRB (objects, horizon) cells whose mass equals their success rate."""
    cells = {
        ("h01_n03", 0.0),
        ("h02_n03", 0.5),
        ("h01_n04", 1.0),
        ("h02_n04", 0.5),
        ("h04_n05", 1.0),
    }
    records: list[dict] = []
    for configuration, rate in sorted(cells):
        for i in range(2):
            sample = f"sample_{i:06d}"
            records.append(_exp3_record("prb", configuration, sample, "oracle", success=True))
            success = (i == 0 and rate >= 0.5) or rate >= 1.0
            records.append(
                _exp3_record(
                    "prb", configuration, sample, STRICT, theta=0.99, success=success, mass=rate
                )
            )
    return records


class ConfidenceSuccessFigureTests(unittest.TestCase):
    def test_points_one_per_cell_with_mean_mass_and_success(self) -> None:
        from mbs_experiments.paper_extras import exp3_confidence_success_points

        points = exp3_confidence_success_points(_fig7_records())
        self.assertEqual(len(points), 5)
        self.assertEqual(points["N"].tolist(), [2] * 5)
        by_cell = points.set_index(["Number of Objects", "Horizon"])
        self.assertAlmostEqual(by_cell.loc[(3, 1), "Success"], 0.0)
        self.assertAlmostEqual(by_cell.loc[(3, 2), "Success"], 0.5)
        self.assertAlmostEqual(by_cell.loc[(4, 1), "Belief Set Mass"], 1.0)
        np.testing.assert_allclose(points["Success"], points["Belief Set Mass"])

    def test_missing_method_record_counts_as_failure(self) -> None:
        from mbs_experiments.paper_extras import exp3_confidence_success_points

        records = [
            r for r in _fig7_records()
            if not (r["configuration"] == "h01_n04" and r["method"] == STRICT and r["sample"] == "sample_000001")
        ]
        points = exp3_confidence_success_points(records).set_index(["Number of Objects", "Horizon"])
        self.assertAlmostEqual(points.loc[(4, 1), "Success"], 0.5)
        self.assertAlmostEqual(points.loc[(4, 1), "Belief Set Mass"], 1.0)

    def test_figure_returns_pearson_correlation(self) -> None:
        from mbs_experiments.paper_extras import (
            exp3_confidence_success_correlation,
            exp3_confidence_success_figure,
        )

        fig, r, p = exp3_confidence_success_figure(_fig7_records())
        try:
            self.assertAlmostEqual(r, 1.0, places=9)
            self.assertLess(p, 1e-6)
            ax = fig.axes[0]
            self.assertEqual(ax.get_xlabel(), "Subset Probability Mass")
            self.assertEqual(ax.get_ylabel(), "Success Rate")
            self.assertEqual(len(ax.collections), 1)
            self.assertEqual(len(ax.collections[0].get_offsets()), 5)
        finally:
            matplotlib.pyplot.close(fig)
        self.assertTrue(
            all(math.isnan(v) for v in exp3_confidence_success_correlation(pd.DataFrame()))
        )


def _exp2_record(
    domain: str,
    theta: float,
    *,
    features: int,
    size: int,
    scored: int,
    groups: int = 1,
    method: str = "prevas",
    status: str = "ok",
    calibration: str = "global",
    sample: str = "sample_000000",
) -> dict:
    return {
        "domain": domain,
        "configuration": "cfg",
        "sample": sample,
        "split_relative_path": f"{domain}/cfg/{sample}",
        "calibration": calibration,
        "method": method,
        "theta": theta,
        "status": status,
        "search_feature_count": features,
        "compact_set_size": size,
        "states_scored": scored,
        "search_selected_group_count": groups,
    }


class StatesVisitedTableTests(unittest.TestCase):
    def test_unconstrained_bound(self) -> None:
        from mbs_experiments.paper_extras import exp2_unconstrained_bound

        self.assertEqual(exp2_unconstrained_bound(10, 3), 31)
        self.assertEqual(exp2_unconstrained_bound(25, 1), 26)

    def test_medians_and_bound_by_domain_and_theta(self) -> None:
        from mbs_experiments.paper_extras import exp2_states_visited_table

        records = [
            _exp2_record("alf", 0.5, features=8, size=2, scored=10, sample="a"),
            _exp2_record("prb", 0.5, features=10, size=3, scored=20, groups=2, sample="a"),
            _exp2_record("prb", 0.5, features=10, size=2, scored=40, sample="b"),
            _exp2_record("prb", 0.9, features=10, size=5, scored=60, sample="a"),
            # Excluded: unconstrained instance, timeout, exhaustive, other calibration.
            _exp2_record("prb", 0.5, features=10, size=1, scored=5, groups=0, sample="c"),
            _exp2_record("prb", 0.5, features=10, size=1, scored=5, status="timeout", sample="d"),
            _exp2_record("prb", 0.5, features=10, size=1, scored=1024, method="exhaustive", sample="a"),
            _exp2_record("prb", 0.5, features=10, size=1, scored=5, calibration="lifted", sample="a"),
        ]
        table = exp2_states_visited_table(records)
        self.assertEqual(table["Domain"].tolist(), ["prb", "prb", "alf"])
        self.assertEqual(table["Theta"].tolist(), [0.5, 0.9, 0.5])
        row = table.set_index(["Domain", "Theta"]).loc[("prb", 0.5)]
        self.assertEqual(row["N"], 2)
        self.assertEqual(row["Median States Visited"], 30.0)
        self.assertEqual(row["Median Unconstrained Bound"], 26.0)  # median of 31 and 21
        self.assertAlmostEqual(row["Median Visited / Bound"], (20 / 31 + 40 / 21) / 2)
        self.assertEqual(row["Within Bound %"], 50.0)
        self.assertEqual(row["Median Set Size"], 2.5)
        self.assertEqual(row["Median Search Features"], 10.0)

        unconstrained_included = exp2_states_visited_table(records, constrained_only=False)
        self.assertEqual(
            unconstrained_included.set_index(["Domain", "Theta"]).loc[("prb", 0.5), "N"], 3
        )
        only_09 = exp2_states_visited_table(records, thetas=(0.9,), domains=("prb",))
        self.assertEqual(len(only_09), 1)
        self.assertEqual(only_09["Median States Visited"].iloc[0], 60.0)

    def test_empty_input_gives_empty_table_with_columns(self) -> None:
        from mbs_experiments.paper_extras import exp2_states_visited_table

        table = exp2_states_visited_table([])
        self.assertTrue(table.empty)
        self.assertIn("Median Unconstrained Bound", table.columns)


if __name__ == "__main__":
    unittest.main()
