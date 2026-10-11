from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

from mbs_experiments import notebook_data as nd

METHODS = ["mbs_prevas", "oracle_joint", "valid_map", "map"]


def _cell(
    domain: str,
    group_mode: str,
    rate: float,
    strength: float,
    *,
    gap: float,
    variation: float,
    ratio: float,
    meets: float,
    mbs_coverage: float,
    map_coverage: float,
) -> dict:
    return {
        "domain": domain,
        "group_mode": group_mode,
        "rate": rate,
        "strength": strength,
        "theta": 0.9,
        "n": 120,
        "mass_coverage_gap": {
            "n": 120,
            "mean": gap,
            "ci_low": gap - 0.01,
            "ci_high": gap + 0.01,
        },
        "coverage_shortfall": {
            "n": 120,
            "mean": max(0.0, -gap),
            "ci_low": 0.0,
            "ci_high": 0.02,
        },
        "meets_theta_rate": meets,
        "total_variation_mean": variation,
        "total_variation_median": variation,
        "size_ratio_median": ratio,
        "method_size_median": {
            "mbs_prevas": 24.0,
            "oracle_joint": 12.0,
            "valid_map": 1.0,
            "map": 1.0,
        },
        "method_truth_coverage": {
            "mbs_prevas": mbs_coverage,
            "oracle_joint": mbs_coverage,
            "valid_map": map_coverage,
            "map": map_coverage,
        },
        "method_true_mass_mean": {method: 0.9 for method in METHODS},
    }


def _summary() -> dict:
    sweep = {
        "prb:constraint:0.0:0.0:0.9": _cell(
            "prb", "constraint", 0.0, 0.0,
            gap=0.0, variation=0.0, ratio=1.0, meets=1.0,
            mbs_coverage=0.93, map_coverage=0.79,
        ),
        "prb:constraint:0.2:1.0:0.9": _cell(
            "prb", "constraint", 0.2, 1.0,
            gap=-0.034, variation=0.412, ratio=2.33, meets=0.88,
            mbs_coverage=0.90, map_coverage=0.41,
        ),
        "prb:uncertain:0.2:1.0:0.9": _cell(
            "prb", "uncertain", 0.2, 1.0,
            gap=-0.058, variation=0.377, ratio=1.90, meets=0.96,
            mbs_coverage=0.91, map_coverage=0.44,
        ),
        "gw:constraint:0.2:1.0:0.9": _cell(
            "gw", "constraint", 0.2, 1.0,
            gap=-0.035, variation=0.370, ratio=3.50, meets=1.0,
            mbs_coverage=1.0, map_coverage=0.0,
        ),
        # A non-selected strength that must never appear in a rho=1.0 selection.
        "gw:constraint:0.2:0.5:0.9": _cell(
            "gw", "constraint", 0.2, 0.5,
            gap=-0.011, variation=0.180, ratio=1.40, meets=1.0,
            mbs_coverage=1.0, map_coverage=0.2,
        ),
    }
    return {
        "config": {
            "model": "Vendor/model",
            "calibrations": ["global"],
            "domains": ["prb", "gw"],
            "methods": METHODS,
            "group_modes": ["constraint", "uncertain"],
            "thetas": [0.9],
        },
        "analysis_decision": {"decision": "analysis-ready"},
        "complete_cells": 600,
        "samples": 120,
        "sweep": sweep,
    }


def test_load_exp7_correlated_noise_summary_validates_protocol(tmp_path: Path) -> None:
    run_dir = tmp_path / "Vendor__model" / "main"
    run_dir.mkdir(parents=True)
    (run_dir / "aggregate_metrics.json").write_text(json.dumps(_summary()))

    with patch.object(nd, "EXP7_CORRELATED_NOISE_ROOT", tmp_path):
        loaded = nd.load_exp7_correlated_noise_summary(
            "Vendor__model", calibration="global", run_name="main"
        )
        assert loaded["samples"] == 120
        with pytest.raises(ValueError, match="Calibration"):
            nd.load_exp7_correlated_noise_summary(
                "Vendor__model", calibration="lifted", run_name="main"
            )


def test_exp7_sweep_selection_keeps_the_control_and_drops_other_strengths() -> None:
    rows = nd.exp7_sweep_rows(
        _summary(), group_mode="constraint", strength=1.0, theta=0.9
    )
    selected = {(row["domain"], row["rate"], row["strength"]) for row in rows}
    # The rho=0.5 cell must not leak into a rho=1.0 selection, but the rate-0
    # control is shared across strengths and must survive.
    assert ("gw", 0.2, 0.5) not in selected
    assert ("prb", 0.0, 0.0) in selected
    assert ("gw", 0.2, 1.0) in selected
    assert [row["domain"] for row in rows] == ["prb", "prb", "gw"]


def test_exp7_tables_report_distortion_robustness_and_block_modes() -> None:
    summary = _summary()
    distortion = nd.exp7_distortion_table(
        summary,
        model="Vendor__model",
        calibration="global",
        group_mode="constraint",
        strength=1.0,
        theta=0.9,
    )
    assert distortion.loc[("PRB", "0.20"), "Total variation"] == "0.412"
    assert distortion.loc[("PRB", "0.20"), "Attributed − true mass"] == (
        "-3.4 pp [-4.4, -2.4]"
    )
    assert distortion.loc[("PRB", "0.20"), "Reaches θ"] == "88.0%"
    assert distortion.loc[("GW", "0.20"), "Size vs true-joint set"] == "3.50×"

    robustness = nd.exp7_robustness_table(
        summary,
        model="Vendor__model",
        calibration="global",
        group_mode="constraint",
        strength=1.0,
        theta=0.9,
        methods=METHODS,
    )
    assert robustness.loc[("GW", "0.20"), "MBSθ (product belief)"] == "100.0% (|S|=24)"
    assert robustness.loc[("GW", "0.20"), "MAP"] == "0.0% (|S|=1)"

    blocks = nd.exp7_block_mode_table(
        summary,
        model="Vendor__model",
        calibration="global",
        group_modes=["constraint", "uncertain"],
        strength=1.0,
        theta=0.9,
    )
    # The uncorrelated control carries no block contrast and is dropped.
    assert ("PRB", "0.00") not in blocks.index
    assert blocks.loc[("PRB", "0.20"), "Mutex-coupled block"].startswith("-3.4 pp")
    assert blocks.loc[("PRB", "0.20"), "Least-decided block"].endswith("θ met 96%")
    assert blocks.loc[("GW", "0.20"), "Least-decided block"] == "—"


def test_exp7_rejects_an_unavailable_block_mode() -> None:
    with pytest.raises(ValueError, match="block mode"):
        nd.exp7_distortion_table(
            _summary(),
            model="Vendor__model",
            calibration="global",
            group_mode="random",
            strength=1.0,
            theta=0.9,
        )


def test_exp7_correlated_noise_plot_has_four_result_panels() -> None:
    fig = nd.exp7_correlated_noise_plot(
        _summary(),
        model="Vendor__model",
        calibration="global",
        group_mode="constraint",
        strength=1.0,
        theta=0.9,
    )
    try:
        assert len(fig.axes) == 4
        assert fig.axes[0].get_title() == "(a) Distance from the true joint"
        assert fig.axes[3].get_title() == "(d) True-state coverage, MBSθ vs MAP"
        # Colour encodes the domain, so panel (d) needs its own line-style key.
        assert [t.get_text() for t in fig.axes[3].get_legend().get_texts()] == [
            "MBSθ",
            "MAP",
        ]
    finally:
        plt.close(fig)
