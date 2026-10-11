from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

from mbs_experiments import notebook_data as nd


def _summary() -> dict:
    methods = ["map", "adaptive_abstention", "ams", "oracle"]
    success = {
        ("map", 0): 83 / 115,
        ("map", 1): 92 / 115,
        ("adaptive_abstention", 0): 104 / 115,
        ("adaptive_abstention", 1): 107 / 115,
        ("ams", 0): 107 / 115,
        ("ams", 1): 111 / 115,
        ("oracle", 0): 1.0,
        ("oracle", 1): 1.0,
    }
    method_round = {
        f"m0:{method}:k{round_index}": {
            "n": 115,
            "success_rate": success[method, round_index],
        }
        for method in methods
        for round_index in (0, 1)
    }
    effects = {
        ("ams", "map", 0): (24 / 115, 0.13, 0.29),
        ("ams", "map", 1): (19 / 115, 0.10, 0.24),
        ("adaptive_abstention", "map", 0): (21 / 115, 0.12, 0.25),
        ("adaptive_abstention", "map", 1): (15 / 115, 0.08, 0.19),
        ("ams", "adaptive_abstention", 0): (3 / 115, -0.01, 0.07),
        ("ams", "adaptive_abstention", 1): (4 / 115, 0.01, 0.08),
    }
    paired = {
        f"m0:k{round_index}:{left}-{right}": {
            "n": 115,
            "mean": values[0],
            "ci_low": values[1],
            "ci_high": values[2],
        }
        for (left, right, round_index), values in effects.items()
    }
    efficiency = {}
    values = {
        ("ams", 0): (14.0, 20, 0.467, 0.455, 154.8, 186.4, 6.4),
        ("ams", 1): (14.7, 20, 0.756, 1.0, 152.0, 174.6, 6.5),
        ("adaptive_abstention", 0): (9.5, 12, 0.346, 0.25, 70.5, 73.2, 3.6),
        ("adaptive_abstention", 1): (10.0, 12, 0.666, 1.0, 70.1, 72.9, 3.7),
    }
    for (method, round_index), row in values.items():
        efficiency[f"m0:{method}:k{round_index}"] = {
            "n": 115,
            "mean_belief_size": row[0],
            "median_belief_size": row[1],
            "mean_probability_mass": row[2],
            "median_probability_mass": row[3],
            "mean_total_search_runtime_seconds": row[4],
            "median_total_search_runtime_seconds": row[5],
            "mean_search_attempts": row[6],
        }
    return {
        "config": {
            "model": "Vendor/model",
            "calibrations": ["global"],
            "methods": methods,
            "prior_mode": "uniform_hidden",
        },
        "analysis_decision": {"decision": "analysis-ready"},
        "complete_sample_conditions": 115,
        "method_round": method_round,
        "paired_success_differences": paired,
        "method_efficiency": efficiency,
    }


def test_load_exp6_uniform_hidden_summary_validates_protocol(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "Vendor__model" / "main"
    run_dir.mkdir(parents=True)
    (run_dir / "aggregate_metrics.json").write_text(json.dumps(_summary()))

    with patch.object(nd, "EXP6_CONTROLLED_PRIOR_ROOT", tmp_path):
        loaded = nd.load_exp6_uniform_hidden_summary(
            "Vendor__model",
            calibration="global",
            run_name="main",
        )
        assert loaded["complete_sample_conditions"] == 115
        with pytest.raises(ValueError, match="Calibration"):
            nd.load_exp6_uniform_hidden_summary(
                "Vendor__model",
                calibration="lifted",
                run_name="main",
            )


def test_exp6_tables_report_success_effects_mass_size_and_search() -> None:
    summary = _summary()
    methods = ["map", "adaptive_abstention", "ams", "oracle"]
    success = nd.exp6_closed_loop_success_table(
        summary,
        model="Vendor__model",
        calibration="global",
        methods=methods,
    )
    assert success.loc["MAP", "Before observation"] == "83/115 (72.2%)"
    assert success.loc["AMS", "After observation"] == "111/115 (96.5%)"

    effects = nd.exp6_closed_loop_effect_table(
        summary,
        model="Vendor__model",
        calibration="global",
        methods=methods,
    )
    assert effects.loc["AMS − MAP", "Before observation"].startswith("+20.9 pp")

    efficiency = nd.exp6_closed_loop_efficiency_table(
        summary,
        model="Vendor__model",
        calibration="global",
        methods=["ams", "adaptive_abstention"],
    )
    assert efficiency.loc[
        ("Before observation", "AMS"),
        "Probability mass, mean (median)",
    ] == "46.7% (45.5%)"
    assert efficiency.loc[
        ("After observation", "Adaptive abstention"),
        "Total search time, mean (median)",
    ] == "70.1s (72.9s)"


def test_exp6_plot_has_four_result_panels() -> None:
    fig = nd.exp6_closed_loop_plot(
        _summary(),
        model="Vendor__model",
        calibration="global",
        methods=["map", "adaptive_abstention", "ams", "oracle"],
    )
    try:
        assert len(fig.axes) == 4
        assert fig.axes[0].get_title() == "(a) Ground-truth plan execution"
        assert fig.axes[3].get_title() == "(d) Total planning search effort"
    finally:
        plt.close(fig)
