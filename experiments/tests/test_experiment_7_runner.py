from __future__ import annotations

import importlib.util
import itertools
import math
from pathlib import Path

from mbs_experiments.experiment7 import (
    build_corruption_model,
    component_normalizers,
    affected_block,
    oracle_joint_prefix,
    select_corrupted_indices,
    summarize_gap,
    sweep_points,
    total_variation,
    true_normalizer,
    true_state_mass,
)
from mbs_experiments.baselines import mbs_prevas
from mbs_experiments.states import (
    is_valid_state,
    selected_groups_normalizer,
    unconstrained_probability,
)

PROBABILITIES = (0.92, 0.08, 0.55, 0.31, 0.74, 0.12)
GROUPS = ((0, 1, 2), (3, 4))


def _brute_force_true_mass(states, model, groups) -> float:
    """Reference mass by enumerating the whole valid space."""
    w0, w1 = model.weights
    total = 0.0
    selected = 0.0
    for assignment in itertools.product(
        (False, True), repeat=len(model.clean)
    ):
        if not is_valid_state(assignment, groups):
            continue
        density = w0 * unconstrained_probability(assignment, model.clean) + w1 * (
            unconstrained_probability(assignment, model.corrupt)
        )
        total += density
        if assignment in set(states):
            selected += density
    return selected / total


def test_estimator_marginals_equal_the_true_joint_marginals() -> None:
    # The whole design rests on this: the marginals handed to the belief converter
    # are exactly the true joint's marginals, so every discrepancy that follows is
    # attributable to the conditional-independence lift and nothing else.
    model = build_corruption_model(
        PROBABILITIES, (0, 2, 4), rate=0.3, strength=0.8
    )
    w0, w1 = model.weights
    for index in range(len(PROBABILITIES)):
        joint_marginal = 0.0
        for assignment in itertools.product(
            (False, True), repeat=len(PROBABILITIES)
        ):
            if not assignment[index]:
                continue
            joint_marginal += w0 * unconstrained_probability(assignment, model.clean)
            joint_marginal += w1 * unconstrained_probability(
                assignment, model.corrupt
            )
        assert math.isclose(joint_marginal, model.marginals[index], abs_tol=1e-9)


def test_zero_rate_and_zero_strength_are_uncorrelated_controls() -> None:
    for rate, strength in ((0.0, 1.0), (0.4, 0.0)):
        model = build_corruption_model(
            PROBABILITIES, (0, 1), rate=rate, strength=strength
        )
        assert total_variation(model, GROUPS) < 1e-12
        z_clean, z_corrupt = component_normalizers(model, GROUPS)
        assert math.isclose(z_clean, z_corrupt, rel_tol=1e-9)


def test_true_normalizer_matches_brute_force_enumeration() -> None:
    model = build_corruption_model(PROBABILITIES, (1, 3), rate=0.25, strength=1.0)
    reference = 0.0
    w0, w1 = model.weights
    for assignment in itertools.product((False, True), repeat=len(PROBABILITIES)):
        if not is_valid_state(assignment, GROUPS):
            continue
        reference += w0 * unconstrained_probability(assignment, model.clean)
        reference += w1 * unconstrained_probability(assignment, model.corrupt)
    assert math.isclose(true_normalizer(model, GROUPS), reference, rel_tol=1e-9)


def test_true_state_mass_matches_brute_force_enumeration() -> None:
    model = build_corruption_model(PROBABILITIES, (0, 2), rate=0.35, strength=0.9)
    product_z = selected_groups_normalizer(model.marginals, GROUPS)
    states = mbs_prevas(model.marginals, GROUPS, 0.8, product_z).state_set.states
    assert states
    assert math.isclose(
        true_state_mass(states, model, GROUPS),
        _brute_force_true_mass(states, model, GROUPS),
        rel_tol=1e-9,
    )


def test_corrupting_a_single_feature_induces_no_correlation() -> None:
    # A mixture of two Bernoullis with the same mean is that Bernoulli, so a
    # one-feature nuisance is indistinguishable from recalibration.  Correlation
    # needs at least two jointly corrupted features.
    model = build_corruption_model(PROBABILITIES, (5,), rate=0.4, strength=1.0)
    assert total_variation(model, GROUPS) < 1e-12


def test_total_variation_over_the_block_equals_the_full_space_distance() -> None:
    # Outside the affected block both distributions are the same product, so the
    # block restriction must lose nothing.
    model = build_corruption_model(PROBABILITIES, (0, 3), rate=0.4, strength=1.0)
    w0, w1 = model.weights
    product_terms: list[float] = []
    true_terms: list[float] = []
    for assignment in itertools.product((False, True), repeat=len(PROBABILITIES)):
        if not is_valid_state(assignment, GROUPS):
            continue
        product_terms.append(unconstrained_probability(assignment, model.marginals))
        true_terms.append(
            w0 * unconstrained_probability(assignment, model.clean)
            + w1 * unconstrained_probability(assignment, model.corrupt)
        )
    product_total = sum(product_terms)
    true_total = sum(true_terms)
    reference = 0.5 * sum(
        abs(p / product_total - t / true_total)
        for p, t in zip(product_terms, true_terms)
    )
    assert reference > 1e-3, "pick a block that actually induces correlation"
    assert math.isclose(total_variation(model, GROUPS), reference, rel_tol=1e-9)


def test_affected_block_absorbs_whole_mutex_groups() -> None:
    # Touching one member of a mutex group pulls in the rest, otherwise the
    # factorization the total-variation shortcut relies on would not hold.
    assert affected_block((0,), GROUPS, len(PROBABILITIES)) == (0, 1, 2)
    assert affected_block((3,), GROUPS, len(PROBABILITIES)) == (3, 4)
    assert affected_block((5,), GROUPS, len(PROBABILITIES)) == (5,)


def test_oracle_prefix_is_certified_and_matches_exhaustive_ranking() -> None:
    model = build_corruption_model(PROBABILITIES, (0, 1, 4), rate=0.3, strength=1.0)
    theta = 0.85
    prefix = oracle_joint_prefix(model, GROUPS, theta)
    assert prefix.certified
    assert prefix.mass + 1e-12 >= theta

    w0, w1 = model.weights
    ranked = sorted(
        (
            w0 * unconstrained_probability(state, model.clean)
            + w1 * unconstrained_probability(state, model.corrupt)
            for state in itertools.product(
                (False, True), repeat=len(PROBABILITIES)
            )
            if is_valid_state(state, GROUPS)
        ),
        reverse=True,
    )
    denominator = true_normalizer(model, GROUPS)
    accumulated = 0.0
    expected_size = 0
    for value in ranked:
        accumulated += value
        expected_size += 1
        if accumulated / denominator + 1e-12 >= theta:
            break
    assert len(prefix.states) == expected_size


def test_oracle_prefix_collapses_to_prevas_when_uncorrelated() -> None:
    # At rate 0 the joint *is* the product belief, so the two searches must agree.
    model = build_corruption_model(PROBABILITIES, (0, 1), rate=0.0, strength=1.0)
    product_z = selected_groups_normalizer(model.marginals, GROUPS)
    states = mbs_prevas(model.marginals, GROUPS, 0.9, product_z).state_set.states
    prefix = oracle_joint_prefix(model, GROUPS, 0.9)
    assert prefix.certified
    assert set(prefix.states) == set(states)


def test_constraint_block_prefers_grouped_features_and_random_is_seeded() -> None:
    constraint_block = select_corrupted_indices(
        PROBABILITIES, GROUPS, mode="constraint", size=3, seed=0
    )
    assert constraint_block == (0, 1, 2)
    uncertain_block = select_corrupted_indices(
        PROBABILITIES, GROUPS, mode="uncertain", size=2, seed=0
    )
    assert uncertain_block == (2, 3)
    first = select_corrupted_indices(
        PROBABILITIES, GROUPS, mode="random", size=3, seed=7
    )
    repeated = select_corrupted_indices(
        PROBABILITIES, GROUPS, mode="random", size=3, seed=7
    )
    assert first == repeated


def test_sweep_points_collapse_strength_only_at_zero_rate() -> None:
    points = sweep_points((0.0, 0.2), (0.5, 1.0), (0.9,))
    assert points.count((0.0, 0.0, 0.9)) == 1
    assert (0.2, 0.5, 0.9) in points and (0.2, 1.0, 0.9) in points
    assert len(points) == 3


def test_summarize_gap_separates_conservatism_from_overconfidence() -> None:
    conservative = summarize_gap(0.90, 0.96, 0.9)
    assert conservative["mass_coverage_gap"] < 0
    assert conservative["coverage_shortfall"] == 0.0
    assert conservative["meets_theta"]

    overconfident = summarize_gap(0.90, 0.82, 0.9)
    assert overconfident["mass_coverage_gap"] > 0
    assert math.isclose(overconfident["coverage_shortfall"], 0.08, abs_tol=1e-9)
    assert not overconfident["meets_theta"]


def _load_runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_experiment_7_correlated_noise.py"
    )
    spec = importlib.util.spec_from_file_location("experiment7_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys_modules = __import__("sys").modules
    sys_modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(sample: str, method: str, **overrides):
    row = {
        "protocol_version": "exp7-correlated-noise-v1",
        "model": "Vendor/model",
        "domain": "prb",
        "configuration": "h01",
        "sample": sample,
        "calibration": "global",
        "method": method,
        "group_mode": "constraint",
        "block_size": 6,
        "rate": 0.2,
        "strength": 1.0,
        "theta": 0.9,
        "corruption_seed": 0,
        "status": "ok",
        "set_size": 4,
        "true_mass": 0.93,
        "attributed_mass": 0.9,
        "mass_coverage_gap": -0.03,
        "coverage_shortfall": 0.0,
        "meets_theta": True,
        "contains_truth": True,
        "total_variation": 0.2,
    }
    if method == "oracle_joint":
        row["certified"] = True
    row.update(overrides)
    return row


def _complete_cell(sample: str, **overrides):
    rows = [_row(sample, "belief", **overrides)]
    for method in ("mbs_prevas", "oracle_joint", "valid_map", "map"):
        rows.append(_row(sample, method, **overrides))
    return rows


def test_aggregate_reports_one_cell_per_complete_sample() -> None:
    runner = _load_runner_module()
    records = _complete_cell("s1") + _complete_cell("s2")
    aggregate = runner.aggregate_experiment_7(records, fingerprint="frozen")
    assert aggregate["analysis_decision"]["decision"] == "analysis-ready"
    assert aggregate["complete_cells"] == 2
    cell = aggregate["sweep"]["prb:constraint:0.2:1.0:0.9"]
    assert cell["n"] == 2
    assert math.isclose(cell["mass_coverage_gap"]["mean"], -0.03, abs_tol=1e-9)
    assert cell["meets_theta_rate"] == 1.0


def test_uncertified_oracle_removes_the_cell_for_every_method() -> None:
    # An uncertified prefix is not an exact comparator, so keeping the other
    # methods would silently break the pairing.
    runner = _load_runner_module()
    records = _complete_cell("s1") + _complete_cell("s2")
    for row in records:
        if row["sample"] == "s2" and row["method"] == "oracle_joint":
            row["certified"] = False
            row["status"] = "uncertified"
    aggregate = runner.aggregate_experiment_7(records, fingerprint="frozen")
    assert aggregate["complete_cells"] == 1
    assert aggregate["sweep"]["prb:constraint:0.2:1.0:0.9"]["n"] == 1


def test_exclusion_row_removes_the_cell_for_every_method() -> None:
    runner = _load_runner_module()
    records = _complete_cell("s1") + _complete_cell("s2")
    records.append(_row("s2", "common_exclusion", status="common_exclusion"))
    aggregate = runner.aggregate_experiment_7(records, fingerprint="frozen")
    assert aggregate["complete_cells"] == 1
    assert aggregate["analysis_decision"]["excluded_cells"] == 1


def test_aggregate_requires_every_method_before_counting_a_cell() -> None:
    runner = _load_runner_module()
    records = _complete_cell("s1")
    records = [row for row in records if row["method"] != "valid_map"]
    aggregate = runner.aggregate_experiment_7(records, fingerprint="frozen")
    assert aggregate["complete_cells"] == 0
    assert aggregate["analysis_decision"]["decision"] == "insufficient-data"
