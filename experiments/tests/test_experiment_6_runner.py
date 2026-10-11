from __future__ import annotations

import importlib.util
from pathlib import Path

from mbs_experiments.experiment6 import (
    PRIMARY_RELEVANCE_VERSION,
    CandidateUnit,
    abstention_completions,
    abstention_completions_by_count,
    controlled_probabilities,
    enumerate_controlled_ordering,
    extract_candidate_units,
    freeze_treatment,
    orient_treatment,
    project_constraint_groups,
    search_adaptive_abstention,
    uniform_hidden_probabilities,
    update_group_observation,
)
from mbs_experiments.pddl_truth import pddl_ground_truth, pddl_initial_truth

PROBLEM = """
(define (problem hidden-key)
 (:domain alf)
 (:objects
   key_1 - item
   drawer_1 drawer_2 table_1 - receptacle
   keytype - otype
   drawertype tabletype - rtype)
 (:init
   (objecttype key_1 keytype)
   (receptacletype drawer_1 drawertype)
   (receptacletype drawer_2 drawertype)
   (receptacletype table_1 tabletype)
   (cancontain drawertype keytype)
   (cancontain tabletype keytype)
   (openable drawer_1)
   (openable drawer_2)
   (isvisiblereceptacle table_1)
   (inreceptacle key_1 drawer_1))
 (:goal (and (isclean key_1) (inreceptacle key_1 table_1)))
)
"""


def _units() -> tuple[CandidateUnit, ...]:
    return (
        CandidateUnit(
            "location(key_1)",
            "categorical",
            (
                "inreceptacle(key_1,drawer_1)",
                "inreceptacle(key_1,drawer_2)",
            ),
            "hidden object",
            "goal support",
        ),
        CandidateUnit(
            "opened(drawer_1)",
            "binary",
            ("opened(drawer_1)",),
            "hidden receptacle",
            "action support",
        ),
    )


def test_candidate_extraction_uses_structure_and_visibility_not_hidden_values() -> None:
    before = extract_candidate_units(
        PROBLEM,
        sample_id="alf/all/sample_x",
        sample_seed=0,
        relevance_version=PRIMARY_RELEVANCE_VERSION,
    )
    # Change the hidden initial location and opened label. Inventory and ranks
    # must remain frozen because neither value participates in extraction.
    changed = PROBLEM.replace(
        "(inreceptacle key_1 drawer_1)",
        "(inreceptacle key_1 drawer_2)\n   (opened drawer_1)",
        1,
    )
    after = extract_candidate_units(
        changed,
        sample_id="alf/all/sample_x",
        sample_seed=0,
        relevance_version=PRIMARY_RELEVANCE_VERSION,
    )
    assert before == after
    assert {unit.unit_id for unit in before} >= {
        "location(key_1)",
        "opened(drawer_1)",
    }
    assert all("key_1" in unit.unit_id or "drawer" in unit.unit_id for unit in before)


def test_held_goal_object_is_operationally_visible_and_not_a_location_unit() -> None:
    held_problem = PROBLEM.replace(
        "(inreceptacle key_1 drawer_1)",
        "(holds key_1)",
        1,
    )
    units = extract_candidate_units(
        held_problem,
        sample_id="alf/all/sample_held",
        sample_seed=0,
        relevance_version=PRIMARY_RELEVANCE_VERSION,
    )
    assert not any(unit.unit_id == "location(key_1)" for unit in units)


def test_pddl_initial_state_is_the_label_source_not_trajectory_ground_truth() -> None:
    truth = pddl_initial_truth(PROBLEM)
    assert "inreceptacle(key_1,drawer_1)" in truth
    assert "inreceptacle(key_1,drawer_2)" not in truth
    projected = pddl_ground_truth(
        (
            "inreceptacle(key_1,drawer_1)",
            "inreceptacle(key_1,drawer_2)",
            "opened(drawer_1)",
        ),
        PROBLEM,
    )
    assert projected == {
        "inreceptacle(key_1,drawer_1)": True,
        "inreceptacle(key_1,drawer_2)": False,
        "opened(drawer_1)": False,
    }


def test_frozen_masks_are_reproducible_and_query_is_corruption_independent() -> None:
    units = _units()
    m0 = freeze_treatment(
        units,
        sample_id="sample",
        corruption_count=0,
        corruption_seed=7,
        query_seed=11,
    )
    m1 = freeze_treatment(
        units,
        sample_id="sample",
        corruption_count=1,
        corruption_seed=7,
        query_seed=11,
    )
    repeated = freeze_treatment(
        units,
        sample_id="sample",
        corruption_count=1,
        corruption_seed=7,
        query_seed=11,
    )
    assert m1 == repeated
    assert m0.queried_unit_id == m1.queried_unit_id
    assert len(m1.corrupted_unit_ids) == 1


def test_controlled_prior_orients_aligned_and_corrupted_units_exactly() -> None:
    units = _units()
    names = tuple(atom for unit in units for atom in unit.atoms)
    gt = {
        "inreceptacle(key_1,drawer_1)": True,
        "inreceptacle(key_1,drawer_2)": False,
        "opened(drawer_1)": False,
    }
    treatment = freeze_treatment(
        units,
        sample_id="sample",
        corruption_count=1,
        corruption_seed=0,
        query_seed=0,
    )
    oriented = orient_treatment(
        names,
        units,
        treatment,
        gt,
        sensor_confidence=0.6,
        sample_id="sample",
        corruption_seed=0,
    )
    index = {name: i for i, name in enumerate(names)}
    corrupted = treatment.corrupted_unit_ids[0]
    if corrupted == "location(key_1)":
        assert oriented.probabilities[index["inreceptacle(key_1,drawer_1)"]] == 0.4
        assert oriented.probabilities[index["inreceptacle(key_1,drawer_2)"]] == 0.6
        assert oriented.probabilities[index["opened(drawer_1)"]] == 0.4
    else:
        assert oriented.probabilities[index["inreceptacle(key_1,drawer_1)"]] == 0.6
        assert oriented.probabilities[index["inreceptacle(key_1,drawer_2)"]] == 0.4
        assert oriented.probabilities[index["opened(drawer_1)"]] == 0.6
    assert dict(oriented.decoy_assignments) == {
        "inreceptacle(key_1,drawer_1)": False,
        "inreceptacle(key_1,drawer_2)": True,
        "opened(drawer_1)": True,
    }


def test_group_observation_collapses_group_and_leaves_other_units_unchanged() -> None:
    units = _units()
    names = tuple(atom for unit in units for atom in unit.atoms)
    prior = (0.4, 0.6, 0.6)
    gt = {
        "inreceptacle(key_1,drawer_1)": True,
        "inreceptacle(key_1,drawer_2)": False,
        "opened(drawer_1)": False,
    }
    posterior = update_group_observation(
        prior,
        names,
        units,
        "location(key_1)",
        gt,
    )
    assert posterior[0] == 1.0 - 1e-6
    assert posterior[1] == 1e-6
    assert posterior[2] == prior[2]


def test_exact_ordering_is_complete_valid_sorted_and_overwrites_pddl_coordinates() -> None:
    units = _units()
    names = (
        "unrelated()",
        "inreceptacle(key_1,drawer_1)",
        "inreceptacle(key_1,drawer_2)",
        "opened(drawer_1)",
    )
    # The base values intentionally contain an impossible/leaky pair. Every
    # controlled coordinate must be overwritten during enumeration.
    base = (True, True, True, True)
    probabilities = (1.0 - 1e-6, 0.6, 0.4, 0.4)
    constraints = ((1, 2),)
    ordering = enumerate_controlled_ordering(
        probabilities,
        names,
        base,
        units,
        constraints,
        max_states=8,
    )
    assert len(ordering.states) == 4
    assert all(state[0] for state in ordering.states)
    assert all(sum(state[i] for i in (1, 2)) == 1 for state in ordering.states)
    assert ordering.cumulative_constrained_masses[-1] == 1.0
    assert list(ordering.state_unconstrained_masses) == sorted(
        ordering.state_unconstrained_masses, reverse=True
    )


def test_non_target_features_are_clamped_to_common_base_not_ground_truth() -> None:
    units = _units()
    names = (
        "unrelated()",
        "inreceptacle(key_1,drawer_1)",
        "inreceptacle(key_1,drawer_2)",
        "opened(drawer_1)",
    )
    base = (False, False, False, False)
    oriented_probabilities = (0.0, 0.6, 0.4, 0.4)
    class Oriented:
        probabilities = oriented_probabilities

    probabilities = controlled_probabilities(names, base, units, Oriented())
    assert probabilities[0] == 1e-6
    assert probabilities[1:] == oriented_probabilities[1:]


def test_uniform_hidden_prior_sets_every_declared_unit_atom_to_half() -> None:
    units = _units()
    names = (
        "unrelated()",
        "inreceptacle(key_1,drawer_1)",
        "inreceptacle(key_1,drawer_2)",
        "opened(drawer_1)",
    )
    probabilities = uniform_hidden_probabilities(
        names,
        (True, True, False, True),
        units,
    )
    assert probabilities[0] == 1.0 - 1e-6
    assert probabilities[1:] == (0.5, 0.5, 0.5)


def test_abstention_materializes_all_valid_completions_under_frozen_threshold() -> None:
    units = _units()
    names = tuple(atom for unit in units for atom in unit.atoms)
    base = (False, False, False)
    probabilities = (0.6, 0.4, 0.6)
    ordering = enumerate_controlled_ordering(
        probabilities,
        names,
        base,
        units,
        ((0, 1),),
        max_states=8,
    )
    states, abstained, logical = abstention_completions(
        ordering,
        probabilities,
        names,
        units,
        threshold=0.61,
        max_states=8,
    )
    assert abstained == 2
    assert logical == 4
    assert states == ordering.states


def test_count_abstention_has_exact_map_fallback_under_uniform_ties() -> None:
    units = _units()
    names = tuple(atom for unit in units for atom in unit.atoms)
    ordering = enumerate_controlled_ordering(
        (0.5, 0.5, 0.5),
        names,
        (False, False, False),
        units,
        ((0, 1),),
        max_states=8,
    )
    singleton, abstained_ids, logical = abstention_completions_by_count(
        ordering,
        (0.5, 0.5, 0.5),
        names,
        units,
        abstained_count=0,
        max_states=8,
    )
    assert singleton == ordering.states[:1]
    assert abstained_ids == ()
    assert logical == 1

    full, abstained_ids, logical = abstention_completions_by_count(
        ordering,
        (0.5, 0.5, 0.5),
        names,
        units,
        abstained_count=len(units),
        max_states=8,
    )
    assert full == ordering.states
    assert set(abstained_ids) == {unit.unit_id for unit in units}
    assert logical == 4


def test_adaptive_abstention_retains_map_after_larger_candidate_failure() -> None:
    from mbs_experiments.planning import PlanningResult

    units = _units()
    names = tuple(atom for unit in units for atom in unit.atoms)
    ordering = enumerate_controlled_ordering(
        (0.5, 0.5, 0.5),
        names,
        (False, False, False),
        units,
        ((0, 1),),
        max_states=8,
    )

    def result(status: str) -> PlanningResult:
        return PlanningResult(
            status=status,
            planner_status=None,
            runtime_seconds=0.1,
            plan_length=1 if status == "ok" else None,
            actions=("act",) if status == "ok" else (),
            success=status == "ok",
            unsafe_first=False,
            pre_fail=False,
            no_plan_fn=False,
        )

    singleton = result("ok")
    adaptive = search_adaptive_abstention(
        ordering,
        (0.5, 0.5, 0.5),
        names,
        units,
        lambda _states: result("timeout"),
        singleton_result=singleton,
        max_states=8,
    )
    assert adaptive.selected.abstained_count == 0
    assert adaptive.selected.completions == ordering.states[:1]
    assert adaptive.selected.planning_result is singleton
    assert adaptive.attempts[0].planning_result is singleton


def test_constraint_projection_is_identical_for_every_method_input() -> None:
    names = ("a()", "b()", "c()")
    raw = [
        {"type": "at_most_one", "atoms": ["a()", "b()", "outside()"]},
        {"type": "at_most_one", "atoms": ["b()", "c()"]},
    ]
    assert project_constraint_groups(raw, names) == ((0, 1), (1, 2))


def _load_runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_experiment_6_controlled_prior.py"
    )
    spec = importlib.util.spec_from_file_location("experiment6_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys_modules = __import__("sys").modules
    sys_modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _aggregate_row(
    sample: str,
    corruption_count: int,
    method: str,
    round_index: int,
    *,
    success: bool = False,
    actions: tuple[str, ...] = (),
    status: str = "ok",
):
    return {
        "sample_path": sample,
        "calibration": "global",
        "corruption_count": corruption_count,
        "method": method,
        "round": round_index,
        "success": success,
        "plan_found": status == "ok",
        "actions": list(actions),
        "status": status,
        "belief_size": 1,
        "runtime_seconds": 0.1,
        "query_hit_corruption": False,
    }


def test_aggregation_preserves_paired_denominators_and_applies_go_rule() -> None:
    runner = _load_runner_module()
    records = []
    for sample in ("a", "b"):
        for corruption_count in (0, 1, 2):
            for round_index in (0, 1):
                records.extend(
                    (
                        _aggregate_row(
                            sample,
                            corruption_count,
                            "map",
                            round_index,
                            success=True,
                            actions=("map",),
                        ),
                        _aggregate_row(
                            sample,
                            corruption_count,
                            "ams",
                            round_index,
                            success=True,
                            actions=("ams",),
                        ),
                        _aggregate_row(
                            sample,
                            corruption_count,
                            "adaptive_abstention",
                            round_index,
                            success=True,
                            actions=("abstain",),
                        ),
                    )
                )
    aggregate = runner.aggregate_experiment_6(
        records,
        fingerprint="frozen",
        bootstrap_seed=0,
        planning_timeout=1.0,
    )
    assert aggregate["complete_sample_conditions"] == 6
    assert aggregate["denominator_mismatches"] == []
    assert aggregate["method_round"]["m2:ams:k1"]["n"] == 2
    assert "true_state_inclusion_rate" in aggregate["method_round"]["m2:ams:k1"]
    assert "m2:k1" in aggregate["representation"]
    assert (
        aggregate["plan_discrimination"]["m2:k1"][
            "any_method_plan_disagreement_rate"
        ]
        == 1
    )
    assert aggregate["phase_e"]["criteria"]["plan_disagreement_at_least_20pct"]
    assert aggregate["phase_e"]["decision"] == "go"


def test_uniform_hidden_aggregation_uses_analysis_ready_integrity_gate() -> None:
    runner = _load_runner_module()
    records = []
    for sample_index in range(100):
        sample = f"sample_{sample_index:03d}"
        for round_index in (0, 1):
            for method in ("map", "ams", "adaptive_abstention"):
                row = _aggregate_row(
                    sample,
                    0,
                    method,
                    round_index,
                    success=True,
                )
                row.update(
                    {
                        "prior_mode": "uniform_hidden",
                        "confidence": 0.75,
                        "belief_size": 8 if method != "map" else 1,
                    }
                )
                if method == "ams":
                    row.update(
                        {
                            "ams_search_runtime_seconds": 2.0,
                            "attempts": [{}, {}],
                        }
                    )
                elif method == "adaptive_abstention":
                    row.update(
                        {
                            "abstention_search_runtime_seconds": 1.0,
                            "map_singleton_reused": True,
                            "attempts": [
                                {
                                    "abstained_count": 0,
                                    "belief_size": 1,
                                }
                            ],
                        }
                    )
                records.append(row)
    aggregate = runner.aggregate_experiment_6(
        records,
        fingerprint="uniform",
        bootstrap_seed=0,
        planning_timeout=30.0,
    )
    assert aggregate["phase_e"] is None
    assert aggregate["analysis_decision"]["decision"] == "analysis-ready"
    assert aggregate["adaptive_invariants"]["m0"]["satisfied"]
    assert (
        aggregate["method_efficiency"]["m0:ams:k0"][
            "mean_probability_mass"
        ]
        == 0.75
    )
    assert (
        aggregate["paired_success_differences"][
            "m0:k0:adaptive_abstention-map"
        ]["mean"]
        == 0
    )


def test_aggregation_keeps_timeout_as_success_zero_in_common_denominator() -> None:
    runner = _load_runner_module()
    records = []
    for method in ("map", "ams", "adaptive_abstention"):
        for round_index in (0, 1):
            records.append(
                _aggregate_row(
                    "a",
                    2,
                    method,
                    round_index,
                    success=False,
                    status="timeout" if method == "ams" else "no_plan",
                )
            )
    aggregate = runner.aggregate_experiment_6(
        records,
        fingerprint="frozen",
        bootstrap_seed=0,
        planning_timeout=1.0,
    )
    assert aggregate["complete_sample_conditions"] == 1
    assert aggregate["method_round"]["m2:ams:k1"]["n"] == 1
    assert aggregate["method_round"]["m2:ams:k1"]["success_rate"] == 0
