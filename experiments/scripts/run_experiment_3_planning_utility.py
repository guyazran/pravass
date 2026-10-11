#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import math
import multiprocessing as mp
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from mbs_experiments.abstention import load_selected_tau_lookup
from mbs_experiments.baselines import (
    THETA_GRID,
    adaptive_fluent_abstention_tau_candidates,
    fluent_abstention_unconstrained,
    fluent_prediction_marginals,
    map_baseline,
    mbs_prevas,
    random_theta_confident,
    unconstrained_top_mass_baseline,
    valid_map,
)
from mbs_experiments.data import (
    SampleId,
    available_calibrations,
    filter_sample_ids,
    load_calibration_bundle,
    load_test_sample_ids,
    model_to_path_segment,
    prediction_path,
)
from mbs_experiments.io import (
    append_jsonl,
    iter_jsonl,
    load_json,
    make_error_record,
    result_key,
    write_json_atomic,
    write_jsonl_atomic,
)
from mbs_experiments.metrics import aggregate_experiment_3
from mbs_experiments.normalizers import (
    exact_constraints_runtime_normalizer,
    implied_selected_groups_normalizer,
    selected_groups_runtime_normalizer,
)
from mbs_experiments.planning import (
    PlanningResult,
    run_belief_planning,
    run_oracle_planning,
    run_singleton_classical_planning,
)
from mbs_experiments.states import (
    State,
    StateSet,
    logical_full_state_count,
    null_indices,
    project_feature_names,
    project_groups,
    state_set_size,
    iter_state_set,
    unconstrained_probability,
    visible_indices,
)

# Outputs are model-sharded: experiments_data/experiment_3_planning_utility/
# <model>/{sample_results.jsonl,aggregate_metrics.json}. Combined with the
# `model` field in every record + RESULT_KEY_FIELDS, results for different
# models never collide, mix, or silently resume each other.
OUTPUT_ROOT = Path("experiments_data/experiment_3_planning_utility")


def experiment_paths(model: str) -> tuple[Path, Path, Path]:
    """Return (output_dir, results_path, agg_path) for a model."""
    base = OUTPUT_ROOT / model_to_path_segment(model)
    return base, base / "sample_results.jsonl", base / "aggregate_metrics.json"


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
ABSTENTION_ROOT = Path("abstention_data")
TEST_SPLIT = "test"
# `model` is last (so `key[:4]` domain/config/sample/calibration prefixing
# still works) and makes every row self-describing and model-safe even if
# files are merged.
RESULT_KEY_FIELDS = (
    "domain",
    "configuration",
    "sample",
    "calibration",
    "method",
    "theta",
    "model",
)
ATOM_RE = re.compile(r"^([^()]+)\((.*)\)$")


def _parse_atom(atom: str) -> tuple[str, tuple[str, ...]]:
    match = ATOM_RE.match(atom)
    if match is None:
        raise ValueError(f"Invalid atom: {atom!r}")
    raw_args = match.group(2)
    args = tuple(part for part in raw_args.split(",") if part)
    return match.group(1), args


def prb_domain_adjusted_probabilities(
    probabilities: tuple[float, ...],
    feature_names: tuple[str, ...],
) -> tuple[float, ...]:
    """Clamp PRB probabilities for hard type/equality invariants.

    The PRB predicate vocabulary contains syntactically possible but physically
    impossible atoms, notably self-support ``shape-on-top-of-block(b,b)``. Also,
    shapes that never appear as a support block (spheres in this benchmark) are
    always clear because no action can place anything on them. These are domain
    invariants, not learned labels, so enforcing them before constrained MBS is
    fair and avoids wasting belief mass on unreachable states.
    """
    support_blocks: set[str] = set()
    parsed = []
    for atom in feature_names:
        pred, args = _parse_atom(atom)
        parsed.append((pred, args))
        if pred == "shape-on-top-of-block" and len(args) == 2:
            support_blocks.add(args[1])

    adjusted = list(map(float, probabilities))
    for i, (pred, args) in enumerate(parsed):
        if pred == "shape-on-top-of-block" and len(args) == 2 and args[0] == args[1]:
            adjusted[i] = 0.0
        elif pred == "is-clear-on-top" and len(args) == 1 and args[0] not in support_blocks:
            adjusted[i] = 1.0
    return tuple(adjusted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 3 downstream planning utility")
    parser.add_argument("--domains", nargs="+", default=("prb", "gw"))
    parser.add_argument("--configs", nargs="+")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model identifier whose predictions / Platt profiles / calibrations "
            "to consume. Must match the --model passed to make_predictions.py, "
            "calibrate_vlm.py, calibrate_tau.py. "
            f"Default: {DEFAULT_MODEL}."
        ),
    )
    parser.add_argument(
        "--calibrations",
        nargs="+",
        default=("global", "lifted", "none"),
        help=(
            "Calibration scopes. 'none' uses raw probabilities (no Platt "
            "scaling). Default: global lifted none."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reset-methods",
        nargs="+",
        default=(),
        help="Drop existing rows for these methods before resuming, so they are recomputed.",
    )
    parser.add_argument(
        "--planning-timeout",
        type=float,
        default=30.0,
        help=(
            "Wallclock timeout (seconds) per planning run. The CPORLib "
            "planner fails ~88%% of the time and consumes the full budget "
            "when it does, so this is the per-call budget, not a rare-explosion "
            "ceiling. Default: 30."
        ),
    )
    parser.add_argument(
        "--method-timeout",
        type=float,
        default=30.0,
        help=(
            "Wallclock timeout (seconds) applied to belief-set construction, "
            "as a backstop so even a pathological PRaVaSS run is bounded. "
            "Default: 30."
        ),
    )
    parser.add_argument("--max-belief-states", type=int, default=200)
    parser.add_argument(
        "--abstention-root",
        type=Path,
        default=ABSTENTION_ROOT,
        help=(
            "Root of the sharded τ calibration tree "
            "(<root>/<model>/<calibration>/<domain>/tau_calibration.json) "
            "written by calibrate_tau.py."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--jobs",
        "--workers",
        dest="jobs",
        type=int,
        default=None,
        help=(
            "Number of parallel sample/calibration worker processes "
            "(default: CPU count - 1; use 1 for sequential)."
        ),
    )
    return parser.parse_args()


SINGLETON_METHODS = frozenset({"map"})#, "valid_map"})
PREVAS_PREFIX_SEARCH_METHOD = "mbs_prevas_prefix_search"
PREVAS_PREFIX_SEARCH_CONSTRAINTS_EXACT_METHOD = (
    "mbs_prevas_prefix_search_constraints_exact"
)
PREVAS_PREFIX_SEARCH_CONSTRAINTS_RELAXED_METHOD = (
    "mbs_prevas_prefix_search_constraints_relaxed"
)
PREVAS_PREFIX_SEARCH_THETA = 0.99
ADAPTIVE_FLUENT_ABSTENTION_METHOD = "adaptive_fluent_abstention"
FLUENT_ABSTENTION_METHOD = "fluent_abstention_unconstrained"
PLANNING_CONSTRAINTS_VERSION = 1
FAILURE_PRUNED_THETA_METHODS = frozenset(
    {
        "mbs_prevas",
        "mbs_prevas_constraints_exact",
        "mbs_prevas_constraints_relaxed",
        "unconstrained_top_mass",
        FLUENT_ABSTENTION_METHOD,
    }
)


def failure_pruning_enabled(record_method: str) -> bool:
    """Return whether one failed parameter value can safely skip later ones."""
    return record_method in FAILURE_PRUNED_THETA_METHODS


def singleton_baseline_inputs(bundle) -> tuple[
    tuple[str, ...],
    tuple[float, ...],
    tuple[tuple[int, ...], ...],
    float,
    tuple[str, ...],
]:
    """Return full-feature inputs for MAP and Valid-MAP baselines.

    Unlike MBS/PRaVaSS belief sets, singleton baselines must commit to a value
    for every feature, including features whose raw prediction mass was None.
    Those features are represented in the bundle with probability 0.5.
    """
    feature_names = tuple(bundle.feature_names)
    probabilities = tuple(bundle.probabilities)
    groups = tuple(tuple(group) for group in bundle.selected_groups)
    normalizer = selected_groups_runtime_normalizer(
        bundle, feature_space="full"
    )
    omitted_feature_names: tuple[str, ...] = ()
    return feature_names, probabilities, groups, normalizer, omitted_feature_names


def stale_singleton_baseline_keys(
    existing_records: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> set[tuple[Any, ...]]:
    stale: set[tuple[Any, ...]] = set()
    for key, record in existing_records.items():
        if record.get("method") not in SINGLETON_METHODS:
            continue
        if int(record.get("omitted_feature_count") or 0) > 0:
            stale.add(key)
    return stale


def stale_fluent_abstention_tau_keys(
    existing_records: Mapping[tuple[Any, ...], Mapping[str, Any]],
    tau_lookup: Mapping[tuple[str, str, float], float],
    *,
    domains: set[str] | None = None,
    calibrations: set[str] | None = None,
) -> set[tuple[Any, ...]]:
    """Return fluent-abstention records whose stored tau is absent or stale."""
    stale: set[tuple[Any, ...]] = set()
    for key, record in existing_records.items():
        method = record.get("method")
        if str(method) != FLUENT_ABSTENTION_METHOD:
            continue
        domain = record.get("domain")
        calibration = record.get("calibration")
        if domains is not None and domain not in domains:
            continue
        if calibrations is not None and calibration not in calibrations:
            continue
        theta = record.get("theta")
        if theta is None or domain is None or calibration is None:
            continue
        expected_tau = tau_lookup.get(
            (str(domain), str(calibration), float(theta))
        )
        if expected_tau is None:
            continue
        recorded_tau = record.get("abstention_tau")
        try:
            tau_matches = recorded_tau is not None and math.isclose(
                float(recorded_tau),
                float(expected_tau),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        except (TypeError, ValueError):
            tau_matches = False
        if not tau_matches:
            stale.add(key)
    return stale


def planning_record(
    bundle,
    method: str,
    result: PlanningResult,
    *,
    theta: float | None,
    belief_size: int | None,
    search_feature_count: int | None = None,
    omitted_feature_count: int = 0,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "status": result.status,
        "domain": bundle.sample_id.domain,
        "configuration": bundle.sample_id.configuration,
        "sample": bundle.sample_id.sample,
        "split_relative_path": bundle.sample_id.split_relative_path,
        "calibration": bundle.calibration,
        "model": bundle.model,
        "method": method,
        "theta": theta,
        "feature_count": len(bundle.feature_names),
        "search_feature_count": search_feature_count
        if search_feature_count is not None
        else len(bundle.feature_names),
        "omitted_feature_count": omitted_feature_count,
        "belief_size": belief_size,
        "logical_full_belief_size": None
        if belief_size is None
        else logical_full_state_count(belief_size, omitted_feature_count),
        "planner_status": result.planner_status,
        "runtime_seconds": result.runtime_seconds,
        "plan_length": result.plan_length,
        "actions": list(result.actions),
        "success": result.success,
        "unsafe_first": result.unsafe_first,
        "pre_fail": result.pre_fail,
        "no_plan_fn": result.no_plan_fn,
        "error_message": result.error_message,
    }
    if extra:
        record.update(dict(extra))
    return record


def unavailable_record(
    bundle,
    method: str,
    *,
    theta: float | None,
    belief_size: int,
    oracle_success: bool,
    message: str,
    status: str = "unsupported",
    search_feature_count: int | None = None,
    omitted_feature_count: int = 0,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "status": status,
        "domain": bundle.sample_id.domain,
        "configuration": bundle.sample_id.configuration,
        "sample": bundle.sample_id.sample,
        "split_relative_path": bundle.sample_id.split_relative_path,
        "calibration": bundle.calibration,
        "model": bundle.model,
        "method": method,
        "theta": theta,
        "feature_count": len(bundle.feature_names),
        "search_feature_count": search_feature_count
        if search_feature_count is not None
        else len(bundle.feature_names),
        "omitted_feature_count": omitted_feature_count,
        "belief_size": belief_size,
        "logical_full_belief_size": logical_full_state_count(
            belief_size, omitted_feature_count
        ),
        "planner_status": None,
        "runtime_seconds": None,
        "plan_length": None,
        "actions": [],
        "success": False,
        "unsafe_first": False,
        "pre_fail": False,
        "no_plan_fn": bool(oracle_success),
        "error_message": message,
    }
    if extra:
        record.update(dict(extra))
    return record


def load_latest_records(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in iter_jsonl(path):
        latest[result_key(record, RESULT_KEY_FIELDS)] = record
    return latest


@dataclass(frozen=True)
class Experiment3Config:
    planning_timeout: float
    method_timeout: float
    max_belief_states: int
    random_seed: int


@dataclass(frozen=True)
class Experiment3Task:
    sid: SampleId
    calibration: str
    config: Experiment3Config
    model: str
    tau_lookup: dict[tuple[str, str, float], float]
    existing_records: dict[tuple[Any, ...], dict[str, Any]]


@dataclass(frozen=True)
class PrefixSearchPlanningOutcome:
    result: PlanningResult
    belief_size: int
    extra: dict[str, Any]


def stable_seed(*parts: object) -> int:
    text = "|".join(map(str, parts))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) & 0xFFFFFFFF


def constraint_group_names(
    feature_names: tuple[str, ...], groups: tuple[tuple[int, ...], ...]
) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(feature_names[idx] for idx in group) for group in groups)


def experiment_key(
    sid: SampleId,
    calibration: str | None,
    method: str,
    theta: float | None,
    *,
    model: str,
) -> tuple[Any, ...]:
    # Order must match RESULT_KEY_FIELDS (model last).
    return (
        sid.domain,
        sid.configuration,
        sid.sample,
        calibration,
        method,
        theta,
        model,
    )


def _base_error_context(
    sid: SampleId,
    calibration: str | None,
    method: str,
    theta: float | None,
    *,
    model: str,
) -> dict[str, Any]:
    return {
        "domain": sid.domain,
        "configuration": sid.configuration,
        "sample": sid.sample,
        "split_relative_path": sid.split_relative_path,
        "calibration": calibration,
        "method": method,
        "theta": theta,
        "model": model,
    }


def _existing_records_for_task(
    existing_records: Mapping[tuple[Any, ...], dict[str, Any]],
    sid: SampleId,
    calibration: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    prefix = (sid.domain, sid.configuration, sid.sample, calibration)
    return {key: record for key, record in existing_records.items() if key[:4] == prefix}


def _plan_found(result: PlanningResult) -> bool:
    return result.status == "ok"


def _planning_result_with_runtime(
    result: PlanningResult,
    runtime_seconds: float,
) -> PlanningResult:
    return PlanningResult(
        status=result.status,
        planner_status=result.planner_status,
        runtime_seconds=runtime_seconds,
        plan_length=result.plan_length,
        actions=result.actions,
        success=result.success,
        unsafe_first=result.unsafe_first,
        pre_fail=result.pre_fail,
        no_plan_fn=result.no_plan_fn,
        error_message=result.error_message,
    )


def _final_plan_set_mass_extra(
    *,
    plan_found: bool,
    belief_size: int,
    unconstrained_mass: float,
    constrained_mass: float,
) -> dict[str, Any]:
    """Mass of the exact belief set used to generate the returned plan."""
    return {
        "final_plan_set_size": belief_size if plan_found else None,
        "final_plan_set_unconstrained_mass": unconstrained_mass if plan_found else None,
        "final_plan_set_constrained_mass": constrained_mass if plan_found else None,
    }


def run_mbs_prevas_prefix_search_planning(
    bundle,
    probabilities: tuple[float, ...],
    selected_groups: tuple[tuple[int, ...], ...],
    normalizer: float,
    feature_names: tuple[str, ...],
    omitted_feature_names: tuple[str, ...],
    *,
    method_timeout: float,
    planning_timeout: float,
    oracle_success: bool,
    exactly_one: bool = False,
    normalizer_exact: bool = True,
    planning_constraint_groups: tuple[tuple[str, ...], ...] = (),
    planning_constraints_exactly_one: bool = False,
) -> PrefixSearchPlanningOutcome:
    """AMS: PRaVaSS (θ = .99) followed by a largest-plannable-prefix search.

    PRaVaSS (the pravass package) yields valid states in non-increasing
    probability order. AMS keeps that order, accepts a PRaVaSS timeout as a
    partial prefix, and then searches for the largest prefix for which the
    conformant planner returns a plan.
    """
    start = time.perf_counter()
    prevas_result = mbs_prevas(
        probabilities,
        selected_groups,
        PREVAS_PREFIX_SEARCH_THETA,
        normalizer,
        timeout_seconds=method_timeout,
        exactly_one=exactly_one,
        exact=normalizer_exact,
    )
    state_set = prevas_result.state_set
    if state_set.status not in {"ok", "timeout"}:
        raise RuntimeError(f"PRaVaSS state set not available: {state_set.status}")

    states: tuple[State, ...] = tuple(state_set.states)
    state_probabilities = tuple(
        unconstrained_probability(state, probabilities) for state in states
    )
    prefix_probability_sums: list[float] = [0.0]
    for probability in state_probabilities:
        prefix_probability_sums.append(prefix_probability_sums[-1] + probability)

    attempts: list[dict[str, Any]] = []

    def attempt_plan(prefix_size: int, phase: str) -> PlanningResult:
        prefix_unconstrained_mass = prefix_probability_sums[prefix_size]
        result = run_belief_planning(
            bundle.conditional_domain_path,
            bundle.problem_path,
            feature_names,
            states[:prefix_size],
            planning_timeout,
            oracle_success=oracle_success,
            unknown_feature_names=omitted_feature_names,
            initial_constraint_groups=planning_constraint_groups,
            initial_constraints_exactly_one=planning_constraints_exactly_one,
        )
        attempts.append(
            {
                "phase": phase,
                "prefix_size": prefix_size,
                "unconstrained_mass": prefix_unconstrained_mass,
                "constrained_mass": prefix_unconstrained_mass / normalizer,
                "status": result.status,
                "planner_status": result.planner_status,
                "plan_found": _plan_found(result),
                "success": result.success,
                "runtime_seconds": result.runtime_seconds,
            }
        )
        return result

    n = len(states)
    full_result = attempt_plan(n, "full")
    best_result: PlanningResult | None = full_result if _plan_found(full_result) else None
    best_size = n if best_result is not None else 0
    binary_search_performed = False

    if best_result is None and n > 1:
        binary_search_performed = True
        # Predicate is monotone over prefixes: a plan for a larger belief prefix
        # is also a plan for every smaller prefix.  Maintain lo as the largest
        # plannable prefix found so far (0 is a sentinel) and hi as a known
        # unplannable prefix size (the full prefix was just tried and failed).
        lo = 0
        hi = n
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            mid_result = attempt_plan(mid, "binary_search")
            if _plan_found(mid_result):
                lo = mid
                best_size = mid
                best_result = mid_result
            else:
                hi = mid

    plan_found = best_result is not None
    final_result = best_result if plan_found else full_result
    final_belief_size = best_size if plan_found else n
    total_runtime = time.perf_counter() - start
    final_result = _planning_result_with_runtime(final_result, total_runtime)
    final_unconstrained_mass = prefix_probability_sums[final_belief_size]
    final_constrained_mass = final_unconstrained_mass / normalizer
    prevas_unconstrained_mass = prefix_probability_sums[n]

    return PrefixSearchPlanningOutcome(
        result=final_result,
        belief_size=final_belief_size,
        extra={
            "prevas_prefix_search_theta": PREVAS_PREFIX_SEARCH_THETA,
            "prevas_prefix_search_prevas_status": state_set.status,
            "prevas_prefix_search_prevas_timeout": prevas_result.stats.timeout,
            "prevas_prefix_search_prevas_stats": prevas_result.stats.__dict__,
            "prevas_prefix_search_returned_belief_size": n,
            "prevas_prefix_search_returned_unconstrained_mass": prevas_unconstrained_mass,
            "prevas_prefix_search_returned_constrained_mass": prevas_unconstrained_mass / normalizer,
            "prevas_prefix_search_final_belief_size": final_belief_size,
            "prevas_prefix_search_final_unconstrained_mass": final_unconstrained_mass,
            "prevas_prefix_search_final_constrained_mass": final_constrained_mass,
            "prevas_prefix_search_plan_found": plan_found,
            "prevas_prefix_search_binary_search_performed": binary_search_performed,
            "prevas_prefix_search_attempt_count": len(attempts),
            "prevas_prefix_search_attempts": attempts,
        }
        | _final_plan_set_mass_extra(
            plan_found=plan_found,
            belief_size=final_belief_size,
            unconstrained_mass=final_unconstrained_mass,
            constrained_mass=final_constrained_mass,
        ),
    )


def _unplanned_result(
    *,
    status: str,
    runtime_seconds: float,
    oracle_success: bool,
    message: str,
) -> PlanningResult:
    return PlanningResult(
        status=status,
        planner_status=None,
        runtime_seconds=runtime_seconds,
        plan_length=None,
        actions=(),
        success=False,
        unsafe_first=False,
        pre_fail=False,
        no_plan_fn=bool(oracle_success),
        error_message=message,
    )


def run_adaptive_fluent_abstention_planning(
    bundle,
    probabilities: tuple[float, ...],
    selected_groups: tuple[tuple[int, ...], ...],
    normalizer: float,
    feature_names: tuple[str, ...],
    omitted_feature_names: tuple[str, ...],
    *,
    method_timeout: float,
    planning_timeout: float,
    oracle_success: bool,
    max_belief_states: int | None = None,
) -> PrefixSearchPlanningOutcome:
    """Binary-search the largest plannable unconstrained fluent-abstention set."""
    start = time.perf_counter()
    marginals = fluent_prediction_marginals(probabilities)
    tau_candidates = adaptive_fluent_abstention_tau_candidates(marginals)
    attempts: list[dict[str, Any]] = []

    def attempt_plan(
        candidate_index: int,
    ) -> tuple[PlanningResult, StateSet, int, bool]:
        tau = tau_candidates[candidate_index]
        attempt_start = time.perf_counter()
        state_set = fluent_abstention_unconstrained(
            probabilities,
            tau,
            normalizer,
            selected_groups=selected_groups,
            timeout_seconds=method_timeout,
        )
        belief_size = state_set_size(state_set)
        skipped_reason: str | None = None
        if state_set.status != "ok":
            result = _unplanned_result(
                status="error" if state_set.status == "error" else state_set.status,
                runtime_seconds=time.perf_counter() - attempt_start,
                oracle_success=oracle_success,
                message="state_set_not_available",
            )
        elif max_belief_states is not None and belief_size > max_belief_states:
            skipped_reason = "belief_size_cap"
            result = _unplanned_result(
                status="unsupported",
                runtime_seconds=time.perf_counter() - attempt_start,
                oracle_success=oracle_success,
                message=skipped_reason,
            )
        else:
            result = run_belief_planning(
                bundle.conditional_domain_path,
                bundle.problem_path,
                feature_names,
                list(iter_state_set(state_set)),
                planning_timeout,
                oracle_success=oracle_success,
                unknown_feature_names=omitted_feature_names,
            )
        plan_found = _plan_found(result)
        attempt: dict[str, Any] = {
            "phase": "binary_search",
            "candidate_index": candidate_index,
            "tau": tau,
            "belief_size": belief_size,
            "unconstrained_mass": state_set.unconstrained_mass,
            "constrained_mass": state_set.constrained_mass,
            "status": result.status,
            "planner_status": result.planner_status,
            "plan_found": plan_found,
            "success": result.success,
            "runtime_seconds": result.runtime_seconds,
        }
        if skipped_reason is not None:
            attempt["skipped_reason"] = skipped_reason
        attempts.append(attempt)
        return result, state_set, belief_size, plan_found

    # Candidate indices are monotone: a plan for a larger abstention set also
    # works for every smaller one. Maintain lo as the largest plannable index
    # and hi as the smallest known-unplannable index.
    lo = -1
    hi = len(tau_candidates)
    best_outcome: tuple[PlanningResult, StateSet, int, bool] | None = None
    fallback_outcome: tuple[PlanningResult, StateSet, int, bool] | None = None
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        outcome = attempt_plan(mid)
        if outcome[3]:
            lo = mid
            best_outcome = outcome
        else:
            hi = mid
            fallback_outcome = outcome

    final_outcome = best_outcome if best_outcome is not None else fallback_outcome
    if final_outcome is None:
        # Defensive fallback for an empty candidate list; should be unreachable.
        result = _unplanned_result(
            status="unsupported",
            runtime_seconds=0.0,
            oracle_success=oracle_success,
            message="no_tau_candidates",
        )
        state_set = fluent_abstention_unconstrained(
            probabilities, 0.5, normalizer, selected_groups=selected_groups
        )
        final_outcome = (result, state_set, state_set_size(state_set), False)

    final_result, final_state_set, final_belief_size, plan_found = final_outcome
    total_runtime = time.perf_counter() - start
    final_result = _planning_result_with_runtime(final_result, total_runtime)
    final_unconstrained_mass = final_state_set.unconstrained_mass
    final_constrained_mass = final_state_set.constrained_mass
    final_tau = None
    for attempt in reversed(attempts):
        if (
            attempt["belief_size"] == final_belief_size
            and attempt["plan_found"] == plan_found
        ):
            final_tau = attempt["tau"]
            break

    return PrefixSearchPlanningOutcome(
        result=final_result,
        belief_size=final_belief_size,
        extra={
            "adaptive_fluent_abstention_unique_marginals": list(marginals),
            "adaptive_fluent_abstention_tau_candidates": list(tau_candidates),
            "adaptive_fluent_abstention_final_tau": final_tau,
            "adaptive_fluent_abstention_final_belief_size": final_belief_size,
            "adaptive_fluent_abstention_final_unconstrained_mass": final_unconstrained_mass,
            "adaptive_fluent_abstention_final_constrained_mass": final_constrained_mass,
            "adaptive_fluent_abstention_plan_found": plan_found,
            "adaptive_fluent_abstention_binary_search_performed": len(tau_candidates) > 1,
            "adaptive_fluent_abstention_attempt_count": len(attempts),
            "adaptive_fluent_abstention_attempts": attempts,
            "adaptive_fluent_abstention_belief_size_cap": max_belief_states,
        }
        | _final_plan_set_mass_extra(
            plan_found=plan_found,
            belief_size=final_belief_size,
            unconstrained_mass=final_unconstrained_mass,
            constrained_mass=final_constrained_mass,
        ),
    )


def _run_experiment_3_calibration_task(task: Experiment3Task) -> list[dict[str, Any]]:
    """Run all Experiment 3 methods for one sample/calibration.

    Workers never write to the shared JSONL.  They return records to the parent
    process, which serializes writes and keeps resume/aggregate state coherent.
    """
    sid = task.sid
    calibration = task.calibration
    config = task.config
    model = task.model
    tau_lookup = task.tau_lookup
    existing_records = dict(task.existing_records)
    existing = set(existing_records)
    records: list[dict[str, Any]] = []

    def record_result(record: Mapping[str, Any]) -> None:
        payload = dict(record)
        key = result_key(payload, RESULT_KEY_FIELDS)
        if key in existing:
            return
        records.append(payload)
        existing_records[key] = payload
        existing.add(key)

    bundle_key = experiment_key(sid, calibration, "bundle", None, model=model)
    try:
        bundle = load_calibration_bundle(sid, calibration, model=task.model, split=TEST_SPLIT)
        search_idx = visible_indices(bundle.none_is_max)
        omitted_idx = null_indices(bundle.none_is_max)
        search_feature_names = project_feature_names(bundle.feature_names, search_idx)
        omitted_feature_names = project_feature_names(bundle.feature_names, omitted_idx)
        reduced_probs = tuple(bundle.probabilities[i] for i in search_idx)
        reduced_groups = project_groups(bundle.selected_groups, search_idx)
        selected_z_start = time.perf_counter()
        reduced_z = selected_groups_runtime_normalizer(
            bundle, feature_space="visible"
        )
        selected_z_runtime = time.perf_counter() - selected_z_start
    except Exception as exc:
        if bundle_key not in existing:
            record_result(
                make_error_record(
                    _base_error_context(sid, calibration, "bundle", None, model=model), exc
                )
            )
        return records

    try:
        if sid.domain == "prb":
            # PRB has many none_is_max fluents. Leaving them as hidden unknowns
            # makes exact-one support/clear invariants unsound after projection
            # and makes conformant planning too conservative. For constrained
            # MBS variants, keep the full task vocabulary and use the calibrated
            # 0.5 fallback for none_is_max atoms, then clamp hard type/equality
            # invariants below.
            constraint_search_idx = tuple(range(len(bundle.feature_names)))
            constraint_omitted_idx: tuple[int, ...] = ()
        else:
            constraint_search_idx = search_idx
            constraint_omitted_idx = omitted_idx
        constraint_search_feature_names = project_feature_names(
            bundle.feature_names, constraint_search_idx
        )
        constraint_omitted_feature_names = project_feature_names(
            bundle.feature_names, constraint_omitted_idx
        )
        constraint_probs = tuple(bundle.probabilities[i] for i in constraint_search_idx)
        if sid.domain == "prb":
            constraint_probs = prb_domain_adjusted_probabilities(
                constraint_probs,
                constraint_search_feature_names,
            )
        reduced_constraint_groups = project_groups(
            bundle.constraint_groups, constraint_search_idx
        )
        selected_planning_constraint_groups = constraint_group_names(
            bundle.feature_names,
            bundle.selected_groups,
        )
        all_planning_constraint_groups = constraint_group_names(
            bundle.feature_names,
            bundle.constraint_groups,
        )
        # Exact-one semantics are sound only when every member of each extracted
        # invariant group is represented in the explicit PRaVaSS search space.
        exactly_one_constraints = (
            sid.domain in ("gw", "prb") and not constraint_omitted_idx
        )
        constraints_z_start = time.perf_counter()
        try:
            (
                reduced_constraints_z,
                constraints_z_info,
            ) = exact_constraints_runtime_normalizer(
                constraint_probs,
                reduced_constraint_groups,
                feature_names=constraint_search_feature_names,
                domain=sid.domain,
                exactly_one=exactly_one_constraints,
            )
            constraints_z_runtime = time.perf_counter() - constraints_z_start
            constraints_z_error = None
        except Exception as exc:
            reduced_constraints_z = None
            constraints_z_info = None
            constraints_z_runtime = time.perf_counter() - constraints_z_start
            constraints_z_error = f"{type(exc).__name__}: {exc}"
        selected_relaxed_z_start = time.perf_counter()
        if exactly_one_constraints:
            # Exactly-one only for selected groups that are whole constraint
            # groups; the GW selected groups are mostly strict subsets (an
            # object's tables without its gripper atom), which are at-most-one.
            if sid.domain == "prb":
                selected_relaxed_z = implied_selected_groups_normalizer(
                    constraint_probs,
                    project_groups(bundle.selected_groups, constraint_search_idx),
                    reduced_constraint_groups,
                )
            else:
                selected_relaxed_z = implied_selected_groups_normalizer(
                    bundle.probabilities,
                    bundle.selected_groups,
                    bundle.constraint_groups,
                )
            selected_relaxed_z_note = (
                "selected-groups relaxed normalizer (exactly-one for whole "
                "constraint groups, at-most-one for subsets)"
            )
        else:
            selected_relaxed_z = reduced_z
            selected_relaxed_z_note = None
        selected_relaxed_z_runtime = time.perf_counter() - selected_relaxed_z_start
    except Exception:
        # Match the previous runner behavior: if secondary constraint setup fails
        # after a bundle can be loaded, skip this calibration without polluting
        # results with a broad task-level error row.
        return records

    oracle_key = experiment_key(sid, calibration, "oracle", None, model=model)
    if oracle_key in existing_records:
        oracle_success = bool(existing_records[oracle_key].get("success"))
    else:
        oracle_result = run_oracle_planning(
            bundle.conditional_domain_path,
            bundle.problem_path,
            config.planning_timeout,
        )
        record_result(planning_record(bundle, "oracle", oracle_result, theta=None, belief_size=1))
        oracle_success = bool(oracle_result.success)

    (
        singleton_feature_names,
        singleton_probs,
        singleton_groups,
        singleton_z,
        singleton_omitted_feature_names,
    ) = singleton_baseline_inputs(bundle)

    valid_map_constrained_mass: float | None = None
    valid_map_unconstrained_mass: float | None = None

    # Check if we already have a valid_map record from a previous run.
    vm_existing_key = experiment_key(sid, calibration, "valid_map", None, model=model)
    if vm_existing_key in existing_records:
        vm_rec = existing_records[vm_existing_key]
        valid_map_constrained_mass = vm_rec.get("valid_map_constrained_mass")
        valid_map_unconstrained_mass = vm_rec.get("valid_map_unconstrained_mass")

    for method in SINGLETON_METHODS:
        key = experiment_key(sid, calibration, method, None, model=model)
        if key in existing:
            continue
        try:
            if method == "map":
                state_set = map_baseline(singleton_probs)
            elif method == "valid_map":
                state_set = valid_map(
                    singleton_probs,
                    singleton_groups,
                    singleton_z,
                    timeout_seconds=config.method_timeout,
                )
            else:
                continue
        except Exception as exc:
            record_result(
                make_error_record(
                    _base_error_context(sid, calibration, method, None, model=model), exc
                )
            )
            continue

        if method == "valid_map" and state_set.status == "ok":
            valid_map_constrained_mass = state_set.constrained_mass
            valid_map_unconstrained_mass = state_set.unconstrained_mass

        if state_set.status != "ok":
            record_result(
                unavailable_record(
                    bundle,
                    method,
                    theta=None,
                    belief_size=state_set_size(state_set),
                    oracle_success=oracle_success,
                    message="state_set_not_available",
                    status="error" if state_set.status == "error" else "unsupported",
                    search_feature_count=len(singleton_feature_names),
                    omitted_feature_count=len(singleton_omitted_feature_names),
                )
            )
            continue
        if state_set_size(state_set) > config.max_belief_states:
            record_result(
                unavailable_record(
                    bundle,
                    method,
                    theta=None,
                    belief_size=state_set_size(state_set),
                    oracle_success=oracle_success,
                    message="belief_size_cap",
                    status="unsupported",
                    search_feature_count=len(singleton_feature_names),
                    omitted_feature_count=len(singleton_omitted_feature_names),
                )
            )
            continue

        singleton_extra: dict[str, Any] = {}
        if method == "valid_map":
            singleton_extra["valid_map_constrained_mass"] = valid_map_constrained_mass
            singleton_extra["valid_map_unconstrained_mass"] = valid_map_unconstrained_mass

        result = run_singleton_classical_planning(
            bundle.conditional_domain_path,
            bundle.problem_path,
            singleton_feature_names,
            next(iter_state_set(state_set)),
            config.planning_timeout,
            oracle_success=oracle_success,
        )
        singleton_extra.update(
            _final_plan_set_mass_extra(
                plan_found=_plan_found(result),
                belief_size=state_set_size(state_set),
                unconstrained_mass=state_set.unconstrained_mass,
                constrained_mass=state_set.constrained_mass,
            )
        )
        record_result(
            planning_record(
                bundle,
                method,
                result,
                theta=None,
                belief_size=state_set_size(state_set),
                search_feature_count=len(singleton_feature_names),
                omitted_feature_count=len(singleton_omitted_feature_names),
                extra=singleton_extra if singleton_extra else None,
            )
        )

    prefix_search_common_extra: dict[str, Any] = {
        "selected_groups_normalizer": reduced_z,
        "selected_groups_normalizer_runtime_seconds": selected_z_runtime,
        "selected_groups_relaxed_normalizer": selected_relaxed_z,
        "selected_groups_relaxed_normalizer_runtime_seconds": selected_relaxed_z_runtime,
        "selected_groups_relaxed_normalizer_exactly_one": exactly_one_constraints,
        "selected_groups_relaxed_normalizer_note": selected_relaxed_z_note,
        "selected_group_count": len(reduced_groups),
        "constraint_group_count": len(reduced_constraint_groups),
        "constraint_search_feature_count": len(constraint_search_feature_names),
        "constraint_omitted_feature_count": len(constraint_omitted_feature_names),
        "constraints_normalizer": reduced_constraints_z,
        "constraints_normalizer_runtime_seconds": constraints_z_runtime,
        "constraints_normalizer_error": constraints_z_error,
        "constraints_normalizer_info": constraints_z_info,
        "constraints_exactly_one": exactly_one_constraints,
        "valid_map_constrained_mass": valid_map_constrained_mass,
        "valid_map_unconstrained_mass": valid_map_unconstrained_mass,
    }

    def record_prefix_search_method(
        method: str,
        probabilities: tuple[float, ...],
        groups: tuple[tuple[int, ...], ...],
        normalizer: float | None,
        feature_names: tuple[str, ...],
        omitted_names: tuple[str, ...],
        *,
        exactly_one: bool,
        method_extra: Mapping[str, Any],
        normalizer_error: str | None = None,
        normalizer_exact: bool = True,
        planning_constraint_groups: tuple[tuple[str, ...], ...] = (),
        planning_constraints_exactly_one: bool = False,
    ) -> None:
        key = experiment_key(
            sid,
            calibration,
            method,
            PREVAS_PREFIX_SEARCH_THETA,
            model=model,
        )
        if key in existing:
            return
        prefix_search_extra = prefix_search_common_extra | dict(method_extra)
        if planning_constraint_groups:
            prefix_search_extra.update(
                {
                    "planning_constraints_version": PLANNING_CONSTRAINTS_VERSION,
                    "planning_initial_constraint_group_count": len(planning_constraint_groups),
                    "planning_initial_constraints_exactly_one": planning_constraints_exactly_one,
                }
            )
        try:
            if normalizer is None:
                raise RuntimeError(normalizer_error or "normalizer unavailable")
            prefix_outcome = run_mbs_prevas_prefix_search_planning(
                bundle,
                probabilities,
                groups,
                normalizer,
                feature_names,
                omitted_names,
                method_timeout=config.method_timeout,
                planning_timeout=config.planning_timeout,
                oracle_success=oracle_success,
                exactly_one=exactly_one,
                normalizer_exact=normalizer_exact,
                planning_constraint_groups=planning_constraint_groups,
                planning_constraints_exactly_one=planning_constraints_exactly_one,
            )
            prefix_search_extra.update(prefix_outcome.extra)
            record_result(
                planning_record(
                    bundle,
                    method,
                    prefix_outcome.result,
                    theta=PREVAS_PREFIX_SEARCH_THETA,
                    belief_size=prefix_outcome.belief_size,
                    search_feature_count=len(feature_names),
                    omitted_feature_count=len(omitted_names),
                    extra=prefix_search_extra,
                )
            )
        except Exception as exc:
            record_result(
                make_error_record(
                    _base_error_context(
                        sid,
                        calibration,
                        method,
                        PREVAS_PREFIX_SEARCH_THETA,
                        model=model,
                    ),
                    exc,
                )
            )

    record_prefix_search_method(
        PREVAS_PREFIX_SEARCH_METHOD,
        reduced_probs,
        reduced_groups,
        reduced_z,
        search_feature_names,
        omitted_feature_names,
        exactly_one=False,
        method_extra={
            "validity_constraint_set": "selected_groups",
            "normalizer_constraint_set": "selected_groups",
            "normalizer": reduced_z,
        },
        planning_constraint_groups=selected_planning_constraint_groups,
    )

    record_prefix_search_method(
        PREVAS_PREFIX_SEARCH_CONSTRAINTS_EXACT_METHOD,
        constraint_probs,
        reduced_constraint_groups,
        reduced_constraints_z,
        constraint_search_feature_names,
        constraint_omitted_feature_names,
        exactly_one=exactly_one_constraints,
        normalizer_error=constraints_z_error,
        method_extra={
            "validity_constraint_set": "constraints",
            "normalizer_constraint_set": "constraints",
            "normalizer": reduced_constraints_z,
        },
        planning_constraint_groups=all_planning_constraint_groups,
        planning_constraints_exactly_one=exactly_one_constraints,
    )

    prefix_relaxed_normalizer = selected_relaxed_z
    prefix_relaxed_normalizer_constraint_set = "selected_groups"
    prefix_relaxed_normalizer_note = selected_relaxed_z_note
    prefix_relaxed_normalizer_error = None

    record_prefix_search_method(
        PREVAS_PREFIX_SEARCH_CONSTRAINTS_RELAXED_METHOD,
        constraint_probs,
        reduced_constraint_groups,
        prefix_relaxed_normalizer,
        constraint_search_feature_names,
        constraint_omitted_feature_names,
        exactly_one=exactly_one_constraints,
        normalizer_error=prefix_relaxed_normalizer_error,
        normalizer_exact=False,
        method_extra={
            "validity_constraint_set": "constraints",
            "normalizer_constraint_set": prefix_relaxed_normalizer_constraint_set,
            "normalizer": prefix_relaxed_normalizer,
            "normalizer_note": prefix_relaxed_normalizer_note,
        },
        planning_constraint_groups=all_planning_constraint_groups,
        planning_constraints_exactly_one=exactly_one_constraints,
    )

    adaptive_key = experiment_key(
        sid,
        calibration,
        ADAPTIVE_FLUENT_ABSTENTION_METHOD,
        None,
        model=model,
    )
    if adaptive_key not in existing:
        adaptive_extra = prefix_search_common_extra | {
            "validity_constraint_set": "none",
            "normalizer_constraint_set": "selected_groups",
            "normalizer": reduced_z,
        }
        try:
            adaptive_outcome = run_adaptive_fluent_abstention_planning(
                bundle,
                reduced_probs,
                reduced_groups,
                reduced_z,
                search_feature_names,
                omitted_feature_names,
                method_timeout=config.method_timeout,
                planning_timeout=config.planning_timeout,
                oracle_success=oracle_success,
                max_belief_states=config.max_belief_states,
            )
            adaptive_extra.update(adaptive_outcome.extra)
            record_result(
                planning_record(
                    bundle,
                    ADAPTIVE_FLUENT_ABSTENTION_METHOD,
                    adaptive_outcome.result,
                    theta=None,
                    belief_size=adaptive_outcome.belief_size,
                    search_feature_count=len(search_feature_names),
                    omitted_feature_count=len(omitted_feature_names),
                    extra=adaptive_extra,
                )
            )
        except Exception as exc:
            record_result(
                make_error_record(
                    _base_error_context(
                        sid,
                        calibration,
                        ADAPTIVE_FLUENT_ABSTENTION_METHOD,
                        None,
                        model=model,
                    ),
                    exc,
                )
            )

    # Keyed by *record* method name. defaultdict so any method name resolves.
    method_failed: dict[str, bool] = defaultdict(bool)

    def run_method(
        search_method: str,
        search_theta: float,
        *,
        record_method: str,
    ) -> None:
        """Run one belief-set method and append its (planning) record.

        ``search_method`` selects the validity set / normalizer / search and
        uses ``search_theta`` as the PRaVaSS mass target. ``record_method`` is
        what gets persisted; rows are keyed by θ.
        """
        record_theta = search_theta
        tau = tau_lookup.get((sid.domain, calibration, search_theta))
        key = experiment_key(
            sid, calibration, record_method, record_theta, model=model
        )
        if key in existing:
            record = existing_records[key]
            if (
                failure_pruning_enabled(record_method)
                and record.get("status") != "ok"
            ):
                method_failed[record_method] = True
            return

        if method_failed[record_method]:
            return

        method_extra: dict[str, Any] = {
            "selected_groups_normalizer": reduced_z,
            "selected_groups_normalizer_runtime_seconds": selected_z_runtime,
            "selected_groups_relaxed_normalizer": selected_relaxed_z,
            "selected_groups_relaxed_normalizer_runtime_seconds": selected_relaxed_z_runtime,
            "selected_groups_relaxed_normalizer_exactly_one": exactly_one_constraints,
            "selected_groups_relaxed_normalizer_note": selected_relaxed_z_note,
            "selected_group_count": len(reduced_groups),
            "constraint_group_count": len(reduced_constraint_groups),
            "constraint_search_feature_count": len(constraint_search_feature_names),
            "constraint_omitted_feature_count": len(constraint_omitted_feature_names),
            "constraints_normalizer": reduced_constraints_z,
            "constraints_normalizer_runtime_seconds": constraints_z_runtime,
            "constraints_normalizer_error": constraints_z_error,
            "constraints_normalizer_info": constraints_z_info,
            "constraints_exactly_one": exactly_one_constraints,
            "valid_map_constrained_mass": valid_map_constrained_mass,
            "valid_map_unconstrained_mass": valid_map_unconstrained_mass,
        }

        method_feature_names = search_feature_names
        method_omitted_feature_names = omitted_feature_names
        method_search_feature_count = len(search_feature_names)
        method_omitted_feature_count = len(omitted_feature_names)
        method_planning_constraint_groups: tuple[tuple[str, ...], ...] = ()
        method_planning_constraints_exactly_one = False

        try:
            if search_method == "mbs_prevas":
                method_extra.update(
                    {
                        "validity_constraint_set": "selected_groups",
                        "normalizer_constraint_set": "selected_groups",
                        "normalizer": reduced_z,
                    }
                )
                method_planning_constraint_groups = selected_planning_constraint_groups
                state_set = mbs_prevas(
                    reduced_probs,
                    reduced_groups,
                    search_theta,
                    reduced_z,
                    timeout_seconds=config.method_timeout,
                ).state_set
            elif search_method == "mbs_prevas_constraints_exact":
                if reduced_constraints_z is None:
                    raise RuntimeError(
                        constraints_z_error or "constraints normalizer unavailable"
                    )
                method_extra.update(
                    {
                        "validity_constraint_set": "constraints",
                        "normalizer_constraint_set": "constraints",
                        "normalizer": reduced_constraints_z,
                    }
                )
                method_feature_names = constraint_search_feature_names
                method_omitted_feature_names = constraint_omitted_feature_names
                method_search_feature_count = len(constraint_search_feature_names)
                method_omitted_feature_count = len(constraint_omitted_feature_names)
                method_planning_constraint_groups = all_planning_constraint_groups
                method_planning_constraints_exactly_one = exactly_one_constraints
                state_set = mbs_prevas(
                    constraint_probs,
                    reduced_constraint_groups,
                    search_theta,
                    reduced_constraints_z,
                    timeout_seconds=config.method_timeout,
                    exactly_one=exactly_one_constraints,
                ).state_set
            elif search_method == "mbs_prevas_constraints_relaxed":
                normalizer = selected_relaxed_z
                normalizer_constraint_set = "selected_groups"
                normalizer_note = selected_relaxed_z_note
                method_extra.update(
                    {
                        "validity_constraint_set": "constraints",
                        "normalizer_constraint_set": normalizer_constraint_set,
                        "normalizer": normalizer,
                        "normalizer_note": normalizer_note,
                    }
                )
                method_feature_names = constraint_search_feature_names
                method_omitted_feature_names = constraint_omitted_feature_names
                method_search_feature_count = len(constraint_search_feature_names)
                method_omitted_feature_count = len(constraint_omitted_feature_names)
                method_planning_constraint_groups = all_planning_constraint_groups
                method_planning_constraints_exactly_one = exactly_one_constraints
                state_set = mbs_prevas(
                    constraint_probs,
                    reduced_constraint_groups,
                    search_theta,
                    normalizer,
                    timeout_seconds=config.method_timeout,
                    exactly_one=exactly_one_constraints,
                    exact=False,
                ).state_set
            elif search_method == "unconstrained_top_mass":
                state_set = unconstrained_top_mass_baseline(
                    reduced_probs,
                    search_theta,
                    timeout_seconds=config.method_timeout,
                )
            elif search_method == "random_theta":
                sample_seed = stable_seed(
                    config.random_seed,
                    sid.domain,
                    sid.configuration,
                    sid.sample,
                    calibration,
                )
                method_planning_constraint_groups = selected_planning_constraint_groups
                state_set = random_theta_confident(
                    reduced_probs,
                    reduced_groups,
                    search_theta,
                    reduced_z,
                    seed=sample_seed,
                    timeout_seconds=config.method_timeout,
                )
            elif search_method == "fluent_abstention_unconstrained":
                method_extra["abstention_tau"] = tau
                state_set = fluent_abstention_unconstrained(
                    reduced_probs,
                    tau,
                    reduced_z,
                    selected_groups=reduced_groups,
                    timeout_seconds=config.method_timeout,
                )
            else:
                return
        except Exception as exc:
            record_result(
                make_error_record(
                    _base_error_context(
                        sid, calibration, record_method, record_theta,
                        model=model,
                    ),
                    exc,
                )
            )
            if failure_pruning_enabled(record_method):
                method_failed[record_method] = True
            return

        if state_set.status != "ok":
            record_result(
                unavailable_record(
                    bundle,
                    record_method,
                    theta=record_theta,
                    belief_size=state_set_size(state_set),
                    oracle_success=oracle_success,
                    message="state_set_not_available",
                    status="error" if state_set.status == "error" else "unsupported",
                    search_feature_count=method_search_feature_count,
                    omitted_feature_count=method_omitted_feature_count,
                    extra=method_extra,
                )
            )
            if failure_pruning_enabled(record_method):
                method_failed[record_method] = True
            return
        if state_set_size(state_set) > config.max_belief_states:
            record_result(
                unavailable_record(
                    bundle,
                    record_method,
                    theta=record_theta,
                    belief_size=state_set_size(state_set),
                    oracle_success=oracle_success,
                    message="belief_size_cap",
                    status="unsupported",
                    search_feature_count=method_search_feature_count,
                    omitted_feature_count=method_omitted_feature_count,
                    extra=method_extra,
                )
            )
            if failure_pruning_enabled(record_method):
                method_failed[record_method] = True
            return

        if method_planning_constraint_groups:
            method_extra.update(
                {
                    "planning_constraints_version": PLANNING_CONSTRAINTS_VERSION,
                    "planning_initial_constraint_group_count": len(
                        method_planning_constraint_groups
                    ),
                    "planning_initial_constraints_exactly_one": method_planning_constraints_exactly_one,
                }
            )

        result = run_belief_planning(
            bundle.conditional_domain_path,
            bundle.problem_path,
            method_feature_names,
            list(iter_state_set(state_set)),
            config.planning_timeout,
            oracle_success=oracle_success,
            unknown_feature_names=method_omitted_feature_names,
            initial_constraint_groups=method_planning_constraint_groups,
            initial_constraints_exactly_one=method_planning_constraints_exactly_one,
        )
        method_extra.update(
            _final_plan_set_mass_extra(
                plan_found=_plan_found(result),
                belief_size=state_set_size(state_set),
                unconstrained_mass=state_set.unconstrained_mass,
                constrained_mass=state_set.constrained_mass,
            )
        )

        if (
            failure_pruning_enabled(record_method)
            and not _plan_found(result)
        ):
            method_failed[record_method] = True

        record_result(
            planning_record(
                bundle,
                record_method,
                result,
                theta=record_theta,
                belief_size=state_set_size(state_set),
                search_feature_count=method_search_feature_count,
                omitted_feature_count=method_omitted_feature_count,
                extra=method_extra,
            )
        )

    for theta in THETA_GRID:
        tau = tau_lookup.get((sid.domain, calibration, theta))
        theta_methods = [
            "mbs_prevas",
            "mbs_prevas_constraints_exact",
            "mbs_prevas_constraints_relaxed",
            "unconstrained_top_mass",
            "random_theta",
        ]
        if tau is not None:
            theta_methods.append("fluent_abstention_unconstrained")
        for method in theta_methods:
            run_method(method, theta, record_method=method)

    return records


def _worker_count(requested_jobs: int | None) -> int:
    default_jobs = max(1, (os.cpu_count() or 1) - 1)
    if requested_jobs is None or requested_jobs <= 0:
        return default_jobs
    return requested_jobs


def main() -> None:
    args = parse_args()
    jobs = _worker_count(args.jobs)
    output_dir, results_path, agg_path = experiment_paths(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and results_path.exists():
        results_path.unlink()
    domains = set(args.domains)
    reset_methods = set(args.reset_methods)
    if reset_methods and results_path.exists() and not args.overwrite:
        drop_method = lambda r: r.get("method") in reset_methods and r.get("domain") in domains
        kept = [
            r
            for r in iter_jsonl(results_path)
            if not drop_method(r)
        ]
        write_jsonl_atomic(results_path, kept)
        print(f"Reset {sorted(reset_methods)} for domains {sorted(domains)}: kept {len(kept)} rows in {results_path}")
    configs = set(args.configs) if args.configs else None
    requested_calibrations = set(args.calibrations)
    calibration_list = sorted(requested_calibrations)
    tau_lookup = load_selected_tau_lookup(
        args.model,
        calibrations=calibration_list,
        domains=sorted(domains),
        output_root=args.abstention_root,
    )
    if not tau_lookup:
        sys.exit(
            f"No τ calibration found under {args.abstention_root}/"
            f"{args.model} for calibrations={calibration_list} "
            f"domains={sorted(domains)}. Run `python calibrate_tau.py "
            f"--model {args.model}` before this experiment."
        )
    existing_records = load_latest_records(results_path)
    existing = set(existing_records)
    stale_singleton_keys = stale_singleton_baseline_keys(existing_records)
    stale_tau_keys = stale_fluent_abstention_tau_keys(
        existing_records,
        tau_lookup,
        domains=domains,
        calibrations=requested_calibrations,
    )
    stale_keys = stale_singleton_keys | stale_tau_keys
    for key in stale_keys:
        existing.discard(key)
        existing_records.pop(key, None)
    if stale_singleton_keys:
        print(
            "Recomputing "
            f"{len(stale_singleton_keys)} stale MAP/Valid-MAP records that omitted features"
        )
    if stale_tau_keys:
        print(
            "Recomputing "
            f"{len(stale_tau_keys)} stale fluent-abstention records whose tau "
            "does not match the current calibration"
        )

    def record_result(record: Mapping[str, Any]) -> None:
        payload = dict(record)
        key = result_key(payload, RESULT_KEY_FIELDS)
        if key in existing:
            return
        append_jsonl(results_path, payload)
        existing_records[key] = payload
        existing.add(key)

    sample_ids = filter_sample_ids(
        load_test_sample_ids(),
        domains=domains,
        configs=configs,
        limit=args.limit,
    )
    config = Experiment3Config(
        planning_timeout=args.planning_timeout,
        method_timeout=args.method_timeout,
        max_belief_states=args.max_belief_states,
        random_seed=args.random_seed,
    )
    tasks: list[Experiment3Task] = []

    for sid in tqdm(sample_ids, desc="Exp3 inputs"):
        prediction_key = experiment_key(sid, None, "prediction", None, model=args.model)
        try:
            # Touch the prediction so a missing one is a prediction error
            # rather than a later failure inside the worker.
            load_json(prediction_path(sid, TEST_SPLIT, args.model))
            calibrations = sorted(
                set(available_calibrations(args.model, sid.domain))
                & requested_calibrations
            )
        except Exception as exc:
            if prediction_key not in existing:
                record_result(
                    make_error_record(
                        _base_error_context(sid, None, "prediction", None, model=args.model), exc
                    )
                )
            continue

        for calibration in calibrations:
            tasks.append(
                Experiment3Task(
                    sid=sid,
                    calibration=calibration,
                    config=config,
                    model=args.model,
                    tau_lookup=tau_lookup,
                    existing_records=_existing_records_for_task(
                        existing_records, sid, calibration
                    ),
                )
            )

    if tasks:
        effective_jobs = min(jobs, len(tasks))
        if effective_jobs > 1:
            print(
                f"Running {len(tasks)} sample/calibration tasks with "
                f"{effective_jobs} worker processes"
            )
            # ProcessPoolExecutor uses non-daemon workers, so each calibration
            # task can still call run_belief_planning(), which starts its own
            # timeout-enforced subprocess.
            ctx = mp.get_context("spawn")
            with futures.ProcessPoolExecutor(
                max_workers=effective_jobs,
                mp_context=ctx,
            ) as executor:
                future_to_task = {
                    executor.submit(_run_experiment_3_calibration_task, task): task
                    for task in tasks
                }
                iterator = futures.as_completed(future_to_task)
                for future in tqdm(
                    iterator,
                    total=len(future_to_task),
                    desc="Exp3 calibrations",
                ):
                    for record in future.result():
                        record_result(record)
        else:
            for task in tqdm(tasks, desc="Exp3 calibrations"):
                for record in _run_experiment_3_calibration_task(task):
                    record_result(record)

    aggregate = aggregate_experiment_3(existing_records.values())
    aggregate["calibration_sources"] = {
        "model": args.model,
        "abstention_root": str(args.abstention_root),
        "tau_entries": len(tau_lookup),
    }
    write_json_atomic(agg_path, aggregate)
    print(f"Wrote {results_path}")
    print(f"Wrote {agg_path}")


if __name__ == "__main__":
    main()
