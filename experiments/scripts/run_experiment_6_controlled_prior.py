#!/usr/bin/env python3
"""Experiment 6: controlled misleading priors, observation, and replanning."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from mbs_experiments.closed_loop import (
    PROB_EPS,
    build_common_state_ordering,
    constraint_valid_map,
    fixed_mass_prefix,
    paired_bootstrap_mean_ci,
    planning_result_dict,
    search_ams_prefixes,
)
from mbs_experiments.data import (
    CalibrationBundle,
    SampleId,
    load_calibration_bundle,
    load_test_sample_ids,
    model_to_path_segment,
    prediction_path,
)
from mbs_experiments.experiment6 import (
    PRIMARY_RELEVANCE_VERSION,
    PROTOCOL_VERSION,
    QUERY_POLICY,
    REVISED_RELEVANCE_VERSION,
    CandidateUnit,
    FrozenTreatment,
    controlled_probabilities,
    controlled_universe_size,
    enumerate_controlled_ordering,
    extract_candidate_units,
    freeze_treatment,
    orient_treatment,
    project_constraint_groups,
    search_adaptive_abstention,
    uniform_hidden_probabilities,
    update_group_observation,
)
from mbs_experiments.io import (
    append_jsonl,
    iter_jsonl,
    load_json,
    result_key,
    write_json_atomic,
    write_jsonl_atomic,
)
from mbs_experiments.normalizers import exact_constraints_runtime_normalizer
from mbs_experiments.pddl_truth import pddl_ground_truth
from mbs_experiments.planning import run_belief_planning, run_oracle_planning
from mbs_experiments.states import State, is_valid_state

DEFAULT_MODEL = "OpenAI/gpt-4.1-2025-04-14"
DEFAULT_OUTPUT_ROOT = Path("experiments_data/experiment_6_controlled_prior")
CONSTRAINT_MODE = "full_constraints_plus_controlled_exactly_one"
NORMALIZER_MODE = "exact_full_enumeration"
GROUND_TRUTH_SOURCE = "problem_pddl_init_closed_world"
FIXED_THETAS = (0.5, 0.8, 0.9, 0.95)
METHODS = ("map", "ams", "adaptive_abstention")
PRIOR_MODES = ("controlled_sensor", "uniform_hidden")
ABSTENTION_SEARCH_MODE = "adaptive-count-map-fallback-v1"

RESULT_KEY_FIELDS = (
    "protocol_version",
    "model",
    "domain",
    "configuration",
    "sample",
    "calibration",
    "method",
    "round",
    "prior_mode",
    "corruption_count",
    "sensor_confidence",
    "corruption_seed",
    "corruption_mask_fingerprint",
    "query_seed",
    "query_policy",
    "queried_unit",
    "query_hit_corruption",
    "constraint_mode",
    "normalizer_mode",
    "abstention_threshold",
    "abstention_calibration_id",
    "abstention_search_mode",
    "max_explicit_states",
    "planning_timeout",
    "method_timeout",
    "selected_sample_fingerprint",
    "relevance_version",
    "ground_truth_source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--calibrations", nargs="+", default=("global",))
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--exclude-samples-from",
        type=Path,
        help=(
            "Optional newline-delimited sample list to exclude before sampling; "
            "use this to keep a scaled confirmation disjoint from a pilot."
        ),
    )
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--query-seed", type=int, default=0)
    parser.add_argument("--corruption-counts", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--prior-mode",
        choices=PRIOR_MODES,
        default="controlled_sensor",
        help=(
            "controlled_sensor uses the preregistered aligned/decoy treatment; "
            "uniform_hidden assigns p=0.5 to every declared hidden relevant unit."
        ),
    )
    parser.add_argument("--sensor-confidence", type=float, default=0.6)
    parser.add_argument(
        "--abstention-threshold",
        type=float,
        default=0.61,
        help=(
            "Legacy result-key field retained for frozen-run compatibility; "
            "adaptive-count abstention does not use a probability threshold."
        ),
    )
    parser.add_argument(
        "--abstention-calibration-id",
        default=ABSTENTION_SEARCH_MODE,
    )
    parser.add_argument("--max-units", type=int, default=6)
    parser.add_argument("--max-explicit-states", type=int, default=64)
    parser.add_argument("--planning-timeout", type=float, default=30.0)
    parser.add_argument("--method-timeout", type=float, default=30.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--phase-a-only", action="store_true")
    parser.add_argument(
        "--relevance-version",
        choices=("auto", PRIMARY_RELEVANCE_VERSION, REVISED_RELEVANCE_VERSION),
        default="auto",
    )
    parser.add_argument(
        "--run-name",
        default="pilot",
        help="Separate immutable audit/pilot/main population namespace.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--retry-status",
        nargs="+",
        default=("error", "timeout", "memory_limit"),
    )
    return parser.parse_args()


@dataclass(frozen=True)
class Experiment6Config:
    prior_mode: str
    sensor_confidence: float
    corruption_counts: tuple[int, ...]
    corruption_seed: int
    query_seed: int
    abstention_threshold: float
    abstention_calibration_id: str
    max_explicit_states: int
    planning_timeout: float
    method_timeout: float
    relevance_version: str


@dataclass(frozen=True)
class Experiment6Task:
    sid: SampleId
    calibration: str
    model: str
    units: tuple[CandidateUnit, ...]
    treatments: tuple[FrozenTreatment, ...]
    selected_sample_fingerprint: str
    config: Experiment6Config


def _sample_line(sid: SampleId) -> str:
    return sid.split_relative_path


def _fingerprint_lines(lines: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{line}\n" for line in lines).encode()
    ).hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_paths(
    output_root: Path,
    model: str,
    run_name: str,
) -> tuple[Path, Path, Path]:
    output_dir = output_root / model_to_path_segment(model) / run_name
    return (
        output_dir,
        output_dir / "sample_results.jsonl",
        output_dir / "aggregate_metrics.json",
    )


def _scan_version(
    sample_ids: Sequence[SampleId],
    *,
    model: str,
    sample_seed: int,
    relevance_version: str,
    max_units: int,
    max_states: int,
) -> tuple[list[tuple[SampleId, tuple[CandidateUnit, ...]]], dict[str, Any]]:
    eligible: list[tuple[SampleId, tuple[CandidateUnit, ...]]] = []
    unit_counts: list[int] = []
    universe_counts: list[int] = []
    composition: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    for sid in tqdm(sample_ids, desc=f"Exp6 Phase A ({relevance_version})"):
        if sid.domain != "alf":
            continue
        if not prediction_path(sid, "test", model).exists():
            exclusions["missing_prediction"] += 1
            continue
        try:
            text = (Path("split_data/test") / sid.split_relative_path / "problem.pddl").read_text()
            units = extract_candidate_units(
                text,
                sample_id=sid.split_relative_path,
                sample_seed=sample_seed,
                relevance_version=relevance_version,
                max_units=max_units,
                max_states=max_states,
            )
        except Exception:  # noqa: BLE001 - malformed samples are exclusions
            exclusions["candidate_extraction_failure"] += 1
            continue
        unit_counts.append(len(units))
        if not 2 <= len(units) <= max_units:
            exclusions["outside_2_6_unit_rule"] += 1
            continue
        universe = controlled_universe_size(units)
        universe_counts.append(universe)
        if universe > max_states:
            exclusions["state_capacity"] += 1
            continue
        for unit in units:
            composition[unit.kind] += 1
            composition[unit.unit_id.split("(", 1)[0]] += 1
        eligible.append((sid, units))
    summary = {
        "relevance_version": relevance_version,
        "scanned_alf_samples": sum(sid.domain == "alf" for sid in sample_ids),
        "eligible_samples": len(eligible),
        "exclusions": dict(exclusions),
        "unit_composition": dict(composition),
        "unit_count": _distribution(unit_counts),
        "eligible_universe_size": _distribution(universe_counts),
    }
    return eligible, summary


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
        "mean": mean(ordered),
    }


def phase_a_scan(args: argparse.Namespace, output_dir: Path):
    all_ids = [sid for sid in load_test_sample_ids() if sid.domain == "alf"]
    strict_eligible, strict_summary = _scan_version(
        all_ids,
        model=args.model,
        sample_seed=args.sample_seed,
        relevance_version=PRIMARY_RELEVANCE_VERSION,
        max_units=args.max_units,
        max_states=args.max_explicit_states,
    )
    scans = [strict_summary]
    if args.relevance_version == PRIMARY_RELEVANCE_VERSION:
        chosen = strict_eligible
        chosen_version = PRIMARY_RELEVANCE_VERSION
    elif (
        args.relevance_version == REVISED_RELEVANCE_VERSION
        or len(strict_eligible) < 100
    ):
        chosen, revision = _scan_version(
            all_ids,
            model=args.model,
            sample_seed=args.sample_seed,
            relevance_version=REVISED_RELEVANCE_VERSION,
            max_units=args.max_units,
            max_states=args.max_explicit_states,
        )
        scans.append(revision)
        chosen_version = REVISED_RELEVANCE_VERSION
    else:
        chosen = strict_eligible
        chosen_version = PRIMARY_RELEVANCE_VERSION
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "label_discipline": (
            "candidate extraction reads task schema, object/type/cancontain/openable "
            "facts, goals, and symbolic visibility only; no held-out fluent values"
        ),
        "revision_rule": (
            "use typed-action-support-v2 iff strict-goal-hidden-v2 has fewer "
            "than 100 eligible samples"
        ),
        "scans": scans,
        "chosen_relevance_version": chosen_version,
        "chosen_eligible_samples": len(chosen),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "phase_a_feasibility.json", report)
    return chosen, chosen_version, report


def prepare_population(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[
    list[tuple[SampleId, tuple[CandidateUnit, ...]]],
    str,
    str,
    dict[str, Any],
]:
    eligible, relevance_version, scan_report = phase_a_scan(args, output_dir)
    eligible.sort(key=lambda item: _sample_line(item[0]))
    rng = random.Random(args.sample_seed)
    rng.shuffle(eligible)
    excluded_lines: set[str] = set()
    excluded_source_sha256 = None
    if args.exclude_samples_from is not None:
        excluded_lines = {
            line.strip()
            for line in args.exclude_samples_from.read_text().splitlines()
            if line.strip()
        }
        excluded_source_sha256 = _file_sha256(args.exclude_samples_from)
        eligible = [
            item
            for item in eligible
            if _sample_line(item[0]) not in excluded_lines
        ]
    selected = eligible[: args.limit] if args.limit is not None else eligible
    if not selected:
        raise RuntimeError("no eligible Experiment-6 samples")
    selected_lines = [_sample_line(sid) for sid, _units in selected]
    fingerprint = _fingerprint_lines(selected_lines)

    eligible_path = output_dir / "eligible_samples.txt"
    selected_path = output_dir / "selected_samples.txt"
    candidate_path = output_dir / "candidate_units.jsonl"
    treatment_path = output_dir / "treatment_assignments.jsonl"
    eligible_path.write_text(
        "".join(f"{_sample_line(sid)}\n" for sid, _units in eligible)
    )
    selected_path.write_text("".join(f"{line}\n" for line in selected_lines))
    candidate_records = [
        {
            "protocol_version": PROTOCOL_VERSION,
            "sample": _sample_line(sid),
            "relevance_version": relevance_version,
            "candidate_units": [unit.as_json() for unit in units],
            "candidate_unit_count": len(units),
            "controlled_universe_size": controlled_universe_size(units),
        }
        for sid, units in selected
    ]
    write_jsonl_atomic(candidate_path, candidate_records)
    treatment_records: list[dict[str, Any]] = []
    for sid, units in selected:
        for corruption_count in args.corruption_counts:
            frozen = freeze_treatment(
                units,
                sample_id=_sample_line(sid),
                corruption_count=corruption_count,
                corruption_seed=args.corruption_seed,
                query_seed=args.query_seed,
            )
            treatment_records.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "sample": _sample_line(sid),
                    "corruption_seed": args.corruption_seed,
                    "query_seed": args.query_seed,
                    "query_policy": QUERY_POLICY,
                }
                | frozen.as_json()
            )
    write_jsonl_atomic(treatment_path, treatment_records)
    hashes = {
        path.name: _file_sha256(path)
        for path in (eligible_path, selected_path, candidate_path, treatment_path)
    }
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "model": args.model,
        "domain": "alf",
        "calibrations": list(args.calibrations),
        "rounds": [0, 1],
        "methods": list(METHODS) + ["oracle"],
        "prior_mode": args.prior_mode,
        "corruption_counts": list(args.corruption_counts),
        "sensor_confidence": args.sensor_confidence,
        "sample_seed": args.sample_seed,
        "excluded_sample_count": len(excluded_lines),
        "excluded_samples_source": (
            str(args.exclude_samples_from)
            if args.exclude_samples_from is not None
            else None
        ),
        "excluded_samples_source_sha256": excluded_source_sha256,
        "corruption_seed": args.corruption_seed,
        "query_seed": args.query_seed,
        "query_policy": QUERY_POLICY,
        "selected_sample_count": len(selected),
        "selected_sample_fingerprint": fingerprint,
        "relevance_version": relevance_version,
        "max_units": args.max_units,
        "max_explicit_states": args.max_explicit_states,
        "constraint_mode": CONSTRAINT_MODE,
        "normalizer_mode": NORMALIZER_MODE,
        "ground_truth_source": GROUND_TRUTH_SOURCE,
        "abstention_threshold": args.abstention_threshold,
        "abstention_calibration_id": args.abstention_calibration_id,
        "abstention_search_mode": ABSTENTION_SEARCH_MODE,
        "planning_timeout": args.planning_timeout,
        "method_timeout": args.method_timeout,
        "observation_eps": PROB_EPS,
        "frozen_file_hashes": hashes,
        "phase_a": scan_report,
        "synthetic_stress_test_qualification": (
            "Uniform p=0.5 is assigned only to declared non-visible, "
            "goal/action-relevant uncertainty units; other coordinates remain "
            "clamped to the common natural valid MAP."
            if args.prior_mode == "uniform_hidden"
            else (
                "Held-out labels orient frozen priors; results do not estimate "
                "natural perception error or calibration."
            )
        ),
    }
    config_path = output_dir / "protocol_config.json"
    if args.freeze_protocol and config_path.exists():
        existing = load_json(config_path)
        if existing != metadata:
            raise RuntimeError(
                f"refusing incompatible frozen protocol resume: {config_path}"
            )
    write_json_atomic(
        config_path if args.freeze_protocol else output_dir / "run_config.json",
        metadata,
    )
    return selected, fingerprint, relevance_version, metadata


def _semantic_context(
    task: Experiment6Task,
    treatment: FrozenTreatment,
    *,
    method: str,
    round_index: int | None,
) -> dict[str, Any]:
    sid = task.sid
    config = task.config
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model": task.model,
        "domain": sid.domain,
        "configuration": sid.configuration,
        "sample": sid.sample,
        "sample_path": sid.split_relative_path,
        "calibration": task.calibration,
        "method": method,
        "round": round_index,
        "prior_mode": config.prior_mode,
        "corruption_count": treatment.corruption_count,
        "sensor_confidence": config.sensor_confidence,
        "corruption_seed": config.corruption_seed,
        "corruption_mask_fingerprint": treatment.corruption_mask_fingerprint,
        "query_seed": config.query_seed,
        "query_policy": QUERY_POLICY,
        "queried_unit": treatment.queried_unit_id,
        "query_hit_corruption": (
            treatment.queried_unit_id in treatment.corrupted_unit_ids
        ),
        "constraint_mode": CONSTRAINT_MODE,
        "normalizer_mode": NORMALIZER_MODE,
        "abstention_threshold": config.abstention_threshold,
        "abstention_calibration_id": config.abstention_calibration_id,
        "abstention_search_mode": ABSTENTION_SEARCH_MODE,
        "max_explicit_states": config.max_explicit_states,
        "planning_timeout": config.planning_timeout,
        "method_timeout": config.method_timeout,
        "selected_sample_fingerprint": task.selected_sample_fingerprint,
        "relevance_version": config.relevance_version,
        "ground_truth_source": GROUND_TRUTH_SOURCE,
    }


def _common_exclusion(
    task: Experiment6Task,
    treatment: FrozenTreatment,
    reason: str,
    error: BaseException | None = None,
) -> dict[str, Any]:
    record = _semantic_context(
        task, treatment, method="common_exclusion", round_index=None
    )
    record.update(
        {
            "status": "common_exclusion",
            "exclusion_reason": reason,
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
        }
    )
    return record


def _expanded_vocabulary(
    bundle: CalibrationBundle,
    units: Sequence[CandidateUnit],
) -> tuple[tuple[str, ...], State, tuple[tuple[int, ...], ...], dict[str, Any]]:
    natural_normalizer, _info = exact_constraints_runtime_normalizer(
        bundle.probabilities,
        bundle.constraint_groups,
        feature_names=bundle.feature_names,
        domain=bundle.sample_id.domain,
        exactly_one=False,
    )
    natural_ordering = build_common_state_ordering(
        bundle.probabilities,
        bundle.constraint_groups,
        natural_normalizer,
        max_belief_states=1,
        target_mass=0.0,
        timeout_seconds=30.0,
    )
    natural_map = constraint_valid_map(natural_ordering)
    feature_names = list(bundle.feature_names)
    base_state = list(natural_map)
    for unit in units:
        for atom in unit.atoms:
            if atom not in feature_names:
                feature_names.append(atom)
                base_state.append(False)
    constraints_payload = load_json(bundle.constraints_path)
    constraint_groups = project_constraint_groups(
        constraints_payload.get("constraints", ()),
        feature_names,
    )
    return (
        tuple(feature_names),
        tuple(base_state),
        constraint_groups,
        {
            "natural_base_map": list(natural_map),
            "natural_normalizer": natural_normalizer,
        },
    )


def _planning_call(
    task: Experiment6Task,
    bundle: CalibrationBundle,
    feature_names: Sequence[str],
    constraint_groups: Sequence[Sequence[int]],
    states: Sequence[State],
    *,
    oracle_success: bool,
):
    constraint_names = tuple(
        tuple(feature_names[index] for index in group)
        for group in constraint_groups
    )
    exactly_one_names = tuple(
        unit.atoms for unit in task.units if unit.kind == "categorical"
    )
    return run_belief_planning(
        bundle.conditional_domain_path,
        bundle.problem_path,
        feature_names,
        states,
        task.config.planning_timeout,
        oracle_success=oracle_success,
        initial_constraint_groups=constraint_names,
        initial_constraints_exactly_one=False,
        exactly_one_initial_constraint_groups=exactly_one_names,
    )


def _state_mass(ordering, size: int) -> float:
    return (
        float(ordering.cumulative_constrained_masses[size - 1])
        if size > 0
        else 0.0
    )


def _round_common(
    task: Experiment6Task,
    treatment: FrozenTreatment,
    oriented,
    ordering,
    feature_names: Sequence[str],
    probabilities: Sequence[float],
    ground_truth_state: State,
    natural_info: Mapping[str, Any],
    *,
    round_index: int,
) -> dict[str, Any]:
    fixed_metrics = {}
    for theta in FIXED_THETAS:
        prefix = fixed_mass_prefix(ordering, theta)
        fixed_metrics[f"{theta:g}"] = {
            "size": prefix.size,
            "constrained_mass": prefix.constrained_mass,
            "true_state_in_set": ground_truth_state in prefix.states,
            "status": prefix.status,
        }
    return {
        "candidate_units": [unit.unit_id for unit in task.units],
        "candidate_unit_count": len(task.units),
        "corrupted_units": list(treatment.corrupted_unit_ids),
        "remaining_corrupted_units": (
            treatment.corruption_count
            if round_index == 0
            else oriented.remaining_corrupted_units
        ),
        "prior_source": task.config.prior_mode,
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "controlled_probabilities": list(probabilities),
        "ordered_state_count": len(ordering.states),
        "ordered_states": [list(state) for state in ordering.states],
        "normalizer": ordering.normalizer,
        "normalizer_method": NORMALIZER_MODE,
        "ordering_runtime_seconds": ordering.runtime_seconds,
        "ordering_invalid_states_skipped": ordering.invalid_states_skipped,
        "map_state": list(ordering.states[0]),
        "map_correct": ordering.states[0] == ground_truth_state,
        "true_state_in_ordered_set": ground_truth_state in ordering.states,
        "fixed_theta_metrics": fixed_metrics,
        "ground_truth_assignments": dict(oriented.ground_truth_assignments),
        "decoy_assignments": dict(oriented.decoy_assignments),
        "sensor_state_by_unit": dict(oriented.sensor_state_by_unit),
    } | dict(natural_info)


def run_experiment_6_task(task: Experiment6Task) -> list[dict[str, Any]]:
    try:
        bundle = load_calibration_bundle(
            task.sid,
            task.calibration,
            model=task.model,
            split="test",
        )
        feature_names, base_state, constraint_groups, natural_info = (
            _expanded_vocabulary(bundle, task.units)
        )
    except Exception as exc:  # noqa: BLE001 - task-level exclusion boundary
        return [
            _common_exclusion(task, treatment, "bundle_or_base_failure", exc)
            for treatment in task.treatments
        ]

    oracle = run_oracle_planning(
        bundle.conditional_domain_path,
        bundle.problem_path,
        task.config.planning_timeout,
    )
    problem_text = Path(bundle.problem_path).read_text()
    gt_map = pddl_ground_truth(feature_names, problem_text)
    ground_truth_state = tuple(bool(gt_map.get(atom, False)) for atom in feature_names)
    natural_info["natural_base_map_correct"] = tuple(
        natural_info["natural_base_map"]
    ) == ground_truth_state[: len(bundle.feature_names)]
    records: list[dict[str, Any]] = []
    # Paired conditions frequently induce identical complete belief-state
    # tuples (especially after the shared observation). Planning is a pure
    # function of this tuple and the frozen task/config, so reuse the exact
    # result object across conditions as well as MAP/AMS within a round.
    planning_cache: dict[tuple[State, ...], Any] = {}

    def plan_cached(
        states: Sequence[State],
        trace: list[dict[str, Any]] | None = None,
    ):
        key = tuple(tuple(state) for state in states)
        hit = key in planning_cache
        if not hit:
            planning_cache[key] = _planning_call(
                task,
                bundle,
                feature_names,
                constraint_groups,
                key,
                oracle_success=oracle.success,
            )
        if trace is not None:
            trace.append(
                {
                    "belief_size": len(key),
                    "planning_cache_hit": hit,
                }
            )
        return planning_cache[key]

    for treatment in task.treatments:
        try:
            oriented = orient_treatment(
                feature_names,
                task.units,
                treatment,
                gt_map,
                sensor_confidence=task.config.sensor_confidence,
                sample_id=task.sid.split_relative_path,
                corruption_seed=task.config.corruption_seed,
            )
            if task.config.prior_mode == "uniform_hidden":
                probabilities0 = uniform_hidden_probabilities(
                    feature_names,
                    base_state,
                    task.units,
                )
            else:
                probabilities0 = controlled_probabilities(
                    feature_names,
                    base_state,
                    task.units,
                    oriented,
                )
            probabilities1 = update_group_observation(
                probabilities0,
                feature_names,
                task.units,
                treatment.queried_unit_id,
                gt_map,
            )
            orderings = tuple(
                enumerate_controlled_ordering(
                    probabilities,
                    feature_names,
                    base_state,
                    task.units,
                    constraint_groups,
                    max_states=task.config.max_explicit_states,
                )
                for probabilities in (probabilities0, probabilities1)
            )
            # The leading state is the frozen joint sensor/decoy assignment.
            if not is_valid_state(orderings[0].states[0], constraint_groups):
                raise ValueError("joint decoy MAP violates common constraints")
        except Exception as exc:  # noqa: BLE001 - task-level exclusion boundary
            records.append(
                _common_exclusion(
                    task,
                    treatment,
                    "controlled_belief_construction_failure",
                    exc,
                )
            )
            continue

        for round_index, (probabilities, ordering) in enumerate(
            zip((probabilities0, probabilities1), orderings)
        ):
            common = _round_common(
                task,
                treatment,
                oriented,
                ordering,
                feature_names,
                probabilities,
                ground_truth_state,
                natural_info,
                round_index=round_index,
            )
            belief_record = _semantic_context(
                task, treatment, method="belief", round_index=round_index
            )
            belief_record.update(common)
            belief_record["status"] = "ok"
            records.append(belief_record)

            map_trace: list[dict[str, Any]] = []
            singleton = plan_cached(ordering.states[:1], map_trace)
            map_record = _semantic_context(
                task, treatment, method="map", round_index=round_index
            )
            map_record.update(common)
            map_record.update(
                {
                    "belief_size": 1,
                    "confidence": _state_mass(ordering, 1),
                    "true_state_in_set": ground_truth_state == ordering.states[0],
                    "planning_cache_hit": map_trace[0]["planning_cache_hit"],
                }
            )
            map_record.update(planning_result_dict(singleton))
            records.append(map_record)

            ams_trace: list[dict[str, Any]] = []
            ams = search_ams_prefixes(
                ordering.states,
                lambda states, trace=ams_trace: plan_cached(states, trace),
                singleton_result=singleton,
            )
            ams_record = _semantic_context(
                task, treatment, method="ams", round_index=round_index
            )
            ams_record.update(common)
            ams_record.update(
                {
                    "belief_size": ams.prefix_size,
                    "confidence": _state_mass(ordering, ams.prefix_size),
                    "true_state_in_set": (
                        ground_truth_state in ordering.states[: ams.prefix_size]
                    ),
                    "attempts": [
                        {"prefix_size": size} | planning_result_dict(result)
                        for size, result in ams.attempts
                    ],
                    "map_singleton_reused": ams.attempts[0][1] is singleton,
                    "planning_cache_trace": [
                        {
                            "prefix_size": item["belief_size"],
                            "planning_cache_hit": item["planning_cache_hit"],
                        }
                        for item in ams_trace
                    ],
                    "ams_search_runtime_seconds": sum(
                        result.runtime_seconds for _size, result in ams.attempts
                    ),
                }
            )
            ams_record.update(planning_result_dict(ams.planning_result))
            records.append(ams_record)

            abst_record = _semantic_context(
                task,
                treatment,
                method="adaptive_abstention",
                round_index=round_index,
            )
            abst_record.update(common)
            try:
                abst_trace: list[dict[str, Any]] = []
                adaptive = search_adaptive_abstention(
                    ordering,
                    probabilities,
                    feature_names,
                    task.units,
                    lambda states, trace=abst_trace: plan_cached(states, trace),
                    singleton_result=singleton,
                    max_states=task.config.max_explicit_states,
                )
                selected = adaptive.selected
                completions = selected.completions
                abstained_ids = selected.abstained_unit_ids
                logical_count = selected.logical_completion_count
                abst_result = selected.planning_result
                attempts = [
                    {
                        "abstained_count": attempt.abstained_count,
                        "abstained_unit_ids": list(
                            attempt.abstained_unit_ids
                        ),
                        "belief_size": len(attempt.completions),
                        "logical_completion_count": (
                            attempt.logical_completion_count
                        ),
                    }
                    | planning_result_dict(attempt.planning_result)
                    for attempt in adaptive.attempts
                ]
                abst_record.update(
                    {
                        "belief_size": len(completions),
                        "logical_completion_count": logical_count,
                        "materialized_planning_count": len(completions),
                        "abstained_count": selected.abstained_count,
                        "abstained_unit_ids": list(abstained_ids),
                        "confidence": sum(
                            ordering.state_unconstrained_masses[
                                ordering.states.index(state)
                            ]
                            for state in completions
                        )
                        / ordering.normalizer,
                        "true_state_in_set": ground_truth_state in completions,
                        "attempts": attempts,
                        "map_singleton_reused": (
                            adaptive.attempts[0].planning_result is singleton
                        ),
                        "planning_cache_trace": [
                            {
                                "belief_size": item["belief_size"],
                                "planning_cache_hit": item[
                                    "planning_cache_hit"
                                ],
                            }
                            for item in abst_trace
                        ],
                        "abstention_search_runtime_seconds": sum(
                            attempt.planning_result.runtime_seconds
                            for attempt in adaptive.attempts
                        ),
                    }
                )
                abst_record.update(planning_result_dict(abst_result))
            except Exception as exc:  # noqa: BLE001 - method failure is recorded
                abst_record.update(
                    {
                        "status": "unsupported",
                        "plan_found": False,
                        "success": False,
                        "runtime_seconds": 0.0,
                        "error_message": f"{type(exc).__name__}: {exc}",
                    }
                )
            records.append(abst_record)

            oracle_record = _semantic_context(
                task, treatment, method="oracle", round_index=round_index
            )
            oracle_record.update(common)
            oracle_record.update(
                {
                    "belief_size": 1,
                    "confidence": 1.0,
                    "true_state_in_set": True,
                }
            )
            oracle_record.update(planning_result_dict(oracle))
            records.append(oracle_record)
    return records


def aggregate_experiment_6(
    records: Iterable[Mapping[str, Any]],
    *,
    fingerprint: str,
    bootstrap_seed: int,
    planning_timeout: float,
) -> dict[str, Any]:
    records = tuple(dict(row) for row in records)
    prior_modes = {
        str(row.get("prior_mode"))
        for row in records
        if row.get("prior_mode") is not None
    }
    uniform_hidden = prior_modes == {"uniform_hidden"}

    def numeric_summary(
        rows: Sequence[Mapping[str, Any]],
        field: str,
    ) -> dict[str, float | None]:
        values = [
            float(row[field])
            for row in rows
            if row.get(field) is not None
        ]
        return {
            "mean": mean(values) if values else None,
            "median": median(values) if values else None,
        }

    latest: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    belief_latest: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    exclusions: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in records:
        sample_key = (
            str(row.get("sample_path") or row.get("sample")),
            str(row.get("calibration")),
            int(row.get("corruption_count", -1)),
        )
        if row.get("method") == "common_exclusion":
            exclusions[sample_key] = row
        elif row.get("method") == "belief":
            belief_latest[sample_key + (int(row.get("round", -1)),)] = row
        elif row.get("method") in METHODS:
            latest[
                sample_key
                + (str(row.get("method")), int(row.get("round", -1)))
            ] = row

    condition_keys = sorted(
        {
            key[:3]
            for key in latest
        }
        | set(exclusions)
    )
    complete: list[tuple[str, str, int]] = []
    mismatches: list[dict[str, Any]] = []
    for condition in condition_keys:
        if condition in exclusions:
            continue
        missing = [
            f"{method}:k{round_index}"
            for method in METHODS
            for round_index in (0, 1)
            if condition + (method, round_index) not in latest
        ]
        if missing:
            mismatches.append({"condition": list(condition), "missing": missing})
        else:
            complete.append(condition)

    method_round: dict[str, Any] = {}
    for corruption_count in (0, 1, 2):
        for method in METHODS:
            for round_index in (0, 1):
                keys = [key for key in complete if key[2] == corruption_count]
                rows = [latest[key + (method, round_index)] for key in keys]
                belief_size = numeric_summary(rows, "belief_size")
                probability_mass = numeric_summary(rows, "confidence")
                final_runtime = numeric_summary(rows, "runtime_seconds")
                method_round[f"m{corruption_count}:{method}:k{round_index}"] = {
                    "n": len(rows),
                    "success_rate": (
                        mean(bool(row.get("success")) for row in rows)
                        if rows
                        else None
                    ),
                    "plan_found_rate": (
                        mean(bool(row.get("plan_found")) for row in rows)
                        if rows
                        else None
                    ),
                    "true_state_inclusion_rate": (
                        mean(bool(row.get("true_state_in_set")) for row in rows)
                        if rows
                        else None
                    ),
                    "map_correct_rate": (
                        mean(bool(row.get("map_correct")) for row in rows)
                        if rows
                        else None
                    ),
                    "mean_belief_size": belief_size["mean"],
                    "median_belief_size": belief_size["median"],
                    "mean_probability_mass": probability_mass["mean"],
                    "median_probability_mass": probability_mass["median"],
                    "mean_runtime_seconds": final_runtime["mean"],
                    "median_runtime_seconds": final_runtime["median"],
                    "timeout_rate": (
                        mean(row.get("status") == "timeout" for row in rows)
                        if rows
                        else None
                    ),
                    "error_rate": (
                        mean(
                            row.get("status")
                            in {"error", "unsupported", "memory_limit"}
                            for row in rows
                        )
                        if rows
                        else None
                    ),
                    "status_counts": dict(
                        Counter(str(row.get("status")) for row in rows)
                    ),
                }

    paired: dict[str, Any] = {}
    for corruption_count in (0, 1, 2):
        for round_index in (0, 1):
            keys = [key for key in complete if key[2] == corruption_count]
            for left, right in (
                ("ams", "map"),
                ("adaptive_abstention", "map"),
                ("ams", "adaptive_abstention"),
            ):
                deltas = {
                    "|".join(key[:2]): float(
                        bool(latest[key + (left, round_index)].get("success"))
                    )
                    - float(
                        bool(latest[key + (right, round_index)].get("success"))
                    )
                    for key in keys
                }
                paired[f"m{corruption_count}:k{round_index}:{left}-{right}"] = (
                    paired_bootstrap_mean_ci(deltas, seed=bootstrap_seed)
                )

    representation: dict[str, Any] = {}
    plan_discrimination: dict[str, Any] = {}
    for corruption_count in (0, 1, 2):
        keys = [key for key in complete if key[2] == corruption_count]
        for round_index in (0, 1):
            belief_rows = [
                belief_latest[key + (round_index,)]
                for key in keys
                if key + (round_index,) in belief_latest
            ]
            representation[f"m{corruption_count}:k{round_index}"] = {
                "n": len(belief_rows),
                "map_correct_rate": (
                    mean(bool(row.get("map_correct")) for row in belief_rows)
                    if belief_rows
                    else None
                ),
                "true_state_in_ordered_set_rate": (
                    mean(
                        bool(row.get("true_state_in_ordered_set"))
                        for row in belief_rows
                    )
                    if belief_rows
                    else None
                ),
                "mean_ordered_state_count": (
                    mean(
                        float(row.get("ordered_state_count", 0))
                        for row in belief_rows
                    )
                    if belief_rows
                    else None
                ),
                "fixed_mass": {
                    f"{theta:g}": {
                        "mean_size": (
                            mean(
                                float(
                                    row.get("fixed_theta_metrics", {})
                                    .get(f"{theta:g}", {})
                                    .get("size", 0)
                                )
                                for row in belief_rows
                            )
                            if belief_rows
                            else None
                        ),
                        "true_state_coverage": (
                            mean(
                                bool(
                                    row.get("fixed_theta_metrics", {})
                                    .get(f"{theta:g}", {})
                                    .get("true_state_in_set")
                                )
                                for row in belief_rows
                            )
                            if belief_rows
                            else None
                        ),
                    }
                    for theta in FIXED_THETAS
                },
            }

            map_ams_equal: list[bool] = []
            map_abst_equal: list[bool] = []
            any_disagreement: list[bool] = []
            success_diff_equal_length: list[bool] = []
            success_when_disagree: dict[str, list[bool]] = defaultdict(list)
            for key in keys:
                rows = {
                    method: latest[key + (method, round_index)]
                    for method in METHODS
                }
                action_sequences = {
                    method: tuple(rows[method].get("actions", ()))
                    for method in METHODS
                }
                map_ams_equal.append(action_sequences["map"] == action_sequences["ams"])
                map_abst_equal.append(
                    action_sequences["map"]
                    == action_sequences["adaptive_abstention"]
                )
                disagree = len(set(action_sequences.values())) > 1
                any_disagreement.append(disagree)
                lengths = {
                    method: rows[method].get("plan_length")
                    for method in METHODS
                }
                success_diff_equal_length.append(
                    len(set(lengths.values())) == 1
                    and len(
                        {
                            bool(rows[method].get("success"))
                            for method in METHODS
                        }
                    )
                    > 1
                )
                if disagree:
                    for method in METHODS:
                        success_when_disagree[method].append(
                            bool(rows[method].get("success"))
                        )
            plan_discrimination[f"m{corruption_count}:k{round_index}"] = {
                "n": len(keys),
                "map_ams_action_equality_rate": (
                    mean(map_ams_equal) if map_ams_equal else None
                ),
                "map_abstention_action_equality_rate": (
                    mean(map_abst_equal) if map_abst_equal else None
                ),
                "any_method_plan_disagreement_rate": (
                    mean(any_disagreement) if any_disagreement else None
                ),
                "success_difference_with_equal_plan_length_rate": (
                    mean(success_diff_equal_length)
                    if success_diff_equal_length
                    else None
                ),
                "success_given_plan_disagreement": {
                    method: (
                        mean(success_when_disagree[method])
                        if success_when_disagree[method]
                        else None
                    )
                    for method in METHODS
                },
            }

        paired_belief_rows = [
            (
                belief_latest[key + (0,)],
                belief_latest[key + (1,)],
            )
            for key in keys
            if key + (0,) in belief_latest and key + (1,) in belief_latest
        ]
        representation[f"m{corruption_count}:observation_change"] = {
            "n": len(paired_belief_rows),
            "map_changed_rate": (
                mean(
                    tuple(round0.get("map_state", ()))
                    != tuple(round1.get("map_state", ()))
                    for round0, round1 in paired_belief_rows
                )
                if paired_belief_rows
                else None
            ),
            "ordering_changed_rate": (
                mean(
                    round0.get("ordered_states") != round1.get("ordered_states")
                    for round0, round1 in paired_belief_rows
                )
                if paired_belief_rows
                else None
            ),
        }

    transitions: dict[str, Any] = {}
    for corruption_count in (0, 1, 2):
        keys = [key for key in complete if key[2] == corruption_count]
        for method in METHODS:
            counts: Counter[str] = Counter()
            for key in keys:
                before = bool(latest[key + (method, 0)].get("success"))
                after = bool(latest[key + (method, 1)].get("success"))
                label = {
                    (True, True): "retained_success",
                    (False, True): "recovered",
                    (True, False): "regressed",
                    (False, False): "failed_both",
                }[(before, after)]
                counts[label] += 1
            transitions[f"m{corruption_count}:{method}"] = {
                "n": len(keys),
                **{
                    label: counts[label]
                    for label in (
                        "retained_success",
                        "recovered",
                        "regressed",
                        "failed_both",
                    )
                },
            }

    method_efficiency: dict[str, Any] = {}
    search_runtime_field = {
        "ams": "ams_search_runtime_seconds",
        "adaptive_abstention": "abstention_search_runtime_seconds",
    }
    for corruption_count in (0, 1, 2):
        keys = [key for key in complete if key[2] == corruption_count]
        for round_index in (0, 1):
            for method, runtime_field in search_runtime_field.items():
                rows = [latest[key + (method, round_index)] for key in keys]
                size = numeric_summary(rows, "belief_size")
                mass = numeric_summary(rows, "confidence")
                search_runtime = numeric_summary(rows, runtime_field)
                attempts = [
                    len(row.get("attempts", ()))
                    for row in rows
                    if row.get("attempts") is not None
                ]
                method_efficiency[
                    f"m{corruption_count}:{method}:k{round_index}"
                ] = {
                    "n": len(rows),
                    "mean_belief_size": size["mean"],
                    "median_belief_size": size["median"],
                    "mean_probability_mass": mass["mean"],
                    "median_probability_mass": mass["median"],
                    "mean_total_search_runtime_seconds": search_runtime["mean"],
                    "median_total_search_runtime_seconds": (
                        search_runtime["median"]
                    ),
                    "mean_search_attempts": (
                        mean(attempts) if attempts else None
                    ),
                }

    adaptive_invariants: dict[str, Any] = {}
    for corruption_count in (0, 1, 2):
        keys = [key for key in complete if key[2] == corruption_count]
        evaluated = [
            (key, round_index)
            for key in keys
            for round_index in (0, 1)
        ]
        availability_violations = [
            [key[0], key[1], round_index]
            for key, round_index in evaluated
            if bool(latest[key + ("map", round_index)].get("plan_found"))
            and not bool(
                latest[
                    key + ("adaptive_abstention", round_index)
                ].get("plan_found")
            )
        ]
        singleton_reuse_failures = [
            [key[0], key[1], round_index]
            for key, round_index in evaluated
            if not bool(
                latest[
                    key + ("adaptive_abstention", round_index)
                ].get("map_singleton_reused")
            )
        ]
        zero_candidate_failures = []
        for key, round_index in evaluated:
            attempts = latest[
                key + ("adaptive_abstention", round_index)
            ].get("attempts", ())
            if (
                not attempts
                or attempts[0].get("abstained_count") != 0
                or attempts[0].get("belief_size") != 1
            ):
                zero_candidate_failures.append(
                    [key[0], key[1], round_index]
                )
        adaptive_invariants[f"m{corruption_count}"] = {
            "evaluated_method_rounds": len(evaluated),
            "map_plan_availability_violations": availability_violations,
            "map_singleton_reuse_failures": singleton_reuse_failures,
            "zero_abstention_candidate_failures": zero_candidate_failures,
            "satisfied": not (
                availability_violations
                or singleton_reuse_failures
                or zero_candidate_failures
            ),
        }

    corrupted = [key for key in complete if key[2] in (1, 2)]
    disagreements = []
    map_ams_differences = []
    query_strata: dict[str, list[bool]] = defaultdict(list)
    for key in corrupted:
        rows = {
            (method, round_index): latest[key + (method, round_index)]
            for method in METHODS
            for round_index in (0, 1)
        }
        disagreements.append(
            any(
                len(
                    {
                        tuple(rows[(method, round_index)].get("actions", ()))
                        for method in METHODS
                    }
                )
                > 1
                for round_index in (0, 1)
            )
        )
        map_ams_differences.append(
            any(
                bool(rows[("map", round_index)].get("success"))
                != bool(rows[("ams", round_index)].get("success"))
                for round_index in (0, 1)
            )
        )
        hit = bool(rows[("map", 0)].get("query_hit_corruption"))
        query_strata["hit" if hit else "miss"].append(
            bool(rows[("ams", 1)].get("success"))
            != bool(rows[("map", 1)].get("success"))
        )

    m0 = [key for key in complete if key[2] == 0]
    map_retention = (
        mean(bool(latest[key + ("map", 1)].get("success")) for key in m0)
        if m0
        else None
    )
    ams_retention = (
        mean(bool(latest[key + ("ams", 1)].get("success")) for key in m0)
        if m0
        else None
    )
    all_method_rows = [
        latest[key + (method, round_index)]
        for key in complete
        for method in METHODS
        for round_index in (0, 1)
    ]
    abst_rows = [
        latest[key + ("adaptive_abstention", round_index)]
        for key in complete
        for round_index in (0, 1)
    ]
    timeout_rate = (
        mean(row.get("status") == "timeout" for row in all_method_rows)
        if all_method_rows
        else 1.0
    )
    disagreement_rate = mean(disagreements) if disagreements else 0.0
    map_ams_difference_rate = (
        mean(map_ams_differences) if map_ams_differences else 0.0
    )
    retention_regression = (
        map_retention - ams_retention
        if map_retention is not None and ams_retention is not None
        else None
    )
    controlled_criteria = {
        "plan_disagreement_at_least_20pct": disagreement_rate >= 0.20,
        "map_ams_execution_difference_at_least_10pct": (
            map_ams_difference_rate >= 0.10
        ),
        "no_denominator_mismatch": not mismatches,
        "no_severe_zero_corruption_ams_regression": (
            retention_regression is not None and retention_regression <= 0.10
        ),
        "runtime_compatible": timeout_rate <= 0.25,
        "adaptive_abstention_fully_operational": bool(abst_rows)
        and all(
            row.get("status") not in {"error", "unsupported", "memory_limit"}
            for row in abst_rows
        ),
    }
    go = (
        (
            controlled_criteria["plan_disagreement_at_least_20pct"]
            or controlled_criteria[
                "map_ams_execution_difference_at_least_10pct"
            ]
        )
        and all(list(controlled_criteria.values())[2:])
    )
    controlled_decision = {
        "decision": "go" if go else "no-go",
        "criteria": controlled_criteria,
        "corrupted_plan_disagreement_rate": disagreement_rate,
        "corrupted_map_ams_execution_difference_rate": map_ams_difference_rate,
        "zero_corruption_map_k1_success": map_retention,
        "zero_corruption_ams_k1_success": ams_retention,
        "zero_corruption_ams_regression": retention_regression,
        "planner_timeout_rate": timeout_rate,
        "planning_timeout_seconds": planning_timeout,
        "interpretation": (
            "Pilot passes preregistered discrimination/fairness/operations criteria."
            if go
            else "Stop before Phase F; at least one preregistered go criterion failed."
        ),
    }
    if uniform_hidden:
        uniform_criteria = {
            "at_least_100_complete_pairs": len(complete) >= 100,
            "no_denominator_mismatch": not mismatches,
            "adaptive_map_fallback_satisfied": adaptive_invariants[
                "m0"
            ]["satisfied"],
            "no_method_errors": all(
                row.get("status")
                not in {"error", "unsupported", "memory_limit"}
                for row in all_method_rows
            ),
            "runtime_compatible": timeout_rate <= 0.25,
        }
        decision = {
            "decision": (
                "analysis-ready"
                if all(uniform_criteria.values())
                else "invalid"
            ),
            "criteria": uniform_criteria,
            "complete_sample_conditions": len(complete),
            "common_exclusions": len(exclusions),
            "planner_timeout_rate": timeout_rate,
            "planning_timeout_seconds": planning_timeout,
            "interpretation": (
                "Uniform-hidden run passes its sample-size, pairing, "
                "fallback, and planner-health integrity checks."
                if all(uniform_criteria.values())
                else (
                    "Do not interpret the uniform-hidden outcomes until all "
                    "integrity failures are resolved."
                )
            ),
        }
    else:
        decision = controlled_decision
    return {
        "protocol_version": PROTOCOL_VERSION,
        "selected_sample_fingerprint": fingerprint,
        "complete_sample_conditions": len(complete),
        "common_exclusions": len(exclusions),
        "denominator_mismatches": mismatches,
        "method_round": method_round,
        "paired_success_differences": paired,
        "representation": representation,
        "transitions": transitions,
        "plan_discrimination": plan_discrimination,
        "method_efficiency": method_efficiency,
        "adaptive_invariants": adaptive_invariants,
        "query_hit_strata": {
            name: {
                "n": len(values),
                "map_ams_k1_difference_rate": mean(values) if values else None,
            }
            for name, values in query_strata.items()
        },
        "analysis_decision": decision,
        "phase_e": controlled_decision if not uniform_hidden else None,
    }


def _load_latest(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in iter_jsonl(path):
        latest[result_key(record, RESULT_KEY_FIELDS)] = record
    return latest


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if any(count not in (0, 1, 2) for count in args.corruption_counts):
        raise ValueError("primary corruption counts must be drawn from 0, 1, 2")
    if args.prior_mode == "uniform_hidden" and tuple(args.corruption_counts) != (0,):
        raise ValueError(
            "--prior-mode uniform_hidden requires --corruption-counts 0; "
            "there is no aligned/decoy corruption treatment in this condition"
        )
    os.environ.setdefault("MONO_CRASH_NOFILE", "1")
    os.environ.setdefault("MBS_PLANNER_MAX_GIB", "15")
    output_dir, results_path, aggregate_path = experiment_paths(
        args.output_root, args.model, args.run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase_a_only:
        _eligible, _version, report = phase_a_scan(args, output_dir)
        print(json.dumps(report, indent=2))
        return
    if args.overwrite and results_path.exists():
        results_path.unlink()

    selected, fingerprint, relevance_version, metadata = prepare_population(
        args, output_dir
    )
    if args.prepare_only:
        print(f"Prepared {len(selected)} samples ({fingerprint})")
        return
    config = Experiment6Config(
        prior_mode=args.prior_mode,
        sensor_confidence=args.sensor_confidence,
        corruption_counts=tuple(args.corruption_counts),
        corruption_seed=args.corruption_seed,
        query_seed=args.query_seed,
        abstention_threshold=args.abstention_threshold,
        abstention_calibration_id=args.abstention_calibration_id,
        max_explicit_states=args.max_explicit_states,
        planning_timeout=args.planning_timeout,
        method_timeout=args.method_timeout,
        relevance_version=relevance_version,
    )
    tasks: list[Experiment6Task] = []
    for sid, units in selected:
        treatments = tuple(
            freeze_treatment(
                units,
                sample_id=_sample_line(sid),
                corruption_count=count,
                corruption_seed=args.corruption_seed,
                query_seed=args.query_seed,
            )
            for count in args.corruption_counts
        )
        for calibration in args.calibrations:
            tasks.append(
                Experiment6Task(
                    sid,
                    calibration,
                    args.model,
                    units,
                    treatments,
                    fingerprint,
                    config,
                )
            )

    latest = _load_latest(results_path)
    retry = set(args.retry_status)

    def task_needs_run(task: Experiment6Task) -> bool:
        for treatment in task.treatments:
            exclusion_key = result_key(
                _semantic_context(
                    task,
                    treatment,
                    method="common_exclusion",
                    round_index=None,
                ),
                RESULT_KEY_FIELDS,
            )
            exclusion = latest.get(exclusion_key)
            if exclusion is not None and str(exclusion.get("status")) not in retry:
                continue
            for method in ("belief",) + METHODS + ("oracle",):
                for round_index in (0, 1):
                    key = result_key(
                        _semantic_context(
                            task,
                            treatment,
                            method=method,
                            round_index=round_index,
                        ),
                        RESULT_KEY_FIELDS,
                    )
                    row = latest.get(key)
                    if row is None or str(row.get("status")) in retry:
                        return True
        return False

    tasks = [task for task in tasks if task_needs_run(task)]

    def record_all(rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            payload = dict(row)
            append_jsonl(results_path, payload)
            latest[result_key(payload, RESULT_KEY_FIELDS)] = payload

    jobs = min(max(1, args.jobs), max(1, len(tasks)))
    if jobs > 1 and tasks:
        ctx = mp.get_context("spawn")
        with futures.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as executor:
            submitted = {
                executor.submit(run_experiment_6_task, task): task for task in tasks
            }
            for future in tqdm(
                futures.as_completed(submitted),
                total=len(submitted),
                desc="Exp6 samples",
            ):
                record_all(future.result())
    else:
        for task in tqdm(tasks, desc="Exp6 samples"):
            record_all(run_experiment_6_task(task))

    write_jsonl_atomic(results_path, latest.values())
    aggregate = aggregate_experiment_6(
        latest.values(),
        fingerprint=fingerprint,
        bootstrap_seed=args.sample_seed,
        planning_timeout=args.planning_timeout,
    )
    aggregate["config"] = metadata
    write_json_atomic(aggregate_path, aggregate)
    write_json_atomic(
        output_dir / "analysis_decision.json",
        aggregate["analysis_decision"],
    )
    if aggregate["phase_e"] is not None:
        write_json_atomic(
            output_dir / "phase_e_decision.json",
            aggregate["phase_e"],
        )
    else:
        write_json_atomic(
            output_dir / "phase_e_decision.json",
            {
                "decision": "not-applicable",
                "reason": (
                    "The uniform-hidden protocol has no m1/m2 controlled-"
                    "corruption conditions."
                ),
                "replacement": "analysis_decision.json",
            },
        )
    print(f"Wrote {results_path}")
    print(f"Wrote {aggregate_path}")
    print(json.dumps(aggregate["analysis_decision"], indent=2))


if __name__ == "__main__":
    main()
