#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import multiprocessing as mp
import os
import sys
import time
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
    fluent_abstention_unconstrained,
    map_baseline,
    mbs_first,
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
    load_existing_keys,
    load_json,
    make_error_record,
    result_key,
    write_json_atomic,
    write_jsonl_atomic,
)
from mbs_experiments.metrics import aggregate_experiment_1, contains_truth
from mbs_experiments.normalizers import selected_groups_runtime_normalizer
from mbs_experiments.states import (
    constraint_satisfying_full_state_count,
    logical_full_state_count,
    null_indices,
    preview_state_set,
    project_feature_names,
    project_groups,
    project_state,
    state_set_size,
    truth_state_from_ground_truth,
    visible_indices,
)

# Outputs are model-sharded: experiments_data/experiment_1_static_quality/
# <model>/{sample_results.jsonl,aggregate_metrics.json}. Combined with the
# `model` field in every record + RESULT_KEY_FIELDS, results for different
# models never collide, mix, or silently resume each other.
OUTPUT_ROOT = Path("experiments_data/experiment_1_static_quality")


def experiment_paths(model: str) -> tuple[Path, Path, Path]:
    """Return (output_dir, results_path, agg_path) for a model."""
    base = OUTPUT_ROOT / model_to_path_segment(model)
    return base, base / "sample_results.jsonl", base / "aggregate_metrics.json"


DEFAULT_MODEL = "OpenAI/gpt-4.1-2025-04-14"
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
STATE_SET_METHODS = frozenset(
    {
        "mbs_prevas",
        "mbs_first",
        "mbs_first_constraints",
        "valid_map",
        "map",
        "unconstrained_top_mass",
        "random_theta_confident",
        "fluent_abstention_unconstrained",
    }
)
FULL_SET_SIZE_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 1 static belief-set quality")
    parser.add_argument("--domains", nargs="+", default=("prb", "gw", "alf"))
    parser.add_argument("--configs", nargs="+")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model identifier whose predictions / Platt profiles / calibrations "
            "to consume. Must match the --model passed to make_predictions.py, "
            f"calibrate_vlm.py, calibrate_tau.py. "
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
    parser.add_argument("--state-preview-limit", type=int, default=100)
    parser.add_argument(
        "--method-timeout",
        type=float,
        default=30.0,
        help=(
            "Wallclock timeout (seconds) applied to every belief-set method, "
            "so even a pathological run is bounded. Default: 30."
        ),
    )
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


def stable_seed(*parts: object) -> int:
    text = "|".join(map(str, parts))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) & 0xFFFFFFFF


def downstream_full_set_size(
    bundle,
    method: str,
    state_set,
    *,
    compact_set_size: int,
    omitted_feature_count: int,
    search_indices: tuple[int, ...],
    full_set_size_groups: tuple[tuple[int, ...], ...] | None = None,
) -> int:
    """Size of the full feature-space belief represented downstream in Exp. 3.

    MBS-like methods choose explicit assignments for the non-null/search
    fluents. Null fluents remain unknown downstream, so we count all full
    extensions of those chosen assignments that still satisfy the applicable
    at-most-one constraints. Unconstrained top-mass intentionally ignores those
    constraints, so its downstream full size is the Cartesian product with null
    fluent assignments. Feature-abstention uses a committed/wildcard template;
    downstream semantics are simply all wildcard completions.
    """
    if method in {"map", "valid_map"}:
        # Experiment 3 commits singleton baselines over the full feature set.
        return compact_set_size
    if method == "unconstrained_top_mass":
        return logical_full_state_count(compact_set_size, omitted_feature_count)
    if state_set.committed_template is not None:
        return logical_full_state_count(compact_set_size, omitted_feature_count)
    return constraint_satisfying_full_state_count(
        state_set,
        full_feature_count=len(bundle.feature_names),
        selected_groups=full_set_size_groups
        if full_set_size_groups is not None
        else bundle.selected_groups,
        search_indices=search_indices,
    )


def make_record(
    bundle,
    method: str,
    theta: float | None,
    state_set,
    *,
    runtime_seconds: float,
    state_preview_limit: int,
    extra: dict[str, Any] | None = None,
    search_feature_names: tuple[str, ...] | None = None,
    search_truth: tuple[bool, ...] | None = None,
    search_indices: tuple[int, ...] | None = None,
    full_set_size_groups: tuple[tuple[int, ...], ...] | None = None,
) -> dict[str, Any]:
    feature_names = (
        search_feature_names if search_feature_names is not None else bundle.feature_names
    )
    truth = (
        search_truth
        if search_truth is not None
        else truth_state_from_ground_truth(bundle.ground_truth)
    )
    omitted_feature_count = 0
    if extra and "omitted_feature_count" in extra:
        omitted_feature_count = int(extra["omitted_feature_count"])
    compact_set_size = state_set_size(state_set)
    logical_set_size = logical_full_state_count(
        compact_set_size, omitted_feature_count
    )
    if search_indices is None:
        if search_feature_names is None:
            search_indices = tuple(range(len(bundle.feature_names)))
        else:
            full_name_to_idx = {name: i for i, name in enumerate(bundle.feature_names)}
            search_indices = tuple(full_name_to_idx[name] for name in feature_names)
    full_set_size = downstream_full_set_size(
        bundle,
        method,
        state_set,
        compact_set_size=compact_set_size,
        omitted_feature_count=omitted_feature_count,
        search_indices=search_indices,
        full_set_size_groups=full_set_size_groups,
    )
    base = {
        "status": state_set.status,
        "domain": bundle.sample_id.domain,
        "configuration": bundle.sample_id.configuration,
        "sample": bundle.sample_id.sample,
        "split_relative_path": bundle.sample_id.split_relative_path,
        "calibration": bundle.calibration,
        "model": bundle.model,
        "method": method,
        "theta": theta,
        "feature_count": len(bundle.feature_names),
        "search_feature_count": len(feature_names),
        "selected_group_count": len(bundle.selected_groups),
        "normalizer": None,
        "invalid_mass": None,
        "set_size": logical_set_size,
        "compact_set_size": compact_set_size,
        "logical_full_set_size": logical_set_size,
        "full_set_size": full_set_size,
        "full_set_size_version": FULL_SET_SIZE_VERSION,
        "contains_truth": contains_truth(state_set, truth),
        "unconstrained_mass": state_set.unconstrained_mass,
        "constrained_mass": state_set.constrained_mass,
        "total_states_considered": state_set.total_states_considered,
        "runtime_seconds": runtime_seconds,
        "state_preview": preview_state_set(
            state_set, feature_names, state_preview_limit
        ),
    }
    if extra:
        base.update(extra)
    return base


def record_needs_full_set_size(
    record: dict[str, Any],
    sample_keys: set[tuple[str, str, str]],
    calibrations: set[str],
) -> bool:
    sample_key = (
        str(record.get("domain")),
        str(record.get("configuration")),
        str(record.get("sample")),
    )
    return (
        sample_key in sample_keys
        and record.get("calibration") in calibrations
        and record.get("method") in STATE_SET_METHODS
        and record.get("status") in {"ok", "timeout"}
        and (
            "full_set_size" not in record
            or (
                record.get("method") == "unconstrained_top_mass"
                and record.get("full_set_size_version") != FULL_SET_SIZE_VERSION
            )
        )
    )



@dataclass(frozen=True)
class Experiment1Config:
    method_timeout: float
    state_preview_limit: int


@dataclass(frozen=True)
class Experiment1Task:
    sid: SampleId
    calibration: str
    config: Experiment1Config
    model: str
    tau_lookup: dict[tuple[str, str, float], float]
    # Model-sharded results path (workers don't write; kept for coherence).
    results_path: Path
    existing_keys: set[tuple[Any, ...]]


def _worker_count(requested_jobs: int | None) -> int:
    default_jobs = max(1, (os.cpu_count() or 1) - 1)
    if requested_jobs is None or requested_jobs <= 0:
        return default_jobs
    return requested_jobs


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


def _existing_keys_for_task(
    existing: set[tuple[Any, ...]],
    sid: SampleId,
    calibration: str,
) -> set[tuple[Any, ...]]:
    prefix = (sid.domain, sid.configuration, sid.sample, calibration)
    return {key for key in existing if key[:4] == prefix}


def _run_experiment_1_calibration_task(task: Experiment1Task) -> list[dict[str, Any]]:
    """Run all Experiment 1 methods for one sample/calibration.

    Workers never write to the shared JSONL.  They return records to the parent
    process, which serializes writes and keeps resume/aggregate state coherent.
    """
    sid = task.sid
    config = task.config
    model = task.model
    results_path = task.results_path
    tau_lookup = task.tau_lookup
    existing = set(task.existing_keys)
    records: list[dict[str, Any]] = []
    available = (task.calibration,)

    def record_result(record: Mapping[str, Any]) -> None:
        payload = dict(record)
        key = result_key(payload, RESULT_KEY_FIELDS)
        if key in existing:
            return
        records.append(payload)
        existing.add(key)

    def append_jsonl(_path: Path, record: Mapping[str, Any]) -> None:
        # Shadow the module-level append_jsonl used by the sequential runner so
        # the calibration body below can stay close to the original logic while
        # avoiding concurrent writes from worker processes.
        record_result(record)

    def append_method_error(
        sid: SampleId,
        calibration: str | None,
        method: str,
        theta: float | None,
        exc: BaseException,
        *,
        existing: set[tuple[Any, ...]],
    ) -> None:
        del existing
        record_result(
            make_error_record(
                _base_error_context(
                    sid, calibration, method, theta, model=model
                ),
                exc,
            )
        )

    for calibration in sorted(available):
        bundle_error_key = experiment_key(
            sid, calibration, "bundle", None, model=model
        )
        if bundle_error_key in existing:
            continue
        try:
            bundle = load_calibration_bundle(
                sid, calibration, model=task.model, split=TEST_SPLIT
            )
            full_z = selected_groups_runtime_normalizer(
                bundle, feature_space="full"
            )
            search_idx = visible_indices(bundle.none_is_max)
            omitted_idx = null_indices(bundle.none_is_max)
            search_feature_names = project_feature_names(
                bundle.feature_names, search_idx
            )
            omitted_feature_names = project_feature_names(
                bundle.feature_names, omitted_idx
            )
            search_truth = project_state(bundle.ground_truth, search_idx)
            reduced_probs = tuple(bundle.probabilities[i] for i in search_idx)
            reduced_groups = project_groups(bundle.selected_groups, search_idx)
            reduced_z = selected_groups_runtime_normalizer(
                bundle, feature_space="visible"
            )
            if sid.domain == "prb":
                # Match the constrained-MBS setup used in Experiment 3:
                # PRB keeps the full feature vocabulary so support/clear
                # invariants are not weakened by hidden none_is_max atoms.
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
            constraint_truth = project_state(
                bundle.ground_truth, constraint_search_idx
            )
            constraint_probs = tuple(
                bundle.probabilities[i] for i in constraint_search_idx
            )
            reduced_constraint_groups = project_groups(
                bundle.constraint_groups, constraint_search_idx
            )
            # Experiment 1 constraint baselines use the extracted mutexes as
            # at-most-one constraints. Experiment 3 has additional planning-
            # specific exact-one variants, but here the requested variants
            # are selected_groups vs. the full extracted constraints set.
            exactly_one_constraints = False
        except Exception as exc:
            append_jsonl(
                results_path,
                make_error_record(
                    _base_error_context(
                        sid, calibration, "bundle", None, model=model
                    ),
                    exc,
                ),
            )
            existing.add(bundle_error_key)
            continue

        invalid_mass = 1.0 - reduced_z
        base_extra = {
            "normalizer": reduced_z,
            "invalid_mass": invalid_mass,
            "full_normalizer": full_z,
            "full_invalid_mass": 1.0 - full_z,
            "search_selected_group_count": len(reduced_groups),
            "omitted_feature_count": len(omitted_idx),
            "omitted_feature_names": omitted_feature_names,
            "valid_map_constrained_mass": None,
            "valid_map_unconstrained_mass": None,
        }
        constraints_extra = {
            "normalizer": 1.0,
            "invalid_mass": 0.0,
            "full_normalizer": None,
            "full_invalid_mass": None,
            "search_selected_group_count": len(reduced_constraint_groups),
            "search_constraint_group_count": len(reduced_constraint_groups),
            "omitted_feature_count": len(constraint_omitted_idx),
            "omitted_feature_names": constraint_omitted_feature_names,
            "validity_constraint_set": "constraints",
            "normalizer_constraint_set": "unit_for_first_state_only",
            "constraints_exactly_one": exactly_one_constraints,
        }
        tau_per_theta = {
            theta: tau_lookup.get((sid.domain, calibration, theta))
            for theta in THETA_GRID
        }
        has_tau = any(tau is not None for tau in tau_per_theta.values())
        method_names = [
            "mbs_prevas",
            "mbs_first",
            "mbs_first_constraints",
            "valid_map",
            "map",
            "unconstrained_top_mass",
            "random_theta_confident",
        ]
        if has_tau:
            method_names.append("fluent_abstention_unconstrained")

        def _any_theta_missing(method: str) -> bool:
            return any(
                (
                    sid.domain,
                    sid.configuration,
                    sid.sample,
                    calibration,
                    method,
                    theta,
                )
                not in existing
                for theta in THETA_GRID
            )

        valid_map_cache: tuple[Any, float] | BaseException | None = None
        if _any_theta_missing("valid_map"):
            start = time.perf_counter()
            try:
                valid_map_cache = (
                    valid_map(
                        reduced_probs,
                        reduced_groups,
                        reduced_z,
                        timeout_seconds=config.method_timeout,
                    ),
                    time.perf_counter() - start,
                )
            except Exception as exc:
                valid_map_cache = exc

        if isinstance(valid_map_cache, tuple):
            vm_state_set, _ = valid_map_cache
            base_extra["valid_map_constrained_mass"] = vm_state_set.constrained_mass
            base_extra["valid_map_unconstrained_mass"] = vm_state_set.unconstrained_mass

        mbs_first_cache: tuple[Any, float] | BaseException | None = None
        if _any_theta_missing("mbs_first"):
            start = time.perf_counter()
            try:
                mbs_first_cache = (
                    mbs_first(
                        reduced_probs,
                        reduced_groups,
                        reduced_z,
                        timeout_seconds=config.method_timeout,
                    ),
                    time.perf_counter() - start,
                )
            except Exception as exc:
                mbs_first_cache = exc

        mbs_first_constraints_cache: tuple[Any, float] | BaseException | None = None
        if _any_theta_missing("mbs_first_constraints"):
            start = time.perf_counter()
            try:
                mbs_first_constraints_cache = (
                    mbs_first(
                        constraint_probs,
                        reduced_constraint_groups,
                        1.0,
                        timeout_seconds=config.method_timeout,
                        exactly_one=exactly_one_constraints,
                    ),
                    time.perf_counter() - start,
                )
            except Exception as exc:
                mbs_first_constraints_cache = exc

        map_cache: tuple[Any, float] | BaseException | None = None
        if _any_theta_missing("map"):
            start = time.perf_counter()
            try:
                map_cache = (
                    map_baseline(reduced_probs),
                    time.perf_counter() - start,
                )
            except Exception as exc:
                map_cache = exc

        mbs_prevas_timeout_cache: tuple[Any, float] | None = None
        unconstrained_top_mass_timeout_cache: tuple[Any, float] | None = None
        random_theta_timeout_cache: tuple[Any, float, int] | None = None

        for theta in THETA_GRID:
            theta_keys = {
                method: experiment_key(
                    sid, calibration, method, theta, model=model
                )
                for method in method_names
            }
            if all(key in existing for key in theta_keys.values()):
                continue

            if theta_keys["mbs_prevas"] not in existing:
                try:
                    if mbs_prevas_timeout_cache is not None:
                        mbs_result, runtime_seconds = mbs_prevas_timeout_cache
                    else:
                        start = time.perf_counter()
                        mbs_result = mbs_prevas(
                            reduced_probs,
                            reduced_groups,
                            theta,
                            reduced_z,
                            timeout_seconds=config.method_timeout,
                        )
                        runtime_seconds = time.perf_counter() - start
                        if (
                            mbs_result.stats.timeout
                            and mbs_result.state_set.constrained_mass < theta
                        ):
                            mbs_prevas_timeout_cache = (mbs_result, runtime_seconds)
                    record = make_record(
                        bundle,
                        "mbs_prevas",
                        theta,
                        mbs_result.state_set,
                        runtime_seconds=runtime_seconds,
                        state_preview_limit=config.state_preview_limit,
                        extra=base_extra | {"prevas_stats": mbs_result.stats.__dict__},
                        search_feature_names=search_feature_names,
                        search_truth=search_truth,
                        search_indices=search_idx,
                    )
                    append_jsonl(results_path, record)
                    existing.add(theta_keys["mbs_prevas"])
                except Exception as exc:
                    append_method_error(
                        sid,
                        calibration,
                        "mbs_prevas",
                        theta,
                        exc,
                        existing=existing,
                    )

            if theta_keys["mbs_first"] not in existing:
                if isinstance(mbs_first_cache, BaseException):
                    append_method_error(
                        sid,
                        calibration,
                        "mbs_first",
                        theta,
                        mbs_first_cache,
                        existing=existing,
                    )
                else:
                    try:
                        if mbs_first_cache is None:
                            raise RuntimeError("Missing mbs_first cache")
                        state_set, runtime_seconds = mbs_first_cache
                        record = make_record(
                            bundle,
                            "mbs_first",
                            theta,
                            state_set,
                            runtime_seconds=runtime_seconds,
                            state_preview_limit=config.state_preview_limit,
                            extra=base_extra
                            | {
                                "validity_constraint_set": "selected_groups",
                                "normalizer_constraint_set": "selected_groups",
                            },
                            search_feature_names=search_feature_names,
                            search_truth=search_truth,
                            search_indices=search_idx,
                            full_set_size_groups=bundle.selected_groups,
                        )
                        append_jsonl(results_path, record)
                        existing.add(theta_keys["mbs_first"])
                    except Exception as exc:
                        append_method_error(
                            sid,
                            calibration,
                            "mbs_first",
                            theta,
                            exc,
                            existing=existing,
                        )

            if theta_keys["mbs_first_constraints"] not in existing:
                if isinstance(mbs_first_constraints_cache, BaseException):
                    append_method_error(
                        sid,
                        calibration,
                        "mbs_first_constraints",
                        theta,
                        mbs_first_constraints_cache,
                        existing=existing,
                    )
                else:
                    try:
                        if mbs_first_constraints_cache is None:
                            raise RuntimeError("Missing mbs_first_constraints cache")
                        state_set, runtime_seconds = mbs_first_constraints_cache
                        record = make_record(
                            bundle,
                            "mbs_first_constraints",
                            theta,
                            state_set,
                            runtime_seconds=runtime_seconds,
                            state_preview_limit=config.state_preview_limit,
                            extra=constraints_extra,
                            search_feature_names=constraint_search_feature_names,
                            search_truth=constraint_truth,
                            search_indices=constraint_search_idx,
                            full_set_size_groups=bundle.constraint_groups,
                        )
                        append_jsonl(results_path, record)
                        existing.add(theta_keys["mbs_first_constraints"])
                    except Exception as exc:
                        append_method_error(
                            sid,
                            calibration,
                            "mbs_first_constraints",
                            theta,
                            exc,
                            existing=existing,
                        )

            if theta_keys["valid_map"] not in existing:
                if isinstance(valid_map_cache, BaseException):
                    append_method_error(
                        sid,
                        calibration,
                        "valid_map",
                        theta,
                        valid_map_cache,
                        existing=existing,
                    )
                else:
                    try:
                        if valid_map_cache is None:
                            raise RuntimeError("Missing valid_map cache")
                        state_set, runtime_seconds = valid_map_cache
                        record = make_record(
                            bundle,
                            "valid_map",
                            theta,
                            state_set,
                            runtime_seconds=runtime_seconds,
                            state_preview_limit=config.state_preview_limit,
                            extra=base_extra,
                            search_feature_names=search_feature_names,
                            search_truth=search_truth,
                            search_indices=search_idx,
                        )
                        append_jsonl(results_path, record)
                        existing.add(theta_keys["valid_map"])
                    except Exception as exc:
                        append_method_error(
                            sid,
                            calibration,
                            "valid_map",
                            theta,
                            exc,
                            existing=existing,
                        )

            if theta_keys["map"] not in existing:
                if isinstance(map_cache, BaseException):
                    append_method_error(
                        sid,
                        calibration,
                        "map",
                        theta,
                        map_cache,
                        existing=existing,
                    )
                else:
                    try:
                        if map_cache is None:
                            raise RuntimeError("Missing map cache")
                        state_set, runtime_seconds = map_cache
                        record = make_record(
                            bundle,
                            "map",
                            theta,
                            state_set,
                            runtime_seconds=runtime_seconds,
                            state_preview_limit=config.state_preview_limit,
                            extra=base_extra,
                            search_feature_names=search_feature_names,
                            search_truth=search_truth,
                            search_indices=search_idx,
                        )
                        append_jsonl(results_path, record)
                        existing.add(theta_keys["map"])
                    except Exception as exc:
                        append_method_error(
                            sid,
                            calibration,
                            "map",
                            theta,
                            exc,
                            existing=existing,
                        )

            if theta_keys["unconstrained_top_mass"] not in existing:
                try:
                    if unconstrained_top_mass_timeout_cache is not None:
                        state_set, runtime_seconds = unconstrained_top_mass_timeout_cache
                    else:
                        start = time.perf_counter()
                        state_set = unconstrained_top_mass_baseline(
                            reduced_probs,
                            theta,
                            timeout_seconds=config.method_timeout,
                        )
                        runtime_seconds = time.perf_counter() - start
                        if (
                            state_set.status == "timeout"
                            and state_set.constrained_mass < theta
                        ):
                            unconstrained_top_mass_timeout_cache = (
                                state_set,
                                runtime_seconds,
                            )
                    record = make_record(
                        bundle,
                        "unconstrained_top_mass",
                        theta,
                        state_set,
                        runtime_seconds=runtime_seconds,
                        state_preview_limit=config.state_preview_limit,
                        extra=base_extra,
                        search_feature_names=search_feature_names,
                        search_truth=search_truth,
                        search_indices=search_idx,
                    )
                    append_jsonl(results_path, record)
                    existing.add(theta_keys["unconstrained_top_mass"])
                except Exception as exc:
                    append_method_error(
                        sid,
                        calibration,
                        "unconstrained_top_mass",
                        theta,
                        exc,
                        existing=existing,
                    )

            if theta_keys["random_theta_confident"] not in existing:
                try:
                    random_seed = stable_seed(
                        sid.split_relative_path, calibration, theta
                    )
                    if random_theta_timeout_cache is not None:
                        state_set, runtime_seconds, _ = random_theta_timeout_cache
                    else:
                        start = time.perf_counter()
                        state_set = random_theta_confident(
                            reduced_probs,
                            reduced_groups,
                            theta,
                            reduced_z,
                            seed=random_seed,
                            timeout_seconds=config.method_timeout,
                        )
                        runtime_seconds = time.perf_counter() - start
                        if (
                            state_set.status == "timeout"
                            and state_set.constrained_mass < theta
                        ):
                            random_theta_timeout_cache = (
                                state_set,
                                runtime_seconds,
                                random_seed,
                            )
                    record = make_record(
                        bundle,
                        "random_theta_confident",
                        theta,
                        state_set,
                        runtime_seconds=runtime_seconds,
                        state_preview_limit=config.state_preview_limit,
                        extra=base_extra | {"random_seed": random_seed},
                        search_feature_names=search_feature_names,
                        search_truth=search_truth,
                        search_indices=search_idx,
                    )
                    append_jsonl(results_path, record)
                    existing.add(theta_keys["random_theta_confident"])
                except Exception as exc:
                    append_method_error(
                        sid,
                        calibration,
                        "random_theta_confident",
                        theta,
                        exc,
                        existing=existing,
                    )

            tau = tau_per_theta.get(theta)

            if (
                "fluent_abstention_unconstrained" in theta_keys
                and theta_keys["fluent_abstention_unconstrained"] not in existing
                and tau is not None
            ):
                try:
                    start = time.perf_counter()
                    state_set = fluent_abstention_unconstrained(
                        reduced_probs,
                        tau,
                        reduced_z,
                        selected_groups=reduced_groups,
                        timeout_seconds=config.method_timeout,
                    )
                    record = make_record(
                        bundle,
                        "fluent_abstention_unconstrained",
                        theta,
                        state_set,
                        runtime_seconds=time.perf_counter() - start,
                        state_preview_limit=config.state_preview_limit,
                        extra=base_extra | {"tau": tau},
                        search_feature_names=search_feature_names,
                        search_truth=search_truth,
                        search_indices=search_idx,
                    )
                    append_jsonl(results_path, record)
                    existing.add(theta_keys["fluent_abstention_unconstrained"])
                except Exception as exc:
                    append_method_error(
                        sid,
                        calibration,
                        "fluent_abstention_unconstrained",
                        theta,
                        exc,
                        existing=existing,
                    )

    return records


def main() -> None:
    args = parse_args()
    jobs = _worker_count(args.jobs)
    output_dir, results_path, agg_path = experiment_paths(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    domains = set(args.domains)
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

    sample_ids = filter_sample_ids(
        load_test_sample_ids(), domains=domains, configs=configs, limit=args.limit
    )
    sample_keys = {(sid.domain, sid.configuration, sid.sample) for sid in sample_ids}

    if args.overwrite and results_path.exists():
        results_path.unlink()
    reset_methods = set(args.reset_methods)
    if reset_methods and results_path.exists() and not args.overwrite:
        drop_method = lambda r: r.get("method") in reset_methods and r.get("domain") in domains
        kept = [
            r
            for r in iter_jsonl(results_path)
            if not drop_method(r)
        ]
        write_jsonl_atomic(results_path, kept)
        print(
            f"Reset {sorted(reset_methods)} for domains {sorted(domains)}: kept {len(kept)} rows in {results_path}"
        )
    if results_path.exists() and not args.overwrite:
        stale_full_size = 0
        kept = []
        for record in iter_jsonl(results_path):
            if record_needs_full_set_size(record, sample_keys, requested_calibrations):
                stale_full_size += 1
                continue
            kept.append(record)
        if stale_full_size:
            write_jsonl_atomic(results_path, kept)
            print(
                f"Recomputing {stale_full_size} Experiment 1 rows missing full_set_size"
            )
    existing = (
        set()
        if args.overwrite
        else load_existing_keys(results_path, RESULT_KEY_FIELDS)
    )

    def record_result(record: Mapping[str, Any]) -> None:
        payload = dict(record)
        key = result_key(payload, RESULT_KEY_FIELDS)
        if key in existing:
            return
        append_jsonl(results_path, payload)
        existing.add(key)

    config = Experiment1Config(
        method_timeout=args.method_timeout,
        state_preview_limit=args.state_preview_limit,
    )
    tasks: list[Experiment1Task] = []

    for sid in tqdm(sample_ids, desc="Exp1 inputs"):
        prediction_error_key = experiment_key(
            sid, None, "prediction", None, model=args.model
        )
        if prediction_error_key in existing:
            continue
        try:
            # Touch the prediction file so a missing one is reported as a
            # prediction error rather than failing later inside the worker.
            load_json(prediction_path(sid, TEST_SPLIT, args.model))
            available = (
                set(available_calibrations(args.model, sid.domain))
                & requested_calibrations
            )
        except Exception as exc:
            record_result(
                make_error_record(
                    _base_error_context(
                        sid, None, "prediction", None, model=args.model
                    ),
                    exc,
                )
            )
            continue

        for calibration in sorted(available):
            bundle_error_key = experiment_key(
                sid, calibration, "bundle", None, model=args.model
            )
            if bundle_error_key in existing:
                continue
            tasks.append(
                Experiment1Task(
                    sid=sid,
                    calibration=calibration,
                    config=config,
                    model=args.model,
                    tau_lookup=tau_lookup,
                    results_path=results_path,
                    existing_keys=_existing_keys_for_task(existing, sid, calibration),
                )
            )

    if tasks:
        effective_jobs = min(jobs, len(tasks))
        if effective_jobs > 1:
            print(
                f"Running {len(tasks)} sample/calibration tasks with "
                f"{effective_jobs} worker processes"
            )
            # Use spawn to avoid forking a process that may already have library
            # state initialized by tqdm/numpy and to mirror Experiment 3.
            ctx = mp.get_context("spawn")
            with futures.ProcessPoolExecutor(
                max_workers=effective_jobs,
                mp_context=ctx,
            ) as executor:
                future_to_task = {
                    executor.submit(_run_experiment_1_calibration_task, task): task
                    for task in tasks
                }
                iterator = futures.as_completed(future_to_task)
                for future in tqdm(
                    iterator,
                    total=len(future_to_task),
                    desc="Exp1 calibrations",
                ):
                    for record in future.result():
                        record_result(record)
        else:
            for task in tqdm(tasks, desc="Exp1 calibrations"):
                for record in _run_experiment_1_calibration_task(task):
                    record_result(record)

    aggregate = aggregate_experiment_1(iter_jsonl(results_path))
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
