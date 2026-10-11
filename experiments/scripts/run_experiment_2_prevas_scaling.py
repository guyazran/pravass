#!/usr/bin/env python3
"""Experiment 2: PRaVaSS exactness and scaling against exhaustive enumeration.

For every test instance, calibration scope and θ the runner records one
``exhaustive`` row, one ``prevas`` row and one ``random_theta_confident`` row
with the same fields as the paper-era ``sample_results.jsonl`` consumed by
``mbs_experiments.results_display`` (plus ``model`` for provenance).
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import multiprocessing as mp
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from mbs_experiments.baselines import exhaustive_mbs, random_theta_confident
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
)
from mbs_experiments.metrics import aggregate_experiment_2, jaccard
from mbs_experiments.normalizers import selected_groups_runtime_normalizer
from mbs_experiments.search import ranked_states
from mbs_experiments.states import (
    logical_full_state_count,
    null_indices,
    project_groups,
    visible_indices,
)

# Outputs are model-sharded like the Experiment 1/3 runners:
# experiments_data/experiment_2_prevas_scaling/<model>/{sample_results.jsonl,
# aggregate_metrics.json}. `model` is also stored in every record and is the
# last RESULT_KEY_FIELDS entry, so `key[:4]` prefixing by
# (domain, configuration, sample, calibration) keeps working.
OUTPUT_ROOT = Path("experiments_data/experiment_2_prevas_scaling")
DEFAULT_MODEL = "OpenAI/gpt-4.1-2025-04-14"
TEST_SPLIT = "test"
# θ operating points reported in the paper (Sec. 6 Exp. 2, App. J, Table 4).
DEFAULT_THETAS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
# Per-instance wall-clock budget used in the paper (App. J).
DEFAULT_METHOD_TIMEOUT = 10.0
RESULT_KEY_FIELDS = (
    "domain",
    "configuration",
    "sample",
    "calibration",
    "method",
    "theta",
    "model",
)


def experiment_paths(model: str) -> tuple[Path, Path, Path]:
    """Return (output_dir, results_path, agg_path) for a model."""
    base = OUTPUT_ROOT / model_to_path_segment(model)
    return base, base / "sample_results.jsonl", base / "aggregate_metrics.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 2 PRaVaSS exactness and scaling")
    parser.add_argument("--domains", nargs="+", default=("prb", "gw", "alf"))
    parser.add_argument("--configs", nargs="+")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model identifier whose predictions / Platt profiles to consume. "
            "Must match the --model passed to make_predictions.py and "
            f"calibrate_vlm.py. Default: {DEFAULT_MODEL} (the paper's backbone)."
        ),
    )
    parser.add_argument("--calibrations", nargs="+", default=("global", "lifted"))
    parser.add_argument(
        "--thetas",
        nargs="+",
        type=float,
        default=DEFAULT_THETAS,
        help=(
            "θ operating points for PRaVaSS, exhaustive and random baselines. "
            f"Default: {' '.join(str(t) for t in DEFAULT_THETAS)} (paper Table 4)."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--method-timeout",
        type=float,
        default=DEFAULT_METHOD_TIMEOUT,
        help=(
            "Wallclock timeout (seconds) applied to PRaVaSS, exhaustive "
            f"enumeration and the random baseline. Default: {DEFAULT_METHOD_TIMEOUT} "
            "(the paper's per-instance budget)."
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
    args = parser.parse_args(argv)
    thetas = tuple(dict.fromkeys(float(t) for t in args.thetas))
    if any(not 0.0 < t <= 1.0 for t in thetas):
        parser.error(f"--thetas must lie in (0, 1], got {args.thetas}")
    args.thetas = thetas
    return args


def peak_rss_mb() -> float:
    # ru_maxrss is bytes on macOS and kilobytes on Linux. This repo is developed
    # on macOS, but keep the value best-effort for aggregate runtime summaries.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def stable_seed(*parts: object) -> int:
    text = "|".join(map(str, parts))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) & 0xFFFFFFFF


@dataclass(frozen=True)
class Experiment2Config:
    method_timeout: float
    thetas: tuple[float, ...] = DEFAULT_THETAS


@dataclass(frozen=True)
class Experiment2Task:
    sid: SampleId
    calibration: str
    config: Experiment2Config
    model: str
    existing_records: dict[tuple[Any, ...], dict[str, Any]]


def experiment_key(
    sid: SampleId,
    calibration: str | None,
    method: str,
    theta: float | None,
    *,
    model: str,
) -> tuple[Any, ...]:
    # Order must match RESULT_KEY_FIELDS (model last).
    return (sid.domain, sid.configuration, sid.sample, calibration, method, theta, model)


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


def _worker_count(requested_jobs: int | None) -> int:
    default_jobs = max(1, (os.cpu_count() or 1) - 1)
    if requested_jobs is None or requested_jobs <= 0:
        return default_jobs
    return requested_jobs


def _run_experiment_2_calibration_task(task: Experiment2Task) -> list[dict[str, Any]]:
    """Run all Experiment 2 methods for one sample/calibration.

    Workers never write to the shared JSONL.  They return records to the parent
    process, which serializes writes and keeps resume/aggregate state coherent.
    """
    sid = task.sid
    calibration = task.calibration
    config = task.config
    model = task.model
    existing_records: dict[tuple[Any, ...], dict[str, Any]] = dict(task.existing_records)
    existing: set[tuple[Any, ...]] = set(existing_records)
    records: list[dict[str, Any]] = []

    def record_result(record: Mapping[str, Any]) -> None:
        payload = dict(record)
        key = result_key(payload, RESULT_KEY_FIELDS)
        if key in existing:
            return
        records.append(payload)
        existing_records[key] = payload
        existing.add(key)

    bundle_error_key = experiment_key(sid, calibration, "bundle", None, model=model)
    if bundle_error_key in existing:
        return records
    try:
        bundle = load_calibration_bundle(
            sid, calibration, model=model, split=TEST_SPLIT
        )
        full_z = selected_groups_runtime_normalizer(bundle, feature_space="full")
        search_idx = visible_indices(bundle.none_is_max)
        omitted_idx = null_indices(bundle.none_is_max)
        reduced_probs = tuple(bundle.probabilities[i] for i in search_idx)
        reduced_groups = project_groups(bundle.selected_groups, search_idx)
        reduced_z = selected_groups_runtime_normalizer(
            bundle, feature_space="visible"
        )
    except Exception as exc:
        record_result(
            make_error_record(
                _base_error_context(sid, calibration, "bundle", None, model=model),
                exc,
            )
        )
        return records

    feature_count = len(bundle.feature_names)
    search_feature_count = len(reduced_probs)
    common_metadata = {
        "model": model,
        "feature_count": feature_count,
        "search_feature_count": search_feature_count,
        "omitted_feature_count": len(omitted_idx),
        "normalizer": reduced_z,
        "full_normalizer": full_z,
        "search_selected_group_count": len(reduced_groups),
    }
    prevas_timeout_cache = None
    exhaustive_timeout_cache: tuple[Any, float] | None = None
    random_theta_timeout_cache: tuple[Any, float, int] | None = None
    for theta in config.thetas:
        exact_key = experiment_key(sid, calibration, "exhaustive", theta, model=model)
        prevas_key = experiment_key(sid, calibration, "prevas", theta, model=model)
        random_key = experiment_key(
            sid, calibration, "random_theta_confident", theta, model=model
        )
        exact_record = existing_records.get(exact_key)
        # Re-run prior runs that were skipped by the old feature cap.
        exact_stale = (
            exact_record is not None
            and exact_record.get("status") == "unsupported"
        )
        if (
            exact_key in existing
            and prevas_key in existing
            and random_key in existing
            and not exact_stale
        ):
            continue

        exact_state_set = None
        exact_runtime = exact_record.get("runtime_seconds") if exact_record is not None else None
        should_record_exact = exact_key not in existing or exact_stale
        # Exact may be recomputed for PRaVaSS/random comparison when an interrupted
        # run left the exhaustive row behind but not the comparison row.
        should_recompute_exact_for_prevas = prevas_key not in existing
        should_recompute_exact_for_random = random_key not in existing
        if (
            should_record_exact
            or should_recompute_exact_for_prevas
            or should_recompute_exact_for_random
        ):
            start = time.perf_counter()
            try:
                if exhaustive_timeout_cache is not None:
                    exact_state_set, cached_runtime = exhaustive_timeout_cache
                    exact_runtime_now = cached_runtime
                else:
                    exact_state_set = exhaustive_mbs(
                        reduced_probs,
                        reduced_groups,
                        theta,
                        reduced_z,
                        timeout_seconds=config.method_timeout,
                    )
                    exact_runtime_now = time.perf_counter() - start
                    if (
                        exact_state_set.status == "timeout"
                        and exact_state_set.constrained_mass < theta
                    ):
                        exhaustive_timeout_cache = (
                            exact_state_set,
                            exact_runtime_now,
                        )
                if should_record_exact:
                    exact_runtime = exact_runtime_now
                    record_result(
                        {
                            "status": exact_state_set.status,
                            "domain": sid.domain,
                            "configuration": sid.configuration,
                            "sample": sid.sample,
                            "split_relative_path": sid.split_relative_path,
                            "calibration": calibration,
                            "method": "exhaustive",
                            "theta": theta,
                            **common_metadata,
                            "set_size": logical_full_state_count(
                                len(exact_state_set.states), len(omitted_idx)
                            ),
                            "compact_set_size": len(exact_state_set.states),
                            "runtime_seconds": exact_runtime,
                            "peak_rss_mb": peak_rss_mb(),
                            "states_scored": exact_state_set.total_states_considered,
                        }
                    )
            except Exception as exc:
                if should_record_exact:
                    record_result(
                        make_error_record(
                            {
                                **_base_error_context(
                                    sid, calibration, "exhaustive", theta, model=model
                                ),
                                **common_metadata,
                                "runtime_seconds": time.perf_counter() - start,
                                "peak_rss_mb": peak_rss_mb(),
                            },
                            exc,
                        )
                    )

        if prevas_key not in existing or exact_stale:
            start = time.perf_counter()
            try:
                if prevas_timeout_cache is not None:
                    ours, runtime = prevas_timeout_cache
                else:
                    ours = ranked_states(
                        reduced_probs,
                        reduced_groups,
                        reduced_z,
                        theta=theta,
                        timeout_seconds=config.method_timeout,
                    )
                    runtime = time.perf_counter() - start
                    if (
                        ours.stats.timeout
                        and ours.state_set.constrained_mass < theta
                    ):
                        prevas_timeout_cache = (ours, runtime)
                exact_match = None
                jac = None
                if (
                    exact_state_set is not None
                    and exact_state_set.status == "ok"
                    and ours.state_set.status == "ok"
                ):
                    exact_match = set(ours.state_set.states) == set(exact_state_set.states)
                    jac = jaccard(ours.state_set.states, exact_state_set.states)
                record_result(
                    {
                        "status": ours.state_set.status,
                        "domain": sid.domain,
                        "configuration": sid.configuration,
                        "sample": sid.sample,
                        "split_relative_path": sid.split_relative_path,
                        "calibration": calibration,
                        "method": "prevas",
                        "theta": theta,
                        **common_metadata,
                        "set_size": logical_full_state_count(
                            len(ours.state_set.states), len(omitted_idx)
                        ),
                        "compact_set_size": len(ours.state_set.states),
                        "runtime_seconds": runtime,
                        "peak_rss_mb": peak_rss_mb(),
                        "states_scored": ours.stats.states_scored,
                        "states_popped": ours.stats.states_popped,
                        "timeout": ours.stats.timeout,
                        "exact_match": exact_match,
                        "jaccard": jac,
                        "exhaustive_runtime_seconds": exact_runtime,
                    }
                )
            except Exception as exc:
                record_result(
                    make_error_record(
                        {
                            **_base_error_context(
                                sid, calibration, "prevas", theta, model=model
                            ),
                            **common_metadata,
                            "runtime_seconds": time.perf_counter() - start,
                            "peak_rss_mb": peak_rss_mb(),
                        },
                        exc,
                    )
                )

        if random_key not in existing or exact_stale:
            random_seed = stable_seed(
                sid.split_relative_path, calibration, theta
            )
            start = time.perf_counter()
            try:
                if random_theta_timeout_cache is not None:
                    random_state_set, random_runtime, _ = random_theta_timeout_cache
                else:
                    random_state_set = random_theta_confident(
                        reduced_probs,
                        reduced_groups,
                        theta,
                        reduced_z,
                        seed=random_seed,
                        timeout_seconds=config.method_timeout,
                    )
                    random_runtime = time.perf_counter() - start
                    if (
                        random_state_set.status == "timeout"
                        and random_state_set.constrained_mass < theta
                    ):
                        random_theta_timeout_cache = (
                            random_state_set,
                            random_runtime,
                            random_seed,
                        )
                random_exact_match = None
                random_jac = None
                if (
                    exact_state_set is not None
                    and exact_state_set.status == "ok"
                    and random_state_set.status == "ok"
                ):
                    random_exact_match = (
                        set(random_state_set.states) == set(exact_state_set.states)
                    )
                    random_jac = jaccard(
                        random_state_set.states, exact_state_set.states
                    )
                record_result(
                    {
                        "status": random_state_set.status,
                        "domain": sid.domain,
                        "configuration": sid.configuration,
                        "sample": sid.sample,
                        "split_relative_path": sid.split_relative_path,
                        "calibration": calibration,
                        "method": "random_theta_confident",
                        "theta": theta,
                        **common_metadata,
                        "set_size": logical_full_state_count(
                            len(random_state_set.states), len(omitted_idx)
                        ),
                        "compact_set_size": len(random_state_set.states),
                        "runtime_seconds": random_runtime,
                        "peak_rss_mb": peak_rss_mb(),
                        "states_scored": random_state_set.total_states_considered,
                        "exact_match": random_exact_match,
                        "jaccard": random_jac,
                        "random_seed": random_seed,
                        "exhaustive_runtime_seconds": exact_runtime,
                    }
                )
            except Exception as exc:
                record_result(
                    make_error_record(
                        {
                            **_base_error_context(
                                sid,
                                calibration,
                                "random_theta_confident",
                                theta,
                                model=model,
                            ),
                            **common_metadata,
                            "runtime_seconds": time.perf_counter() - start,
                            "peak_rss_mb": peak_rss_mb(),
                            "random_seed": random_seed,
                        },
                        exc,
                    )
                )

    return records


def main() -> None:
    args = parse_args()
    jobs = _worker_count(args.jobs)
    output_dir, results_path, agg_path = experiment_paths(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and results_path.exists():
        results_path.unlink()
    domains = set(args.domains)
    configs = set(args.configs) if args.configs else None
    requested_calibrations = set(args.calibrations)
    existing_records: dict[tuple[Any, ...], dict[str, Any]] = {}
    if not args.overwrite and results_path.exists():
        for record in iter_jsonl(results_path):
            existing_records[result_key(record, RESULT_KEY_FIELDS)] = record
    existing: set[tuple[Any, ...]] = set(existing_records)

    def record_result(record: Mapping[str, Any]) -> None:
        payload = dict(record)
        key = result_key(payload, RESULT_KEY_FIELDS)
        if key in existing:
            return
        append_jsonl(results_path, payload)
        existing_records[key] = payload
        existing.add(key)

    sample_ids = filter_sample_ids(
        load_test_sample_ids(), domains=domains, configs=configs, limit=args.limit
    )
    config = Experiment2Config(method_timeout=args.method_timeout, thetas=args.thetas)
    tasks: list[Experiment2Task] = []

    for sid in tqdm(sample_ids, desc="Exp2 inputs"):
        prediction_error_key = experiment_key(
            sid, None, "prediction", None, model=args.model
        )
        if prediction_error_key in existing:
            continue
        try:
            # Touch the prediction file so a missing one is reported as a
            # prediction error rather than failing later inside the worker.
            load_json(prediction_path(sid, TEST_SPLIT, args.model))
            available = sorted(
                set(available_calibrations(args.model, sid.domain))
                & requested_calibrations
            )
        except Exception as exc:
            record_result(
                make_error_record(
                    _base_error_context(sid, None, "prediction", None, model=args.model),
                    exc,
                )
            )
            continue

        for calibration in available:
            tasks.append(
                Experiment2Task(
                    sid=sid,
                    calibration=calibration,
                    config=config,
                    model=args.model,
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
            ctx = mp.get_context("spawn")
            with futures.ProcessPoolExecutor(
                max_workers=effective_jobs,
                mp_context=ctx,
            ) as executor:
                future_to_task = {
                    executor.submit(_run_experiment_2_calibration_task, task): task
                    for task in tasks
                }
                iterator = futures.as_completed(future_to_task)
                for future in tqdm(
                    iterator,
                    total=len(future_to_task),
                    desc="Exp2 calibrations",
                ):
                    for record in future.result():
                        record_result(record)
        else:
            for task in tqdm(tasks, desc="Exp2 calibrations"):
                for record in _run_experiment_2_calibration_task(task):
                    record_result(record)

    records = existing_records.values() if existing_records else []
    aggregate = aggregate_experiment_2(records)
    aggregate["run_config"] = {
        "model": args.model,
        "split": TEST_SPLIT,
        "calibrations": sorted(requested_calibrations),
        "thetas": list(args.thetas),
        "method_timeout": args.method_timeout,
    }
    write_json_atomic(agg_path, aggregate)
    print(f"Wrote {results_path}")
    print(f"Wrote {agg_path}")


if __name__ == "__main__":
    main()
