#!/usr/bin/env python3
"""Experiment 7: correlated perception errors versus the conditional-independence lift."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
from collections import defaultdict
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

from mbs_experiments.baselines import map_baseline, mbs_prevas, valid_map
from mbs_experiments.data import (
    SampleId,
    load_calibration_bundle,
    load_test_sample_ids,
    model_to_path_segment,
)
from mbs_experiments.experiment7 import (
    GROUP_MODES,
    PROTOCOL_VERSION,
    build_corruption_model,
    component_normalizers,
    oracle_joint_prefix,
    paired_bootstrap_mean_ci,
    select_corrupted_indices,
    summarize_gap,
    sweep_points,
    total_variation,
    true_normalizer,
    true_state_mass,
)
from mbs_experiments.io import (
    append_jsonl,
    iter_jsonl,
    result_key,
    write_json_atomic,
    write_jsonl_atomic,
)
from mbs_experiments.metrics import contains_truth
from mbs_experiments.states import (
    State,
    StateSet,
    project_groups,
    project_state,
    selected_groups_normalizer,
    state_set_size,
    visible_indices,
)

DEFAULT_MODEL = "OpenAI/gpt-4.1-2025-04-14"
DEFAULT_OUTPUT_ROOT = Path("experiments_data/experiment_7_correlated_noise")
CONSTRAINT_MODE = "selected_groups"
NORMALIZER_MODE = "selected_groups_product"
BELIEF_METHOD = "belief"
METHODS = ("mbs_prevas", "oracle_joint", "valid_map", "map")
# Methods whose set is scored against the correlated joint.  MAP variants are singletons
# and carry no mass target, but their true mass is still the exact probability that the
# determinized state is the true one.
EXCLUSION_METHOD = "common_exclusion"

RESULT_KEY_FIELDS = (
    "protocol_version",
    "model",
    "domain",
    "configuration",
    "sample",
    "calibration",
    "method",
    "group_mode",
    "block_size",
    "rate",
    "strength",
    "theta",
    "corruption_seed",
    "constraint_mode",
    "normalizer_mode",
    "method_timeout",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--calibrations", nargs="+", default=("global",))
    parser.add_argument("--domains", nargs="+", default=("prb", "gw", "alf"))
    parser.add_argument("--configs", nargs="+")
    parser.add_argument(
        "--limit",
        type=int,
        default=150,
        help="Samples per domain, drawn with --sample-seed after a frozen shuffle.",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument(
        "--group-modes",
        nargs="+",
        choices=GROUP_MODES,
        default=("constraint", "uncertain"),
        help=(
            "How the jointly corrupted feature block is chosen. 'constraint' corrupts "
            "features the mutex structure already couples; 'uncertain' corrupts the "
            "least decided features, which the constraint layer cannot re-couple."
        ),
    )
    parser.add_argument("--block-size", type=int, default=6)
    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
        help="Nuisance-event probabilities pi. 0.0 is the uncorrelated control.",
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=(0.5, 1.0),
        help="Corruption pull rho; 1.0 mirrors each corrupted marginal.",
    )
    parser.add_argument("--thetas", nargs="+", type=float, default=(0.5, 0.9, 0.99))
    parser.add_argument(
        "--max-features",
        type=int,
        default=32,
        help="Skip samples with more visible features; recorded as a common exclusion.",
    )
    parser.add_argument("--method-timeout", type=float, default=20.0)
    parser.add_argument("--max-candidates", type=int, default=200000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--run-name", default="main")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--retry-status", nargs="+", default=("error", "timeout", "uncertified")
    )
    return parser.parse_args()


@dataclass(frozen=True)
class Experiment7Config:
    group_modes: tuple[str, ...]
    block_size: int
    points: tuple[tuple[float, float, float], ...]
    corruption_seed: int
    max_features: int
    method_timeout: float
    max_candidates: int


@dataclass(frozen=True)
class Experiment7Task:
    sid: SampleId
    calibration: str
    model: str
    config: Experiment7Config


def experiment_paths(
    output_root: Path, model: str, run_name: str
) -> tuple[Path, Path, Path]:
    output_dir = output_root / model_to_path_segment(model) / run_name
    return (
        output_dir,
        output_dir / "sample_results.jsonl",
        output_dir / "aggregate_metrics.json",
    )


def _sample_line(sid: SampleId) -> str:
    return sid.split_relative_path


def _fingerprint_lines(lines: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{line}\n" for line in lines).encode()
    ).hexdigest()[:16]


def select_samples(args: argparse.Namespace) -> list[SampleId]:
    """Freeze one population per domain: sort, shuffle by seed, take --limit."""
    by_domain: dict[str, list[SampleId]] = defaultdict(list)
    for sid in load_test_sample_ids():
        if sid.domain not in set(args.domains):
            continue
        if args.configs and sid.configuration not in set(args.configs):
            continue
        by_domain[sid.domain].append(sid)
    selected: list[SampleId] = []
    for domain in sorted(by_domain):
        pool = sorted(by_domain[domain], key=_sample_line)
        rng = random.Random(f"{args.sample_seed}:{domain}")
        rng.shuffle(pool)
        selected.extend(pool if args.limit is None else pool[: args.limit])
    return sorted(selected, key=_sample_line)


def _semantic_context(
    task: Experiment7Task,
    *,
    method: str,
    group_mode: str | None,
    rate: float | None,
    strength: float | None,
    theta: float | None,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model": task.model,
        "domain": task.sid.domain,
        "configuration": task.sid.configuration,
        "sample": task.sid.sample,
        "calibration": task.calibration,
        "method": method,
        "group_mode": group_mode,
        "block_size": task.config.block_size,
        "rate": rate,
        "strength": strength,
        "theta": theta,
        "corruption_seed": task.config.corruption_seed,
        "constraint_mode": CONSTRAINT_MODE,
        "normalizer_mode": NORMALIZER_MODE,
        "method_timeout": task.config.method_timeout,
        "sample_path": _sample_line(task.sid),
    }


def _exclusion(
    task: Experiment7Task,
    reason: str,
    *,
    group_mode: str | None = None,
    rate: float | None = None,
    strength: float | None = None,
    theta: float | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    record = _semantic_context(
        task,
        method=EXCLUSION_METHOD,
        group_mode=group_mode,
        rate=rate,
        strength=strength,
        theta=theta,
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


def run_experiment_7_task(task: Experiment7Task) -> list[dict[str, Any]]:
    """Score every method against the correlated joint for one sample."""
    config = task.config
    try:
        bundle = load_calibration_bundle(
            task.sid, task.calibration, model=task.model, split="test"
        )
    except Exception as exc:  # noqa: BLE001 - task-level exclusion boundary
        return [_exclusion(task, "bundle_load_failed", error=exc)]

    search_idx = visible_indices(bundle.none_is_max)
    probabilities = tuple(bundle.probabilities[i] for i in search_idx)
    groups = project_groups(bundle.selected_groups, search_idx)
    truth = project_state(bundle.ground_truth, search_idx)
    feature_count = len(probabilities)
    if feature_count == 0:
        return [_exclusion(task, "no_visible_features")]
    if feature_count > config.max_features:
        return [_exclusion(task, "too_many_features")]

    records: list[dict[str, Any]] = []
    for group_mode in config.group_modes:
        block = select_corrupted_indices(
            probabilities,
            groups,
            mode=group_mode,
            size=config.block_size,
            seed=config.corruption_seed,
        )
        for rate, strength, theta in config.points:
            context = {
                "group_mode": group_mode,
                "rate": rate,
                "strength": strength,
                "theta": theta,
            }
            try:
                model = build_corruption_model(
                    probabilities, block, rate=rate, strength=strength
                )
                product_z = selected_groups_normalizer(model.marginals, groups)
                z_clean, z_corrupt = component_normalizers(model, groups)
                joint_z = true_normalizer(model, groups)
                if product_z <= 0.0 or joint_z <= 0.0:
                    records.append(
                        _exclusion(task, "degenerate_normalizer", **context)
                    )
                    continue
                variation = total_variation(model, groups)
            except Exception as exc:  # noqa: BLE001 - condition-level exclusion
                records.append(
                    _exclusion(task, "corruption_model_failed", error=exc, **context)
                )
                continue

            belief = _semantic_context(task, method=BELIEF_METHOD, **context)
            belief.update(
                {
                    "status": "ok",
                    "feature_count": feature_count,
                    "selected_group_count": len(groups),
                    "corrupted_indices": list(block),
                    "corrupted_block_size": len(block),
                    "product_normalizer": product_z,
                    "clean_normalizer": z_clean,
                    "corrupt_normalizer": z_corrupt,
                    "joint_normalizer": joint_z,
                    "total_variation": variation,
                }
            )

            method_rows: list[dict[str, Any]] = []
            failed = False
            for method in METHODS:
                row = _semantic_context(task, method=method, **context)
                try:
                    if method == "mbs_prevas":
                        result = mbs_prevas(
                            model.marginals,
                            groups,
                            theta,
                            product_z,
                            timeout_seconds=config.method_timeout,
                        )
                        state_set = result.state_set
                        runtime = result.stats.runtime_seconds
                        certified = None
                        status = state_set.status
                        attributed = state_set.constrained_mass
                    elif method == "oracle_joint":
                        prefix = oracle_joint_prefix(
                            model,
                            groups,
                            theta,
                            max_candidates=config.max_candidates,
                            timeout_seconds=config.method_timeout,
                        )
                        state_set = _state_set_from_states(prefix.states)
                        runtime = prefix.runtime_seconds
                        certified = prefix.certified
                        status = "ok" if prefix.certified else "uncertified"
                        # The oracle reports its own mass under the joint it optimizes.
                        attributed = prefix.mass
                        row["candidates_examined"] = prefix.candidates_examined
                    else:
                        state_set = (
                            valid_map(model.marginals, groups, product_z)
                            if method == "valid_map"
                            else map_baseline(model.marginals)
                        )
                        runtime = 0.0
                        certified = None
                        status = state_set.status
                        attributed = state_set.constrained_mass
                except Exception as exc:  # noqa: BLE001 - method failure excludes the cell
                    records.append(
                        _exclusion(task, f"{method}_failed", error=exc, **context)
                    )
                    failed = True
                    break

                if status not in {"ok", "uncertified"}:
                    records.append(_exclusion(task, f"{method}_{status}", **context))
                    failed = True
                    break

                true_mass = true_state_mass(state_set.states, model, groups)
                row.update(
                    {
                        "status": status,
                        "set_size": state_set_size(state_set),
                        "runtime_seconds": runtime,
                        "certified": certified,
                        "contains_truth": contains_truth(state_set, truth),
                        "total_variation": variation,
                        "feature_count": feature_count,
                    }
                )
                row.update(summarize_gap(attributed, true_mass, theta))
                method_rows.append(row)

            if failed:
                continue
            records.append(belief)
            records.extend(method_rows)
    return records


def _state_set_from_states(states: Sequence[State]) -> StateSet:
    """Wrap the oracle prefix so size and containment use the shared StateSet helpers."""
    return StateSet(
        states=tuple(states),
        unconstrained_mass=0.0,
        constrained_mass=0.0,
        total_states_considered=len(states),
    )


def _load_latest(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in iter_jsonl(path):
        latest[result_key(record, RESULT_KEY_FIELDS)] = record
    return latest


def _cell(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Identity of one paired comparison cell, shared by every method."""
    return (
        record.get("domain"),
        record.get("configuration"),
        record.get("sample"),
        record.get("calibration"),
        record.get("group_mode"),
        record.get("rate"),
        record.get("strength"),
        record.get("theta"),
    )


def aggregate_experiment_7(
    records: Iterable[Mapping[str, Any]],
    *,
    fingerprint: str,
    bootstrap_seed: int = 0,
    bootstrap_repetitions: int = 1000,
) -> dict[str, Any]:
    """Aggregate on a common denominator: a cell counts only if every method scored it.

    Any excluded, timed-out, or uncertified cell removes that cell for *all* methods, so
    every reported contrast is paired across exactly the same conditions.
    """
    rows = [dict(record) for record in records]
    excluded: set[tuple[Any, ...]] = {
        _cell(row) for row in rows if row.get("method") == EXCLUSION_METHOD
    }
    # An uncertified oracle prefix is not an exact comparator, so drop its whole cell.
    excluded |= {
        _cell(row)
        for row in rows
        if row.get("method") == "oracle_joint" and not row.get("certified")
    }

    by_cell: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    belief_by_cell: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        method = row.get("method")
        if method == EXCLUSION_METHOD:
            continue
        cell = _cell(row)
        if cell in excluded:
            continue
        if method == BELIEF_METHOD:
            belief_by_cell[cell] = row
        else:
            by_cell[cell][str(method)] = row

    complete = {
        cell: methods
        for cell, methods in by_cell.items()
        if set(METHODS) <= set(methods) and cell in belief_by_cell
    }

    def sample_key(cell: tuple[Any, ...]) -> str:
        return f"{cell[0]}/{cell[1]}/{cell[2]}"

    sweep: dict[str, Any] = {}
    grouped: dict[tuple[Any, ...], list[tuple[Any, ...]]] = defaultdict(list)
    for cell in complete:
        grouped[(cell[0], cell[4], cell[5], cell[6], cell[7])].append(cell)

    for key, cells in sorted(grouped.items(), key=lambda item: [str(p) for p in item[0]]):
        domain, group_mode, rate, strength, theta = key
        gaps: dict[str, float] = {}
        shortfalls: dict[str, float] = {}
        variations: list[float] = []
        meets: list[float] = []
        size_ratio: list[float] = []
        sizes: dict[str, list[float]] = {method: [] for method in METHODS}
        coverage: dict[str, list[float]] = {method: [] for method in METHODS}
        true_masses: dict[str, list[float]] = {method: [] for method in METHODS}
        for cell in cells:
            methods = complete[cell]
            mbs = methods["mbs_prevas"]
            oracle = methods["oracle_joint"]
            gaps[sample_key(cell)] = float(mbs["mass_coverage_gap"])
            shortfalls[sample_key(cell)] = float(mbs["coverage_shortfall"])
            meets.append(1.0 if mbs["meets_theta"] else 0.0)
            variation = belief_by_cell[cell].get("total_variation")
            if variation is not None:
                variations.append(float(variation))
            oracle_size = float(oracle["set_size"])
            if oracle_size > 0:
                size_ratio.append(float(mbs["set_size"]) / oracle_size)
            for method in METHODS:
                sizes[method].append(float(methods[method]["set_size"]))
                coverage[method].append(
                    1.0 if methods[method]["contains_truth"] else 0.0
                )
                true_masses[method].append(float(methods[method]["true_mass"]))
        sweep[f"{domain}:{group_mode}:{rate}:{strength}:{theta}"] = {
            "domain": domain,
            "group_mode": group_mode,
            "rate": rate,
            "strength": strength,
            "theta": theta,
            "n": len(cells),
            "mass_coverage_gap": paired_bootstrap_mean_ci(
                gaps, seed=bootstrap_seed, repetitions=bootstrap_repetitions
            ),
            "coverage_shortfall": paired_bootstrap_mean_ci(
                shortfalls, seed=bootstrap_seed, repetitions=bootstrap_repetitions
            ),
            "meets_theta_rate": mean(meets) if meets else None,
            "total_variation_mean": mean(variations) if variations else None,
            "total_variation_median": median(variations) if variations else None,
            "size_ratio_median": median(size_ratio) if size_ratio else None,
            "method_size_median": {
                method: median(values) if values else None
                for method, values in sizes.items()
            },
            "method_truth_coverage": {
                method: mean(values) if values else None
                for method, values in coverage.items()
            },
            "method_true_mass_mean": {
                method: mean(values) if values else None
                for method, values in true_masses.items()
            },
        }

    oracle_rows = [row for row in rows if row.get("method") == "oracle_joint"]
    certified = [row for row in oracle_rows if row.get("certified")]
    exclusion_reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("method") == EXCLUSION_METHOD:
            exclusion_reasons[str(row.get("exclusion_reason"))] += 1

    certification_rate = len(certified) / len(oracle_rows) if oracle_rows else None
    decision = {
        "decision": "analysis-ready" if complete else "insufficient-data",
        "complete_cells": len(complete),
        "excluded_cells": len(excluded),
        "oracle_certification_rate": certification_rate,
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "sample_list_fingerprint": fingerprint,
    }
    return {
        "analysis_decision": decision,
        "complete_cells": len(complete),
        "sweep": sweep,
        "samples": len({sample_key(cell) for cell in complete}),
    }


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    os.environ.setdefault("MONO_CRASH_NOFILE", "1")

    output_dir, results_path, aggregate_path = experiment_paths(
        args.output_root, args.model, args.run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and results_path.exists():
        results_path.unlink()

    selected = select_samples(args)
    fingerprint = _fingerprint_lines([_sample_line(sid) for sid in selected])
    (output_dir / "selected_samples.txt").write_text(
        "".join(f"{_sample_line(sid)}\n" for sid in selected)
    )

    points = sweep_points(args.rates, args.strengths, args.thetas)
    config = Experiment7Config(
        group_modes=tuple(args.group_modes),
        block_size=args.block_size,
        points=points,
        corruption_seed=args.corruption_seed,
        max_features=args.max_features,
        method_timeout=args.method_timeout,
        max_candidates=args.max_candidates,
    )
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "model": args.model,
        "calibrations": list(args.calibrations),
        "domains": list(args.domains),
        "methods": list(METHODS),
        "group_modes": list(config.group_modes),
        "block_size": config.block_size,
        "rates": [float(rate) for rate in args.rates],
        "strengths": [float(strength) for strength in args.strengths],
        "thetas": [float(theta) for theta in args.thetas],
        "sweep_points": len(points),
        "max_features": config.max_features,
        "method_timeout": config.method_timeout,
        "constraint_mode": CONSTRAINT_MODE,
        "normalizer_mode": NORMALIZER_MODE,
        "sample_limit_per_domain": args.limit,
        "sample_seed": args.sample_seed,
        "corruption_seed": args.corruption_seed,
        "selected_samples": len(selected),
        "sample_list_fingerprint": fingerprint,
        "run_name": args.run_name,
    }
    write_json_atomic(output_dir / "run_config.json", metadata)

    tasks = [
        Experiment7Task(sid, calibration, args.model, config)
        for sid in selected
        for calibration in args.calibrations
    ]

    latest = _load_latest(results_path)
    retry = set(args.retry_status)

    def task_needs_run(task: Experiment7Task) -> bool:
        for group_mode in config.group_modes:
            for rate, strength, theta in config.points:
                context = {
                    "group_mode": group_mode,
                    "rate": rate,
                    "strength": strength,
                    "theta": theta,
                }
                exclusion_key = result_key(
                    _semantic_context(task, method=EXCLUSION_METHOD, **context),
                    RESULT_KEY_FIELDS,
                )
                sample_exclusion_key = result_key(
                    _semantic_context(
                        task,
                        method=EXCLUSION_METHOD,
                        group_mode=None,
                        rate=None,
                        strength=None,
                        theta=None,
                    ),
                    RESULT_KEY_FIELDS,
                )
                for key in (exclusion_key, sample_exclusion_key):
                    row = latest.get(key)
                    if row is not None and str(row.get("status")) not in retry:
                        break
                else:
                    for method in (BELIEF_METHOD,) + METHODS:
                        key = result_key(
                            _semantic_context(task, method=method, **context),
                            RESULT_KEY_FIELDS,
                        )
                        row = latest.get(key)
                        if row is None or str(row.get("status")) in retry:
                            return True
        return False

    pending = [task for task in tasks if task_needs_run(task)]

    def record_all(rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            payload = dict(row)
            append_jsonl(results_path, payload)
            latest[result_key(payload, RESULT_KEY_FIELDS)] = payload

    jobs = min(max(1, args.jobs), max(1, len(pending)))
    if jobs > 1 and pending:
        ctx = mp.get_context("spawn")
        with futures.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as executor:
            submitted = {
                executor.submit(run_experiment_7_task, task): task for task in pending
            }
            for future in tqdm(
                futures.as_completed(submitted),
                total=len(submitted),
                desc="Exp7 samples",
            ):
                record_all(future.result())
    else:
        for task in tqdm(pending, desc="Exp7 samples"):
            record_all(run_experiment_7_task(task))

    # Compact duplicate retry rows only after all parent-process writes finish.
    write_jsonl_atomic(results_path, latest.values())
    aggregate = aggregate_experiment_7(
        latest.values(),
        fingerprint=fingerprint,
        bootstrap_seed=args.sample_seed,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    aggregate["config"] = metadata
    write_json_atomic(aggregate_path, aggregate)
    write_json_atomic(
        output_dir / "analysis_decision.json", aggregate["analysis_decision"]
    )
    print(f"Wrote {results_path}")
    print(f"Wrote {aggregate_path}")
    print(json.dumps(aggregate["analysis_decision"], indent=2))


if __name__ == "__main__":
    main()
