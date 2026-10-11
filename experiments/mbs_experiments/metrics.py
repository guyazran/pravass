from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from .states import State, StateSet, state_set_contains

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CI_LOWER = 0.025
BOOTSTRAP_CI_UPPER = 0.975
BOOTSTRAP_SEED_SALT = "mbs-experiments-bootstrap"


def _bootstrap_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED_SALT}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_ci(
    values: Sequence[float],
    *,
    label: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    
    if n == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": 1}
    
    rng = np.random.default_rng(_bootstrap_seed(label))
    indices = rng.integers(0, n, size=(resamples, n))
    means = np.mean(arr[indices], axis=1)
    means.sort()
    
    low_idx = max(0, min(len(means) - 1, int(BOOTSTRAP_CI_LOWER * len(means))))
    high_idx = max(0, min(len(means) - 1, int(BOOTSTRAP_CI_UPPER * len(means)) - 1))
    
    return {
        "mean": mean,
        "ci_low": float(means[low_idx]),
        "ci_high": float(means[high_idx]),
        "n": n,
    }


def contains_truth(state_set: StateSet, truth: State) -> bool:
    return state_set_contains(state_set, truth)


def safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, pct * 100))


def jaccard(states_a: Sequence[State], states_b: Sequence[State]) -> float:
    a = set(states_a)
    b = set(states_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def aggregate_experiment_1(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    theta_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("status") != "ok":
            continue
        key = (record.get("domain"), record.get("calibration"), record.get("method"))
        groups[key].append(record)
        theta_key = (
            record.get("domain"),
            record.get("calibration"),
            record.get("method"),
            record.get("theta"),
        )
        theta_groups[theta_key].append(record)

    by_method: list[dict[str, Any]] = []
    for key, items in tqdm(sorted(groups.items()), desc="Exp 1 (by_method)"):
        domain, calibration, method = key
        coverage_values = [1.0 if r.get("contains_truth") else 0.0 for r in items]
        sizes = [
            float(r.get("full_set_size", r.get("compact_set_size", r.get("set_size", 0))))
            for r in items
        ]
        successful_sizes = [
            float(r.get("full_set_size", r.get("compact_set_size", r.get("set_size", 0))))
            for r in items
            if r.get("contains_truth")
        ]
        invalid_masses = [float(r.get("invalid_mass", 0.0)) for r in items]
        label = f"exp1|{domain}|{calibration}|{method}"
        by_method.append(
            {
                "domain": domain,
                "calibration": calibration,
                "method": method,
                "n": len(items),
                "coverage": safe_mean(coverage_values),
                "coverage_ci": bootstrap_ci(coverage_values, label=f"{label}|coverage"),
                "average_size": safe_mean(sizes),
                "average_size_ci": bootstrap_ci(sizes, label=f"{label}|size"),
                "successful_average_size": safe_mean(successful_sizes),
                "successful_average_size_ci": bootstrap_ci(
                    successful_sizes, label=f"{label}|successful_size"
                ),
                "invalid_mass_mean": safe_mean(invalid_masses),
                "invalid_mass_ci": bootstrap_ci(
                    invalid_masses, label=f"{label}|invalid_mass"
                ),
            }
        )

    by_theta: list[dict[str, Any]] = []
    for key, items in tqdm(sorted(theta_groups.items()), desc="Exp 1 (by_theta)"):
        domain, calibration, method, theta = key
        coverage_values = [1.0 if r.get("contains_truth") else 0.0 for r in items]
        cov = safe_mean(coverage_values)
        label = f"exp1|{domain}|{calibration}|{method}|theta={theta}"
        by_theta.append(
            {
                "domain": domain,
                "calibration": calibration,
                "method": method,
                "theta": theta,
                "n": len(items),
                "coverage": cov,
                "coverage_ci": bootstrap_ci(coverage_values, label=label),
                "abs_coverage_error": None
                if cov is None or theta is None
                else abs(float(cov) - float(theta)),
            }
        )
    return {"by_method": by_method, "by_theta": by_theta}


def aggregate_experiment_2(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = [r for r in records if r.get("status") in {"ok", "timeout", "unsupported"}]
    by_method_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in items:
        key = (
            record.get("domain"),
            record.get("configuration"),
            record.get("calibration"),
            record.get("method"),
        )
        by_method_groups[key].append(record)
    summaries: list[dict[str, Any]] = []
    for key, group in tqdm(sorted(by_method_groups.items()), desc="Exp 2"):
        runtimes = [
            float(r["runtime_seconds"])
            for r in group
            if r.get("runtime_seconds") is not None
        ]
        timeout_indicators = [
            1.0 if r.get("status") == "timeout" else 0.0 for r in group
        ]
        label = f"exp2|{key[0]}|{key[1]}|{key[2]}|{key[3]}"
        summaries.append(
            {
                "domain": key[0],
                "configuration": key[1],
                "calibration": key[2],
                "method": key[3],
                "n": len(group),
                "timeout_rate": safe_mean(timeout_indicators),
                "timeout_rate_ci": bootstrap_ci(
                    timeout_indicators, label=f"{label}|timeout"
                ),
                "median_runtime": None if not runtimes else statistics.median(runtimes),
                "p95_runtime": percentile(runtimes, 0.95),
                "mean_runtime_ci": bootstrap_ci(runtimes, label=f"{label}|runtime"),
            }
        )
    return {"by_method": summaries}


def aggregate_experiment_3(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("status") not in {"ok", "timeout", "no_plan", "error"}:
            continue
        key = (record.get("domain"), record.get("calibration"), record.get("method"), record.get("theta"))
        groups[key].append(record)
    summaries: list[dict[str, Any]] = []
    for key, items in tqdm(sorted(groups.items(), key=lambda item: (item[0][0] or "", item[0][1] or "", item[0][2] or "", item[0][3] if item[0][3] is not None else -1.0)), desc="Exp 3"):
        success = [1.0 if r.get("success") else 0.0 for r in items]
        unsafe = [1.0 if r.get("unsafe_first") else 0.0 for r in items]
        pre = [1.0 if r.get("pre_fail") else 0.0 for r in items]
        nopf = [1.0 if r.get("no_plan_fn") else 0.0 for r in items]
        label = f"exp3|{key[0]}|{key[1]}|{key[2]}|{key[3]}"
        summaries.append(
            {
                "domain": key[0],
                "calibration": key[1],
                "method": key[2],
                "theta": key[3],
                "n": len(items),
                "success_rate": safe_mean(success),
                "success_rate_ci": bootstrap_ci(success, label=f"{label}|success"),
                "unsafe_first_rate": safe_mean(unsafe),
                "unsafe_first_rate_ci": bootstrap_ci(unsafe, label=f"{label}|unsafe"),
                "pre_fail_rate": safe_mean(pre),
                "pre_fail_rate_ci": bootstrap_ci(pre, label=f"{label}|pre_fail"),
                "no_plan_fn_rate": safe_mean(nopf),
                "no_plan_fn_rate_ci": bootstrap_ci(nopf, label=f"{label}|no_plan_fn"),
            }
        )
    return {"by_method": summaries}
