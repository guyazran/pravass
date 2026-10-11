"""Multi-model data loading, caching, and display for ``results.ipynb``.

This module is the data backbone of the results notebook for the *extended*
study (the one in ``paper.pdf`` plus the new multi-model baselines).  The raw
Experiment-1 ``sample_results.jsonl`` files are several GB *per model* and
cannot be loaded into a notebook directly, so the strategy is:

1.  Stream each raw ``sample_results.jsonl`` **once** and reduce it to a small
    tidy summary frame keyed by ``(model, domain, calibration, method, theta)``
    with coverage / set-size / mass means + bootstrap CIs.
2.  Persist that summary next to the raw data as a pickle, keyed by the source
    file's size+mtime, so subsequent notebook runs load instantly and the user
    can flip model / calibration / method parameters cheaply.

Everything the notebook plots is derived from these summary frames, so changing
the model, calibration scope, or method set on a figure is a fast filter rather
than a re-read of the raw data.

Data layout produced by the experiment runners (model-sharded)::

    experiments_data/experiment_1_static_quality/<model>/sample_results.jsonl
    experiments_data/experiment_3_planning_utility/<model>/sample_results.jsonl

``<model>`` is ``model_to_path_segment(model)`` (``/`` -> ``__``).

Methods
-------
Experiment 1 rows are **θ-grid** rows (``theta`` set): ``mbs_prevas``, ``map``,
``valid_map``, ``mbs_first``, ``mbs_first_constraints``,
``unconstrained_top_mass``, ``random_theta_confident``,
``fluent_abstention_unconstrained``.
"""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
EXP1_ROOT = ROOT / "experiments_data" / "experiment_1_static_quality"
EXP3_ROOT = ROOT / "experiments_data" / "experiment_3_planning_utility"
EXP6_CONTROLLED_PRIOR_ROOT = (
    ROOT / "experiments_data" / "experiment_6_controlled_prior"
)
EXP6_UNIFORM_HIDDEN_RUN = "uniform_hidden_main_n120_adaptive_v3"
EXP7_CORRELATED_NOISE_ROOT = (
    ROOT / "experiments_data" / "experiment_7_correlated_noise"
)
EXP7_CORRELATED_NOISE_RUN = "main"

# Cache schema version -- bump to invalidate every cached pickle at once.
CACHE_VERSION = 15
_EXP1_CACHE_NAME = f"notebook_exp1_summary_v{CACHE_VERSION}.pkl"
_EXP3_CACHE_NAME = f"notebook_exp3_summary_v{CACHE_VERSION}.pkl"
_EXP3_RECORDS_CACHE_NAME = f"notebook_exp3_records_v{CACHE_VERSION}.pkl"

# ---------------------------------------------------------------------------
# Method / model vocabulary (shared by tables and figures)
# ---------------------------------------------------------------------------

# Experiment 1 -- θ-grid methods (curves swept over θ).
EXP1_THETA_METHODS = (
    "mbs_prevas",
    "random_theta_confident",
    "fluent_abstention_unconstrained",
    "unconstrained_top_mass",
)
# Experiment 1 -- point methods (single θ-independent value; drawn as a line).
EXP1_POINT_METHODS = ("map", "valid_map", "mbs_first", "mbs_first_constraints")

EXP1_METHOD_LABELS: dict[str, str] = {
    "mbs_prevas": "MBSθ (Ours)",
    "random_theta_confident": "Random θ-Mass",
    "fluent_abstention_unconstrained": "Feature Abstention",
    "unconstrained_top_mass": "MBS (unconstrained)",
    "map": "MAP",
    "valid_map": "Valid MAP",
    "mbs_first": "MBS-First",
    "mbs_first_constraints": "MBS-First (constraints)",
}

# Experiment 3 -- planning methods.
EXP3_METHOD_LABELS: dict[str, str] = {
    "oracle": "Oracle",
    "map": "MAP",
    # Planning uses two constraint sets (paper App. K): "-Loose" variants use the
    # disjoint selected groups; the others use all extracted mutex groups with the
    # exact normalizer (the primary normalizer of App. G.5) or, as an ablation, the
    # closed-form relaxed one.
    "mbs_prevas": "MBSθ-Loose (Ours)",
    "mbs_prevas_constraints_exact": "MBSθ (Ours)",
    "mbs_prevas_constraints_relaxed": "MBSθ (Ours, relaxed Z)",
    "mbs_prevas_prefix_search": "AMS-Loose (Ours)",
    "mbs_prevas_prefix_search_constraints_exact": "AMS (Ours)",
    "mbs_prevas_prefix_search_constraints_relaxed": "AMS (Ours, relaxed Z)",
    "adaptive_fluent_abstention": "Adaptive Abstention",
    "random_theta": "Random θ-Mass",
    "fluent_abstention_unconstrained": "Feature Abstention",
    "unconstrained_top_mass": "MBS (unconstrained)",
}


EXP6_METHOD_LABELS: dict[str, str] = {
    "map": "MAP",
    "adaptive_abstention": "Adaptive abstention",
    "ams": "AMS",
    "oracle": "Oracle",
}

EXP7_METHOD_LABELS: dict[str, str] = {
    "mbs_prevas": "MBSθ (product belief)",
    "oracle_joint": "MBSθ (true joint)",
    "valid_map": "Valid MAP",
    "map": "MAP",
}

EXP7_GROUP_MODE_LABELS: dict[str, str] = {
    "constraint": "Mutex-coupled block",
    "uncertain": "Least-decided block",
}

# Experiment 3 display operating points.  Plain θ-grid methods are shown at the
# complete low-θ probe by default; prefix-search methods are only recorded at
# their own θ=0.99 operating point.
EXP3_PLAIN_OPERATING_THETA = 0.01
EXP3_PREFIX_OPERATING_THETA = 0.99
EXP3_METHOD_OPERATING_THETAS: dict[str, float] = {
    "mbs_prevas": EXP3_PLAIN_OPERATING_THETA,
    "mbs_prevas_constraints_exact": EXP3_PLAIN_OPERATING_THETA,
    "mbs_prevas_constraints_relaxed": EXP3_PLAIN_OPERATING_THETA,
    "random_theta": EXP3_PLAIN_OPERATING_THETA,
    "fluent_abstention_unconstrained": EXP3_PLAIN_OPERATING_THETA,
    "unconstrained_top_mass": EXP3_PLAIN_OPERATING_THETA,
    "mbs_prevas_prefix_search": EXP3_PREFIX_OPERATING_THETA,
    "mbs_prevas_prefix_search_constraints_exact": EXP3_PREFIX_OPERATING_THETA,
    "mbs_prevas_prefix_search_constraints_relaxed": EXP3_PREFIX_OPERATING_THETA,
}

PREFERRED_DOMAINS = ("prb", "gw", "alf")


def model_label(model_segment: str) -> str:
    """Human-friendly model name from a path segment (``a__b`` -> ``a/b``)."""
    return model_segment.replace("__", "/")


def method_label(method_id: str, *, experiment: int = 1) -> str:
    table = EXP1_METHOD_LABELS if experiment == 1 else EXP3_METHOD_LABELS
    return table.get(method_id, method_id)


def _refresh_method_labels(df: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """Return a copy whose display labels reflect the current label table.

    Cached summaries persist ``Method Label`` for speed, but label-only edits
    should not require re-streaming multi-GB raw result files.
    """
    out = df.copy()
    if "Method" in out.columns and "Method Label" in out.columns:
        out["Method Label"] = out["Method"].map(labels).fillna(out["Method Label"])
    return out


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_models(root: Path) -> list[str]:
    """Model path-segments that have a non-empty ``sample_results.jsonl``."""
    if not root.exists():
        return []
    out = []
    for child in sorted(root.iterdir()):
        results = child / "sample_results.jsonl"
        if child.is_dir() and results.exists() and results.stat().st_size > 0:
            out.append(child.name)
    return out


def exp1_models() -> list[str]:
    return discover_models(EXP1_ROOT)


def exp3_models() -> list[str]:
    return discover_models(EXP3_ROOT)


def order_domains(domains: Iterable[Any]) -> list[Any]:
    values = list(dict.fromkeys(domains))
    head = [d for d in PREFERRED_DOMAINS if d in values]
    return head + [d for d in values if d not in head]


# ---------------------------------------------------------------------------
# Bootstrap (vectorized, deterministic per label)
# ---------------------------------------------------------------------------


def _bootstrap_mean_ci(
    values: np.ndarray, *, label: str, resamples: int
) -> tuple[float | None, float | None, float | None]:
    if values.size == 0:
        return None, None, None
    mean = float(values.mean())
    if values.size == 1 or resamples <= 0:
        return mean, mean, mean
    seed = int.from_bytes(
        __import__("hashlib").sha256(label.encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[idx].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high)


# ---------------------------------------------------------------------------
# Experiment 1 streaming aggregation
# ---------------------------------------------------------------------------


# θ-grid / point methods compared on a *shared* common support: the instances
# every one of them solves (status "ok") at *every* θ. Averaging coverage/size
# over this identical, θ-independent population makes each curve monotone in θ
# (per-instance the set only grows with θ) and the methods directly comparable —
# matching the paper's `exp1_coverage_size_summary`. Other θ-methods (random,
# unconstrained, …) get their own over-θ support.
EXP1_SHARED_SUPPORT_METHODS = ("mbs_prevas", "fluent_abstention_unconstrained", "map")


def _full_size(rec: dict) -> float:
    """Downstream full-feature-space belief size (Experiment-3 semantics)."""
    for key in ("full_set_size", "compact_set_size", "set_size"):
        v = rec.get(key)
        if v is not None:
            return float(v)
    return 0.0


def _compact_size(rec: dict) -> float:
    """Enumerated belief-set size in the search space (the size the paper plots)."""
    for key in ("compact_set_size", "set_size", "full_set_size"):
        v = rec.get(key)
        if v is not None:
            return float(v)
    return 0.0


def _opt_float(v: Any) -> float:
    return float(v) if v is not None else np.nan


_EXP1_GROUP_FIELDS = (
    "inst", "cov", "compact", "full", "mass", "invalid", "runtime",
)


def _aggregate_exp1_file(
    path: Path, model_segment: str, *, resamples: int
) -> pd.DataFrame:
    groups: dict[tuple, dict[str, list]] = defaultdict(
        lambda: {f: [] for f in _EXP1_GROUP_FIELDS}
    )
    n_total: dict[tuple, int] = defaultdict(int)
    interns: dict[str, int] = {}

    def iid(p: str) -> int:
        i = interns.get(p)
        if i is None:
            i = len(interns)
            interns[p] = i
        return i

    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            method = rec.get("method")
            if method not in EXP1_METHOD_LABELS:
                continue
            key = (
                rec.get("domain"),
                rec.get("calibration"),
                method,
                rec.get("theta"),
            )
            n_total[key] += 1
            if rec.get("status") != "ok":
                continue
            g = groups[key]
            g["inst"].append(iid(rec.get("split_relative_path")))
            g["cov"].append(1.0 if rec.get("contains_truth") else 0.0)
            g["compact"].append(_compact_size(rec))
            g["full"].append(_full_size(rec))
            g["mass"].append(_opt_float(rec.get("constrained_mass")))
            g["invalid"].append(_opt_float(rec.get("invalid_mass")))
            g["runtime"].append(_opt_float(rec.get("runtime_seconds")))

    # Resolve the common-support population per (domain, calibration).
    by_dc: dict[tuple, list[tuple]] = defaultdict(list)
    for key in groups:
        by_dc[(key[0], key[1])].append(key)

    def _nanmean(a: np.ndarray) -> float:
        a = a[~np.isnan(a)]
        return float(a.mean()) if a.size else float("nan")

    def _iqm(a: np.ndarray) -> float:
        """Interquartile mean: mean of the values within [Q1, Q3]. Robust to the
        heavy right tail while remaining a mean (cf. rliable, Agarwal 2021)."""
        a = np.sort(a[~np.isnan(a)])
        if a.size == 0:
            return float("nan")
        q1, q3 = np.percentile(a, [25, 75])
        mid = a[(a >= q1) & (a <= q3)]
        return float(mid.mean()) if mid.size else float(a.mean())

    rows: list[dict[str, Any]] = []
    for (domain, calibration), keys in by_dc.items():
        inst_sets = {k: set(groups[k]["inst"]) for k in keys}
        theta_keys = list(keys)
        comp_keys = [k for k in theta_keys if k[2] in EXP1_SHARED_SUPPORT_METHODS]
        shared = (
            set.intersection(*(inst_sets[k] for k in comp_keys)) if comp_keys else set()
        )
        own: dict[str, set] = {}
        for m in {k[2] for k in theta_keys}:
            mkeys = [k for k in theta_keys if k[2] == m]
            own[m] = set.intersection(*(inst_sets[k] for k in mkeys)) if mkeys else set()

        for key in keys:
            _d, _c, method, theta = key
            g = groups[key]
            n_ok = len(g["inst"])
            if n_ok == 0:
                continue
            if method in EXP1_SHARED_SUPPORT_METHODS and shared:
                support, scope = shared, "shared"
            else:
                support, scope = own.get(method, set()), "own-theta"

            inst = np.asarray(g["inst"])
            if support:
                mask = np.isin(inst, np.fromiter(support, dtype=int))
            else:
                mask = np.zeros(inst.shape, dtype=bool)
            if not mask.any():
                continue

            cov = np.asarray(g["cov"])[mask]
            compact = np.asarray(g["compact"])[mask]
            full = np.asarray(g["full"])[mask]
            mass = np.asarray(g["mass"])[mask]
            mass = mass[~np.isnan(mass)]
            # The literal "None" segment is part of the bootstrap seed label;
            # keep it so every seeded CI matches the published results.
            base = f"{model_segment}|{domain}|{calibration}|{method}|{theta}|None|{scope}"
            cov_m, cov_lo, cov_hi = _bootstrap_mean_ci(cov, label=f"{base}|cov", resamples=resamples)
            comp_m, comp_lo, comp_hi = _bootstrap_mean_ci(compact, label=f"{base}|compact", resamples=resamples)
            full_m, full_lo, full_hi = _bootstrap_mean_ci(full, label=f"{base}|full", resamples=resamples)
            mass_m, mass_lo, mass_hi = _bootstrap_mean_ci(mass, label=f"{base}|mass", resamples=resamples)
            support_size = len(support)
            rows.append(
                {
                    "Model": model_segment,
                    "Model Label": model_label(model_segment),
                    "Domain": domain,
                    "Calibration": calibration,
                    "Method": method,
                    "Method Label": EXP1_METHOD_LABELS.get(method, method),
                    "Theta": float(theta) if theta is not None else np.nan,
                    "N": int(mask.sum()),          # instances actually averaged
                    "N OK": n_ok,                   # instances with an ok record at this param
                    "N Total": n_total[key],        # records of any status at this param
                    "Completion": n_ok / n_total[key] if n_total[key] else np.nan,
                    # Fraction of averaged instances with an EMPTY set (size 0).
                    # An empty set never covers the truth, so a high value
                    # explains a small average size paired with lower coverage.
                    "Frac Empty Set": float(np.mean(compact == 0)),
                    "Support": scope,
                    "Support Size": support_size,
                    "Coverage": cov_m,
                    "Coverage CI Low": cov_lo,
                    "Coverage CI High": cov_hi,
                    # Belief-set size enumerated in the search space (paper Exp-1 size).
                    "Avg Compact Size": comp_m,
                    "Avg Compact Size CI Low": comp_lo,
                    "Avg Compact Size CI High": comp_hi,
                    # Median size — robust to the heavy right tail. MBS (cumulative
                    # mass) has occasional enormous sets that dominate the MEAN;
                    # the median shows the typical/per-instance behaviour.
                    "Median Compact Size": float(np.median(compact)),
                    "Median Full Size": float(np.median(full)),
                    # Interquartile mean — robust mean (middle 50%), tail-trimmed
                    # but still a mean.
                    "IQM Compact Size": _iqm(compact),
                    "IQM Full Size": _iqm(full),
                    # Downstream full-feature-space count (Exp-3 belief size).
                    "Avg Full Size": full_m,
                    "Avg Full Size CI Low": full_lo,
                    "Avg Full Size CI High": full_hi,
                    "Avg Mass": mass_m,
                    "Avg Mass CI Low": mass_lo,
                    "Avg Mass CI High": mass_hi,
                    "Avg Invalid Mass": _nanmean(np.asarray(g["invalid"])[mask]),
                    "Avg Runtime": _nanmean(np.asarray(g["runtime"])[mask]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment 3 streaming aggregation
# ---------------------------------------------------------------------------

import re as _re

_HORIZON_RE = _re.compile(r"(?:^|_)h(?P<h>\d+)(?:_|$)")
_OBJECTS_RE = _re.compile(r"(?:^|_)n(?P<n>\d+)(?:_|$)")

_EXP3_STATUSES = {"ok", "timeout", "no_plan", "error", "unsupported"}
# ``method`` values that are error-record placeholders, not real planning methods.
_NON_METHODS = frozenset({"bundle", "prediction", None})


def _parse_int(pattern: _re.Pattern, text: str | None) -> float:
    if not text:
        return np.nan
    m = pattern.search(text)
    return int(m.group(1)) if m else np.nan


@dataclass
class _Exp3Acc:
    success: list[float] = field(default_factory=list)
    unsafe: list[float] = field(default_factory=list)
    pre_fail: list[float] = field(default_factory=list)
    no_plan: list[float] = field(default_factory=list)
    # Status breakdown — a "timeout" is a failure caused by the planning budget,
    # not by the method genuinely finding no plan; keeping them separate makes
    # timeout-dominated rows (e.g. prefix-search) honest in the success table.
    status: Counter = field(default_factory=Counter)
    belief_sum: float = 0.0
    belief_n: int = 0
    n: int = 0


def _aggregate_exp3_file(
    path: Path, model_segment: str, *, resamples: int
) -> pd.DataFrame:
    accs: dict[tuple, _Exp3Acc] = defaultdict(_Exp3Acc)
    population_counts: Counter[tuple[Any, Any, Any]] = Counter()

    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            status = rec.get("status")
            if status not in _EXP3_STATUSES:
                continue
            method = rec.get("method")
            if method in _NON_METHODS:
                continue  # error-record placeholders (bundle/prediction load failures)
            domain = rec.get("domain")
            config = rec.get("configuration")
            calibration = rec.get("calibration")
            if method == "oracle":
                population_counts[(domain, config, calibration)] += 1
            theta = rec.get("theta")
            key = (domain, config, calibration, method, theta)
            acc = accs[key]
            acc.success.append(1.0 if rec.get("success") else 0.0)
            acc.unsafe.append(1.0 if rec.get("unsafe_first") else 0.0)
            acc.pre_fail.append(1.0 if rec.get("pre_fail") else 0.0)
            acc.no_plan.append(1.0 if rec.get("no_plan_fn") else 0.0)
            acc.status[status] += 1
            bs = rec.get("logical_full_belief_size", rec.get("belief_size"))
            if bs is not None:
                acc.belief_sum += float(bs)
                acc.belief_n += 1
            acc.n += 1

    rows: list[dict[str, Any]] = []
    for key, acc in accs.items():
        domain, config, calibration, method, theta = key
        population_n = int(population_counts.get((domain, config, calibration), acc.n))
        succ = np.asarray(acc.success)
        # The literal "None" segment is part of the bootstrap seed label; keep
        # it so every seeded CI matches the published results.
        base = f"{model_segment}|{domain}|{config}|{calibration}|{method}|{theta}|None"
        s_m, s_lo, s_hi = _bootstrap_mean_ci(
            succ, label=f"{base}|success", resamples=resamples
        )
        rows.append(
            {
                "Model": model_segment,
                "Model Label": model_label(model_segment),
                "Domain": domain,
                "Configuration": config,
                "Horizon": _parse_int(_HORIZON_RE, config),
                "Objects": _parse_int(_OBJECTS_RE, config),
                "Calibration": calibration,
                "Method": method,
                "Method Label": EXP3_METHOD_LABELS.get(method, method),
                "Theta": float(theta) if theta is not None else np.nan,
                "N": acc.n,
                "Population N": population_n,
                "Completion Rate": acc.n / population_n if population_n else np.nan,
                "Success Rate": s_m,
                "Success Rate CI Low": s_lo,
                "Success Rate CI High": s_hi,
                "Unsafe First Rate": float(np.mean(acc.unsafe)),
                "Pre Fail Rate": float(np.mean(acc.pre_fail)),
                "No Plan Fn Rate": float(np.mean(acc.no_plan)),
                # Status breakdown (fractions over this group's records).
                "Timeout Rate": acc.status.get("timeout", 0) / acc.n if acc.n else np.nan,
                "No Plan Rate": acc.status.get("no_plan", 0) / acc.n if acc.n else np.nan,
                "OK Rate": acc.status.get("ok", 0) / acc.n if acc.n else np.nan,
                "Unsupported Rate": acc.status.get("unsupported", 0) / acc.n if acc.n else np.nan,
                "Error Rate": acc.status.get("error", 0) / acc.n if acc.n else np.nan,
                "Avg Belief Size": acc.belief_sum / acc.belief_n if acc.belief_n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _ensure_exp3_population_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add oracle-population denominators to older Exp-3 summary caches."""
    if frame.empty:
        return frame
    out = frame.copy()
    keys = ["Model", "Domain", "Configuration", "Calibration"]
    if "Population N" not in out.columns:
        oracle = (
            out[out["Method"].eq("oracle")][keys + ["N"]]
            .rename(columns={"N": "Population N"})
            .drop_duplicates(subset=keys)
        )
        out = out.merge(oracle, on=keys, how="left")
    out["Population N"] = out["Population N"].fillna(out["N"])
    if "Completion Rate" not in out.columns:
        out["Completion Rate"] = out["N"] / out["Population N"].where(
            out["Population N"].ne(0), np.nan
        )
    else:
        out["Completion Rate"] = out["Completion Rate"].fillna(
            out["N"] / out["Population N"].where(out["Population N"].ne(0), np.nan)
        )
    return out


# ---------------------------------------------------------------------------
# Cache plumbing
# ---------------------------------------------------------------------------


def _source_signature(path: Path) -> tuple[int, int]:
    st = path.stat()
    return (st.st_size, int(st.st_mtime))


def _load_or_build_model_summary(
    model_dir: Path,
    cache_name: str,
    builder,
    *,
    rebuild: bool,
    resamples: int,
    verbose: bool,
) -> pd.DataFrame:
    source = model_dir / "sample_results.jsonl"
    cache = model_dir / cache_name
    sig = _source_signature(source)
    if cache.exists() and not rebuild:
        try:
            with cache.open("rb") as fh:
                payload = pickle.load(fh)
            if (
                payload.get("version") == CACHE_VERSION
                and tuple(payload.get("signature", ())) == sig
                and payload.get("resamples") == resamples
            ):
                if verbose:
                    print(f"  [cache] {model_dir.name}: loaded {len(payload['frame'])} rows")
                return payload["frame"]
        except Exception as exc:  # pragma: no cover - corrupt cache -> rebuild
            if verbose:
                print(f"  [cache] {model_dir.name}: rebuild ({exc})")
    if verbose:
        size_gb = sig[0] / 1e9
        print(f"  [build] {model_dir.name}: streaming {size_gb:.2f} GB ...", flush=True)
    frame = builder(source, model_dir.name, resamples=resamples)
    payload = {
        "version": CACHE_VERSION,
        "signature": sig,
        "resamples": resamples,
        "frame": frame,
    }
    tmp = cache.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache)
    if verbose:
        print(f"  [build] {model_dir.name}: wrote {len(frame)} rows -> {cache.name}")
    return frame


def load_exp1_summary(
    models: Sequence[str] | None = None,
    *,
    rebuild: bool = False,
    resamples: int = 1000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Tidy Experiment-1 summary across models (cached per model)."""
    models = list(models) if models is not None else exp1_models()
    frames = []
    if verbose:
        print(f"Experiment 1 summary for {len(models)} model(s):")
    for seg in models:
        model_dir = EXP1_ROOT / seg
        if not (model_dir / "sample_results.jsonl").exists():
            if verbose:
                print(f"  [skip] {seg}: no sample_results.jsonl")
            continue
        frames.append(
            _load_or_build_model_summary(
                model_dir,
                _EXP1_CACHE_NAME,
                _aggregate_exp1_file,
                rebuild=rebuild,
                resamples=resamples,
                verbose=verbose,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_exp3_summary(
    models: Sequence[str] | None = None,
    *,
    rebuild: bool = False,
    resamples: int = 1000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Tidy Experiment-3 summary across models (cached per model)."""
    models = list(models) if models is not None else exp3_models()
    frames = []
    if verbose:
        print(f"Experiment 3 summary for {len(models)} model(s):")
    for seg in models:
        model_dir = EXP3_ROOT / seg
        if not (model_dir / "sample_results.jsonl").exists():
            if verbose:
                print(f"  [skip] {seg}: no sample_results.jsonl")
            continue
        frames.append(
            _load_or_build_model_summary(
                model_dir,
                _EXP3_CACHE_NAME,
                _aggregate_exp3_file,
                rebuild=rebuild,
                resamples=resamples,
                verbose=verbose,
            )
        )
    if not frames:
        return pd.DataFrame()
    return _ensure_exp3_population_columns(pd.concat(frames, ignore_index=True))


# Heavy per-record fields (lists/dicts) dropped from the slim record cache; we
# keep every other (scalar) field so the existing exp3_display stratified
# analyses find what they need.
_EXP3_DROP_FIELDS = frozenset(
    {
        "actions",
        "prevas_prefix_search_attempts",
        "adaptive_fluent_abstention_attempts",
        "adaptive_fluent_abstention_tau_candidates",
        "prevas_prefix_search_prevas_stats",
        "state_preview",
    }
)


def load_exp3_records(
    model: str, *, rebuild: bool = False, verbose: bool = True
) -> list[dict[str, Any]]:
    """Slim per-sample Experiment-3 records for one model (cached pickle).

    Used by the per-sample stratified analyses (MAP recovery/retention,
    cost-quality, decision-quality) that need joins across methods at the
    individual-instance level.  Heavy fields (state previews, plans, prevas
    stats) are dropped.
    """
    model_dir = EXP3_ROOT / model
    source = model_dir / "sample_results.jsonl"
    cache = model_dir / _EXP3_RECORDS_CACHE_NAME
    sig = _source_signature(source)
    if cache.exists() and not rebuild:
        try:
            with cache.open("rb") as fh:
                payload = pickle.load(fh)
            if (
                payload.get("version") == CACHE_VERSION
                and tuple(payload.get("signature", ())) == sig
            ):
                if verbose:
                    print(f"  [cache] exp3 records {model}: {len(payload['records'])}")
                return payload["records"]
        except Exception:
            pass
    if verbose:
        print(f"  [build] exp3 records {model}: streaming ...", flush=True)
    records: list[dict[str, Any]] = []
    with source.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            records.append(
                {k: v for k, v in rec.items() if k not in _EXP3_DROP_FIELDS}
            )
    payload = {"version": CACHE_VERSION, "signature": sig, "records": records}
    tmp = cache.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache)
    if verbose:
        print(f"  [build] exp3 records {model}: {len(records)} rows")
    return records


def load_exp6_uniform_hidden_summary(
    model: str,
    *,
    calibration: str = "global",
    run_name: str = EXP6_UNIFORM_HIDDEN_RUN,
) -> dict[str, Any]:
    """Load and validate one completed Experiment-6 uniform-hidden aggregate."""
    path = (
        EXP6_CONTROLLED_PRIOR_ROOT
        / model
        / run_name
        / "aggregate_metrics.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"No Experiment-6 uniform-hidden aggregate for model {model!r}: "
            f"{path}"
        )
    with path.open() as fh:
        summary = json.load(fh)
    config = summary.get("config", {})
    expected_model = model_label(model)
    if config.get("model") != expected_model:
        raise ValueError(
            f"Experiment-6 model mismatch: requested {expected_model!r}, "
            f"aggregate contains {config.get('model')!r}."
        )
    if calibration not in config.get("calibrations", []):
        raise ValueError(
            f"Calibration {calibration!r} is unavailable; aggregate contains "
            f"{config.get('calibrations', [])!r}."
        )
    if config.get("prior_mode") != "uniform_hidden":
        raise ValueError(
            "Experiment-6 aggregate is not a uniform-hidden run: "
            f"{config.get('prior_mode')!r}."
        )
    decision = summary.get("analysis_decision", {}).get("decision")
    if decision != "analysis-ready":
        raise ValueError(
            f"Experiment-6 aggregate is not analysis-ready: {decision!r}."
        )
    return summary


def load_exp7_correlated_noise_summary(
    model: str,
    *,
    calibration: str = "global",
    run_name: str = EXP7_CORRELATED_NOISE_RUN,
) -> dict[str, Any]:
    """Load and validate one completed Experiment-7 correlated-noise aggregate."""
    path = (
        EXP7_CORRELATED_NOISE_ROOT
        / model
        / run_name
        / "aggregate_metrics.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"No Experiment-7 correlated-noise aggregate for model {model!r}: {path}"
        )
    with path.open() as fh:
        summary = json.load(fh)
    config = summary.get("config", {})
    expected_model = model_label(model)
    if config.get("model") != expected_model:
        raise ValueError(
            f"Experiment-7 model mismatch: requested {expected_model!r}, "
            f"aggregate contains {config.get('model')!r}."
        )
    if calibration not in config.get("calibrations", []):
        raise ValueError(
            f"Calibration {calibration!r} is unavailable; aggregate contains "
            f"{config.get('calibrations', [])!r}."
        )
    decision = summary.get("analysis_decision", {}).get("decision")
    if decision != "analysis-ready":
        raise ValueError(
            f"Experiment-7 aggregate is not analysis-ready: {decision!r}."
        )
    return summary


# ---------------------------------------------------------------------------
# Small filter / introspection helpers (used in the notebook)
# ---------------------------------------------------------------------------


def available(summary: pd.DataFrame, column: str) -> list[Any]:
    if summary.empty or column not in summary.columns:
        return []
    vals = summary[column].dropna().unique().tolist()
    try:
        return sorted(vals)
    except TypeError:
        return vals


# ===========================================================================
# Display: shared styling
# ===========================================================================

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# A stable visual identity per method.  Each method ID has a fixed style slot
# (palette / marker index), resolved through the current display-label maps so
# label-only edits do not silently fall back to gray.  Slots are fixed numbers
# so every method keeps the colour and marker of the published figures.
_STYLE_METHOD_ORDER: tuple[tuple[int, int, str], ...] = (
    (0, 1, "mbs_prevas"),
    (5, 3, "mbs_prevas_constraints_exact"),
    (4, 3, "mbs_prevas_constraints_relaxed"),
    (6, 3, "mbs_prevas_prefix_search"),
    (8, 3, "mbs_prevas_prefix_search_constraints_exact"),
    (7, 3, "mbs_prevas_prefix_search_constraints_relaxed"),
    (9, 1, "random_theta_confident"),
    (10, 1, "fluent_abstention_unconstrained"),
    (12, 3, "adaptive_fluent_abstention"),
    (15, 1, "unconstrained_top_mass"),
    (16, 1, "mbs_first"),
    (17, 1, "mbs_first_constraints"),
    (18, 1, "map"),
    (19, 1, "valid_map"),
    (20, 3, "oracle"),
)
_STYLE_SLOT_COUNT = 21


def _style_slots() -> dict[str, int]:
    """Style slot per display label (first method listed for a label wins)."""
    slots: dict[str, int] = {}
    for slot, experiment, method in _STYLE_METHOD_ORDER:
        slots.setdefault(method_label(method, experiment=experiment), slot)
    next_slot = _STYLE_SLOT_COUNT
    for label in list(EXP1_METHOD_LABELS.values()) + list(EXP3_METHOD_LABELS.values()):
        if label not in slots:
            slots[label] = next_slot
            next_slot += 1
    return slots


_STYLE_SLOTS = _style_slots()
_MARKERS = ["o", "s", "D", "P", "X", "v", "^", "<", ">", "*", "h", "p", "d", "8"]


def _style_indices_by_method() -> dict[tuple[int, str], int]:
    indices: dict[tuple[int, str], int] = {}
    for experiment, labels in ((1, EXP1_METHOD_LABELS), (3, EXP3_METHOD_LABELS)):
        for method, label in labels.items():
            indices[(experiment, method)] = _STYLE_SLOTS[label]
    return indices


_STYLE_INDICES_BY_METHOD = _style_indices_by_method()


def _style_maps() -> tuple[
    dict[str, Any],
    dict[str, str],
    dict[tuple[int, str], Any],
    dict[tuple[int, str], str],
]:
    import seaborn as sns

    n_slots = max(_STYLE_SLOT_COUNT, max(_STYLE_SLOTS.values()) + 1)
    palette = sns.color_palette("tab20", n_colors=max(20, n_slots))
    colors = {label: palette[i % len(palette)] for label, i in _STYLE_SLOTS.items()}
    markers = {label: _MARKERS[i % len(_MARKERS)] for label, i in _STYLE_SLOTS.items()}
    method_colors = {
        key: palette[i % len(palette)] for key, i in _STYLE_INDICES_BY_METHOD.items()
    }
    method_markers = {
        key: _MARKERS[i % len(_MARKERS)] for key, i in _STYLE_INDICES_BY_METHOD.items()
    }
    return colors, markers, method_colors, method_markers


_METHOD_COLORS, _METHOD_MARKERS, _METHOD_ID_COLORS, _METHOD_ID_MARKERS = _style_maps()


def _color(label: str):
    return _METHOD_COLORS.get(label, "0.4")


def _marker(label: str) -> str:
    return _METHOD_MARKERS.get(label, "o")


def _method_style_key(method: str, *, experiment: int = 1) -> tuple[int, str] | None:
    key = (experiment, method)
    if key in _METHOD_ID_COLORS:
        return key
    for fallback_experiment in (1, 3):
        key = (fallback_experiment, method)
        if key in _METHOD_ID_COLORS:
            return key
    return None


def _color_for_method(method: str, *, experiment: int = 1):
    key = _method_style_key(method, experiment=experiment)
    if key is not None:
        return _METHOD_ID_COLORS[key]
    return _color(method_label(method, experiment=experiment))


def _marker_for_method(method: str, *, experiment: int = 1) -> str:
    key = _method_style_key(method, experiment=experiment)
    if key is not None:
        return _METHOD_ID_MARKERS[key]
    return _marker(method_label(method, experiment=experiment))


def show(fig: Figure) -> None:
    """Render a figure inline exactly once, then close it.

    Mirrors ``results_display.show_figures``: with an interactive/inline backend
    we ``plt.show()`` and then ``plt.close()`` so the figure is not re-flushed at
    the end of the cell.  Returns ``None`` so a trailing ``nd.show(fig)`` does not
    also emit an ``execute_result`` (which would double-render the figure).
    """
    if plt.get_backend().lower() != "agg":
        plt.figure(fig.number)
        plt.show()
    plt.close(fig)


def save(fig: Figure, name: str, *, subdir: str = "figures") -> Path:
    out_dir = ROOT / subdir
    out_dir.mkdir(exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    return path


def _resolve_methods(
    requested: Sequence[str] | None, default: Sequence[str]
) -> list[str]:
    return list(requested) if requested is not None else list(default)


def _exp1_size_cols(
    size_field: str, size_stat: str = "mean"
) -> tuple[str, str | None, str | None, str]:
    """Map ``(size_field, size_stat)`` to its (value, ci_low, ci_high, label) cols.

    ``size_field``: ``"compact"`` (default) is the enumerated belief-set size in
    the search space — the size the paper reports (≈10^0–10^2). ``"full"`` is the
    downstream full-feature-space count used in Experiment 3 (≈10^7 on PRB
    because null/omitted fluents are expanded under the at-most-one constraints).

    ``size_stat``: ``"mean"`` (default, with bootstrap CI), ``"median"``, or
    ``"iqm"`` (interquartile mean — mean of the middle 50%). The mean is dominated
    by MBS's heavy right tail (cumulative-mass sets blow up on high-entropy
    instances), so the **median** or **IQM** is the more robust summary. ``median``
    and ``iqm`` have no CI column (``ci_low``/``ci_high`` are ``None``).
    """
    if size_field not in ("compact", "full"):
        raise ValueError("size_field must be 'compact' or 'full'")
    if size_stat not in ("mean", "median", "iqm"):
        raise ValueError("size_stat must be 'mean', 'median', or 'iqm'")
    noun = "full-space belief" if size_field == "full" else "belief-set"
    field = "Full" if size_field == "full" else "Compact"
    if size_stat == "median":
        return (f"Median {field} Size", None, None, f"Median {noun} size")
    if size_stat == "iqm":
        return (f"IQM {field} Size", None, None, f"IQM {noun} size")
    return (
        f"Avg {field} Size",
        f"Avg {field} Size CI Low",
        f"Avg {field} Size CI High",
        f"Average {noun} size",
    )


def _format_ci(value, lo, hi, *, precision: int, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    scale = 100.0 if pct else 1.0
    suffix = "%" if pct else ""
    body = f"{value * scale:.{precision}f}{suffix}"
    if lo is None or hi is None or (isinstance(lo, float) and math.isnan(lo)):
        return body
    return f"{body} [{lo * scale:.{precision}f}, {hi * scale:.{precision}f}]"


def _fmt_size_cell(row, vcol, lcol, hcol, *, precision: int) -> str:
    """Format a size value + CI from a row, tolerating ``None`` CI columns
    (the median stat has no CI)."""
    lo = row[lcol] if lcol is not None else None
    hi = row[hcol] if hcol is not None else None
    return _format_ci(row[vcol], lo, hi, precision=precision)


# ===========================================================================
# Display: Experiment 1
# ===========================================================================


def filter_exp1(
    summary: pd.DataFrame,
    *,
    model: str | None = None,
    calibration: str | None = None,
    methods: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> pd.DataFrame:
    df = summary
    if model is not None:
        df = df[df["Model"] == model]
    if calibration is not None:
        df = df[df["Calibration"] == calibration]
    if methods is not None:
        df = df[df["Method"].isin(list(methods))]
    if domains is not None:
        df = df[df["Domain"].isin(list(domains))]
    return _refresh_method_labels(df, EXP1_METHOD_LABELS)


def exp1_coverage_size_table(
    summary: pd.DataFrame,
    *,
    model: str,
    calibration: str = "global",
    methods: Sequence[str] | None = None,
    thetas: Sequence[float] | None = None,
    domains: Sequence[str] | None = None,
    size_field: str = "compact",
    size_stat: str = "mean",
) -> pd.DataFrame:
    """Coverage / set-size / mass per (θ, method) × domain, with CIs.

    ``size_field`` is ``"compact"`` (paper belief-set size, default) or ``"full"``.
    """
    methods = _resolve_methods(methods, EXP1_THETA_METHODS + EXP1_POINT_METHODS)
    vcol, lcol, hcol, _ = _exp1_size_cols(size_field, size_stat)
    df = filter_exp1(
        summary, model=model, calibration=calibration, methods=methods, domains=domains
    )
    if thetas is not None:
        keep = df["Theta"].isin(list(thetas)) | df["Theta"].isna()
        df = df[keep]
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["Coverage"] = df.apply(
        lambda r: _format_ci(r["Coverage"], r["Coverage CI Low"], r["Coverage CI High"], precision=3),
        axis=1,
    )
    df["Avg Set Size"] = df.apply(
        lambda r: _fmt_size_cell(r, vcol, lcol, hcol, precision=2),
        axis=1,
    )
    df["Avg Mass"] = df.apply(
        lambda r: _format_ci(r["Avg Mass"], r["Avg Mass CI Low"], r["Avg Mass CI High"], precision=3),
        axis=1,
    )
    df["θ"] = df["Theta"].map(lambda t: f"θ={t:g}" if pd.notna(t) else "point")
    rank = {label: i for i, label in enumerate(EXP1_METHOD_LABELS.values())}
    df["_mrank"] = df["Method Label"].map(rank)
    df["_trank"] = df["Theta"].fillna(np.inf)
    df = df.sort_values(["_trank", "_mrank"])
    table = df.pivot_table(
        index=["θ", "Method Label"],
        columns="Domain",
        values=["Coverage", "Avg Set Size", "Avg Mass"],
        aggfunc="first",
        sort=False,
    )
    table = table.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0, sort_remaining=False)
    return table


def _domain_axes(domains: list[str], *, width: float = 4.2, height: float = 3.8):
    fig, axes = plt.subplots(
        1, len(domains), figsize=(width * len(domains), height), squeeze=False
    )
    return fig, {d: axes[0, i] for i, d in enumerate(domains)}


def _collect_legend_entries(axes: Iterable[Any]) -> tuple[list[Any], list[str]]:
    """Collect the first plotted handle for every visible label across axes.

    Multi-panel figures use one figure-level legend for all domain panels.
    Taking handles from only the last panel silently drops any series that has
    no data in that domain (e.g. when the last panel's domain has no data).
    """
    handles_by_label: dict[str, Any] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_") and label not in handles_by_label:
                handles_by_label[label] = handle
    return list(handles_by_label.values()), list(handles_by_label)


def exp1_theta_size_plot(
    summary: pd.DataFrame,
    *,
    model: str,
    calibration: str = "global",
    methods: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    size_field: str = "compact",
    size_stat: str = "mean",
) -> Figure:
    """Average set size (y, log) vs θ."""
    methods = _resolve_methods(methods, EXP1_THETA_METHODS)
    vcol, _, _, ylabel = _exp1_size_cols(size_field, size_stat)
    df = filter_exp1(summary, model=model, calibration=calibration, methods=methods,
                     domains=domains)
    if df.empty:
        raise ValueError("No rows for the requested selection.")
    doms = order_domains(df["Domain"].unique())
    fig, axes = _domain_axes(doms)
    for domain in doms:
        ax = axes[domain]
        ddf = df[df["Domain"] == domain]
        for method in methods:
            mdf = ddf[ddf["Method"] == method].sort_values("Theta")
            if mdf.empty:
                continue
            label = mdf["Method Label"].iloc[0]
            ax.plot(mdf["Theta"], mdf[vcol], marker=_marker_for_method(method),
                    color=_color_for_method(method), label=label, lw=1.8, ms=5)
        ax.set_yscale("log")
        ax.set_xlabel("θ (target mass)")
        ax.set_title(domain.upper())
        ax.grid(True, alpha=0.3)
    axes[doms[0]].set_ylabel(ylabel)
    handles, labels = axes[doms[-1]].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"{model_label(model)} — size vs θ ({calibration})")
    fig.tight_layout()
    return fig


# ===========================================================================
# Display: Experiment 1 — multi-model comparison (new)
# ===========================================================================


def exp1_model_comparison_plot(
    summary: pd.DataFrame,
    *,
    method: str = "mbs_prevas",
    calibration: str = "global",
    models: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    size_field: str = "compact",
    size_stat: str = "mean",
    min_completion: float = 0.0,
) -> Figure:
    """Coverage vs size for a single method, one curve per model — per domain.

    Lets you see how the SSE backbone (the model) shifts the coverage/size
    frontier while holding the set-construction method and calibration fixed.

    Each model's curve uses *that model's* common support (see the build), so the
    populations differ across models. ``min_completion`` drops a model whose
    worst completion for this selection is below the threshold (e.g. 0.9 to hide
    partial runs).
    """
    vcol, _, _, xlabel = _exp1_size_cols(size_field, size_stat)
    df = filter_exp1(summary, calibration=calibration, methods=[method], domains=domains)
    if models is not None:
        df = df[df["Model"].isin(list(models))]
    if df.empty:
        raise ValueError("No rows for the requested selection.")
    is_point = method in EXP1_POINT_METHODS
    sort_col = "Theta"
    doms = order_domains(df["Domain"].unique())
    model_segs = list(df["Model"].unique())
    # Worst completion per model across this selection (for the legend / filter).
    worst_comp = df.groupby("Model")["Completion"].min().to_dict()
    model_segs = [m for m in model_segs if worst_comp.get(m, 0.0) >= min_completion]
    if not model_segs:
        raise ValueError("No model meets min_completion for this selection.")
    import seaborn as sns
    palette = sns.color_palette("tab10", n_colors=max(10, len(model_segs)))
    mcolor = {m: palette[i % len(palette)] for i, m in enumerate(model_segs)}
    fig, axes = _domain_axes(doms)
    for domain in doms:
        ax = axes[domain]
        ddf = df[df["Domain"] == domain]
        for seg in model_segs:
            mdf = ddf[ddf["Model"] == seg].sort_values(sort_col)
            if mdf.empty:
                continue
            lbl = model_label(seg)
            if is_point:
                row = mdf.iloc[0]
                ax.scatter([row[vcol]], [row["Coverage"]], marker="*", s=130,
                           color=mcolor[seg], label=lbl)
            else:
                ax.plot(mdf[vcol], mdf["Coverage"], marker="o",
                        color=mcolor[seg], label=lbl, lw=1.8, ms=4)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(domain.upper())
        ax.grid(True, alpha=0.3)
    axes[doms[0]].set_ylabel("Coverage")
    handles, labels = _collect_legend_entries(axes.values())
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)),
               bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"{method_label(method)} across models ({calibration})")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    return fig


def exp1_coverage_size_by_model_plot(
    summary: pd.DataFrame,
    *,
    methods: Sequence[str],
    calibration: str,
    models: Sequence[str] | None,
    domains: Sequence[str],
    size_field: str,
    size_stat: str,
) -> Figure:
    """Coverage–size grid: one model row, with all methods overlaid by domain."""
    methods = _resolve_methods(methods, EXP1_THETA_METHODS + EXP1_POINT_METHODS)
    vcol, _, _, xlabel = _exp1_size_cols(size_field, size_stat)
    df = filter_exp1(
        summary,
        calibration=calibration,
        methods=methods,
        domains=domains,
    )
    if models is not None:
        df = df[df["Model"].isin(list(models))]
    if df.empty:
        raise ValueError("No rows for the requested selection.")

    doms = order_domains(df["Domain"].unique())
    model_segs = list(df["Model"].unique())
    fig, axes = plt.subplots(
        len(model_segs),
        len(doms),
        figsize=(4.0 * len(doms), 2.6 * len(model_segs) + 1.0),
        sharey=True,
        squeeze=False,
    )
    for model_index, model in enumerate(model_segs):
        for domain_index, domain in enumerate(doms):
            ax = axes[model_index, domain_index]
            ddf = df[(df["Model"] == model) & (df["Domain"] == domain)]
            missing_methods: list[str] = []
            for method in methods:
                mdf = ddf[ddf["Method"] == method].sort_values("Theta")
                if mdf.empty:
                    missing_methods.append(method_label(method))
                    continue
                label = method_label(method)
                color = _color_for_method(method)
                marker = _marker_for_method(method)
                if method in EXP1_POINT_METHODS:
                    row = mdf.iloc[0]
                    ax.scatter(
                        [row[vcol]],
                        [row["Coverage"]],
                        marker="*",
                        s=95,
                        color=color,
                        zorder=5,
                        label=label,
                    )
                    continue
                ax.plot(
                    mdf[vcol],
                    mdf["Coverage"],
                    marker=marker,
                    color=color,
                    label=label,
                    lw=1.5,
                    ms=4,
                )
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.25)
            if missing_methods:
                ax.text(
                    0.02,
                    0.98,
                    "No data: " + ", ".join(missing_methods),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.5,
                    color="0.35",
                )
            if model_index == 0:
                ax.set_title(domain.upper())
            if domain_index == 0:
                compact_model = model_label(model).split("/")[-1]
                for suffix in ("-Instruct", "-2025-04-14", "-it"):
                    compact_model = compact_model.removesuffix(suffix)
                ax.set_ylabel(
                    f"{compact_model}\nCoverage",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=35,
                )
            if model_index == len(model_segs) - 1:
                ax.set_xlabel(xlabel)

    handles, labels = _collect_legend_entries(axes.flat)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(labels)),
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(f"Coverage–size frontiers by backbone ({calibration})")
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    return fig


def exp1_mass_coverage_by_domain_plot(
    summary: pd.DataFrame,
    *,
    method: str,
    calibration: str,
    models: Sequence[str] | None,
    domains: Sequence[str],
) -> Figure:
    """Realized set mass vs empirical coverage, with models overlaid by domain."""
    df = filter_exp1(
        summary,
        calibration=calibration,
        methods=[method],
        domains=domains,
    )
    if models is not None:
        df = df[df["Model"].isin(list(models))]
    df = df[df["Theta"].notna()]
    if df.empty:
        raise ValueError("No rows for the requested selection.")

    doms = order_domains(df["Domain"].unique())
    model_segs = list(df["Model"].unique())
    import seaborn as sns

    palette = sns.color_palette("tab10", n_colors=max(10, len(model_segs)))
    model_colors = {
        model: palette[index % len(palette)]
        for index, model in enumerate(model_segs)
    }
    fig, axes = _domain_axes(doms, height=4.1)
    for domain in doms:
        ax = axes[domain]
        ddf = df[df["Domain"] == domain]
        for model in model_segs:
            mdf = ddf[ddf["Model"] == model].sort_values("Theta")
            if mdf.empty:
                continue
            ax.plot(
                mdf["Avg Mass"],
                mdf["Coverage"],
                marker="o",
                ms=3.5,
                lw=1.2,
                color=model_colors[model],
                label=model_label(model),
            )
        ax.plot(
            [0, 1],
            [0, 1],
            ls="--",
            c="gray",
            lw=1,
            label="Ideal calibration",
        )
        ax.set_title(domain.upper())
        ax.set_xlabel("Average realized set mass")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
    axes[doms[0]].set_ylabel("Empirical true-state coverage")

    handles, labels = _collect_legend_entries(axes.values())
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, len(labels)),
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(f"{method_label(method)} mass–coverage reliability ({calibration})")
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    return fig


def exp1_mass_coverage_by_model_plot(
    summary: pd.DataFrame,
    *,
    methods: Sequence[str],
    calibration: str,
    models: Sequence[str] | None,
    domains: Sequence[str],
) -> Figure:
    """Mass–coverage reliability grid: one model row, all methods overlaid."""
    methods = _resolve_methods(methods, EXP1_THETA_METHODS + EXP1_POINT_METHODS)
    df = filter_exp1(
        summary,
        calibration=calibration,
        methods=methods,
        domains=domains,
    )
    if models is not None:
        df = df[df["Model"].isin(list(models))]
    if df.empty:
        raise ValueError("No rows for the requested selection.")

    doms = order_domains(df["Domain"].unique())
    model_segs = list(df["Model"].unique())
    fig, axes = plt.subplots(
        len(model_segs),
        len(doms),
        figsize=(4.0 * len(doms), 2.6 * len(model_segs) + 1.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for model_index, model in enumerate(model_segs):
        for domain_index, domain in enumerate(doms):
            ax = axes[model_index, domain_index]
            ddf = df[(df["Model"] == model) & (df["Domain"] == domain)]
            missing_methods: list[str] = []
            for method in methods:
                mdf = ddf[ddf["Method"] == method].sort_values("Theta")
                if mdf.empty:
                    missing_methods.append(method_label(method))
                    continue
                label = method_label(method)
                color = _color_for_method(method)
                marker = _marker_for_method(method)
                if method in EXP1_POINT_METHODS:
                    row = mdf.iloc[0]
                    ax.scatter(
                        [row["Avg Mass"]],
                        [row["Coverage"]],
                        marker="*",
                        s=95,
                        color=color,
                        zorder=5,
                        label=label,
                    )
                    continue
                ax.plot(
                    mdf["Avg Mass"],
                    mdf["Coverage"],
                    marker=marker,
                    color=color,
                    label=label,
                    lw=1.5,
                    ms=4,
                )
            ax.plot(
                [0, 1],
                [0, 1],
                ls="--",
                c="gray",
                lw=1,
                label="Ideal calibration",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.25)
            if missing_methods:
                ax.text(
                    0.02,
                    0.98,
                    "No data: " + ", ".join(missing_methods),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.5,
                    color="0.35",
                )
            if model_index == 0:
                ax.set_title(domain.upper())
            if domain_index == 0:
                compact_model = model_label(model).split("/")[-1]
                for suffix in ("-Instruct", "-2025-04-14", "-it"):
                    compact_model = compact_model.removesuffix(suffix)
                ax.set_ylabel(
                    f"{compact_model}\nCoverage",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=35,
                )
            if model_index == len(model_segs) - 1:
                ax.set_xlabel("Average realized set mass")

    handles, labels = _collect_legend_entries(axes.flat)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, len(labels)),
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(f"Mass–coverage reliability by backbone ({calibration})")
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    return fig


# ===========================================================================
# Display: Experiment 3
# ===========================================================================


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0 or p is None or math.isnan(p):
        return (np.nan, np.nan, np.nan)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # Bracket p with the interval: analytically lo ≤ p ≤ hi, but at p∈{0,1}
    # the bound is 0/1 ± floating-point noise, which would make (p−lo)/(hi−p)
    # a tiny negative and break matplotlib's non-negative yerr contract.
    return (p, min(p, max(0.0, centre - half)), max(p, min(1.0, centre + half)))


def _pool_success(
    group: pd.DataFrame,
    *,
    denominator_col: str = "N",
    denominator_n: int | None = None,
) -> tuple[float, float, float, int]:
    """Pool per-configuration Bernoulli success rates into one rate + Wilson CI."""
    if denominator_n is None:
        if denominator_col not in group.columns:
            denominator_col = "N"
        denominator_n = int(group[denominator_col].sum())
    n = int(denominator_n)
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    succ = float((group["Success Rate"] * group["N"]).sum())
    p = succ / n
    return (*_wilson(p, n), n)


def filter_exp3(
    summary: pd.DataFrame,
    *,
    model: str | None = None,
    calibration: str | None = None,
    methods: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    theta: float | None = None,
) -> pd.DataFrame:
    df = summary
    if model is not None:
        df = df[df["Model"] == model]
    if calibration is not None:
        df = df[df["Calibration"] == calibration]
    if methods is not None:
        df = df[df["Method"].isin(list(methods))]
    if domains is not None:
        df = df[df["Domain"].isin(list(domains))]
    if theta is not None:
        df = df[df["Theta"] == theta]
    return _refresh_method_labels(df, EXP3_METHOD_LABELS)


EXP3_DEFAULT_METHODS = (
    "mbs_prevas",
    "mbs_prevas_prefix_search_constraints_exact",
    "random_theta",
    "fluent_abstention_unconstrained",
    "map",
)


def _select_exp3_operating_points(df: pd.DataFrame, theta: float | None) -> pd.DataFrame:
    """Select one θ operating point per Exp-3 method for display helpers.

    ``theta=None`` means "use the documented default operating point for each
    method", not "pool every recorded θ row".  Passing an explicit ``theta``
    keeps the old pin-all-θ-methods behavior.
    """
    if df.empty:
        return df
    if theta is not None:
        return df[(df["Theta"] == theta) | (df["Theta"].isna())].copy()

    selected = df["Theta"].isna()
    for method, default_theta in EXP3_METHOD_OPERATING_THETAS.items():
        selected = selected | (df["Method"].eq(method) & df["Theta"].eq(default_theta))
    return df[selected].copy()


def exp3_success_table(
    summary: pd.DataFrame,
    *,
    model: str,
    calibration: str = "global",
    methods: Sequence[str] | None = None,
    theta: float | None = None,
    domains: Sequence[str] = ("prb", "gw"),
    show_timeout: bool = True,
) -> pd.DataFrame:
    """Planning success rate (pooled over configs) per method × domain, with CIs.

    θ-keyed methods are taken at ``theta`` when set. With ``theta=None``, each
    method uses its documented operating point (plain θ-methods at θ=0.01,
    prefix-search methods at θ=0.99).

    With ``show_timeout`` (default), each cell appends the **planning-timeout
    rate** — the fraction of records that hit the planning budget and were
    therefore counted as failures *without* the method genuinely finding no
    plan. This is essential context: belief-set methods (esp. prefix-search at
    θ=0.99 and the conformant constraint variants) are often timeout-dominated,
    so a low success rate there reflects the planning budget, not the belief
    set. MAP's committed single-state planning is classical and ~never times
    out, so it is not directly comparable at a fixed budget.
    """
    methods = _resolve_methods(methods, EXP3_DEFAULT_METHODS)
    df = filter_exp3(summary, model=model, calibration=calibration, methods=methods,
                     domains=domains)
    df = _select_exp3_operating_points(df, theta)
    if df.empty:
        return pd.DataFrame()
    rows = []
    rank = {label: i for i, label in enumerate(EXP3_METHOD_LABELS.values())}
    for (method, label), mdf in df.groupby(["Method", "Method Label"], sort=False):
        row: dict[str, Any] = {"Method": label, "_rank": rank.get(label, 99)}
        for domain in domains:
            sub = mdf[mdf["Domain"] == domain]
            p, lo, hi, n = _pool_success(sub)
            cell = _format_ci(p, lo, hi, precision=1, pct=True) + (f" (n={n})" if n else "")
            if show_timeout and n and "Timeout Rate" in sub.columns:
                to = float((sub["Timeout Rate"] * sub["N"]).sum() / sub["N"].sum())
                cell += f" [TO {to * 100:.0f}%]"
            row[domain.upper()] = cell
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("_rank").drop(columns="_rank")
    return out.set_index("Method")


# ===========================================================================
# Display: Experiment 6 closed-loop ALF replanning
# ===========================================================================


# ===========================================================================
# Display: Experiment 6 uniform-hidden closed-loop planning
# ===========================================================================


def _validate_exp6_selection(
    summary: Mapping[str, Any],
    *,
    model: str,
    calibration: str,
    methods: Sequence[str],
) -> None:
    config = summary.get("config", {})
    expected_model = model_label(model)
    if config.get("model") != expected_model:
        raise ValueError(
            f"Requested model {expected_model!r}, but the Experiment-6 summary "
            f"contains {config.get('model')!r}."
        )
    if calibration not in config.get("calibrations", []):
        raise ValueError(
            f"Calibration {calibration!r} is not present in the Experiment-6 "
            "summary."
        )
    if config.get("prior_mode") != "uniform_hidden":
        raise ValueError("Experiment-6 selection is not uniform-hidden.")
    unavailable = set(methods) - set(config.get("methods", []))
    if unavailable:
        raise ValueError(f"Unavailable Experiment-6 methods: {sorted(unavailable)}")


def _format_exp6_success(stat: Mapping[str, Any]) -> str:
    n = int(stat["n"])
    successes = round(n * float(stat["success_rate"]))
    return f"{successes}/{n} ({100 * float(stat['success_rate']):.1f}%)"


def exp6_closed_loop_success_table(
    summary: Mapping[str, Any],
    *,
    model: str,
    calibration: str,
    methods: Sequence[str],
) -> pd.DataFrame:
    """Ground-truth plan execution before and after one shared observation."""
    _validate_exp6_selection(
        summary, model=model, calibration=calibration, methods=methods
    )
    rows: list[dict[str, Any]] = []
    for method in methods:
        before = summary["method_round"][f"m0:{method}:k0"]
        after = summary["method_round"][f"m0:{method}:k1"]
        change = 100 * (
            float(after["success_rate"]) - float(before["success_rate"])
        )
        rows.append(
            {
                "Method": EXP6_METHOD_LABELS.get(method, method),
                "Before observation": _format_exp6_success(before),
                "After observation": _format_exp6_success(after),
                "Observation change": f"{change:+.1f} pp",
            }
        )
    return pd.DataFrame(rows).set_index("Method")


def exp6_closed_loop_effect_table(
    summary: Mapping[str, Any],
    *,
    model: str,
    calibration: str,
    methods: Sequence[str],
) -> pd.DataFrame:
    """Paired execution-success contrasts with sample-bootstrap intervals."""
    _validate_exp6_selection(
        summary, model=model, calibration=calibration, methods=methods
    )
    available = set(methods)
    comparisons = [
        ("ams", "map"),
        ("adaptive_abstention", "map"),
        ("ams", "adaptive_abstention"),
    ]
    rows: list[dict[str, Any]] = []
    for left, right in comparisons:
        if left not in available or right not in available:
            continue
        row: dict[str, Any] = {
            "Paired contrast": (
                f"{EXP6_METHOD_LABELS.get(left, left)} − "
                f"{EXP6_METHOD_LABELS.get(right, right)}"
            )
        }
        for round_index, label in (
            (0, "Before observation"),
            (1, "After observation"),
        ):
            stat = summary["paired_success_differences"][
                f"m0:k{round_index}:{left}-{right}"
            ]
            row[label] = (
                f"{100 * float(stat['mean']):+.1f} pp "
                f"[{100 * float(stat['ci_low']):+.1f}, "
                f"{100 * float(stat['ci_high']):+.1f}]"
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("Paired contrast")


def exp6_closed_loop_efficiency_table(
    summary: Mapping[str, Any],
    *,
    model: str,
    calibration: str,
    methods: Sequence[str],
) -> pd.DataFrame:
    """Selected belief size/mass and complete method-level search effort."""
    _validate_exp6_selection(
        summary, model=model, calibration=calibration, methods=methods
    )
    unsupported = set(methods) - {"ams", "adaptive_abstention"}
    if unsupported:
        raise ValueError(
            "Experiment-6 efficiency data are available only for AMS and "
            f"adaptive abstention, not {sorted(unsupported)}."
        )
    rows: list[dict[str, Any]] = []
    for round_index, round_label in (
        (0, "Before observation"),
        (1, "After observation"),
    ):
        for method in methods:
            stat = summary["method_efficiency"][
                f"m0:{method}:k{round_index}"
            ]
            rows.append(
                {
                    "Round": round_label,
                    "Method": EXP6_METHOD_LABELS.get(method, method),
                    "Belief size, mean (median)": (
                        f"{float(stat['mean_belief_size']):.1f} "
                        f"({float(stat['median_belief_size']):.0f})"
                    ),
                    "Probability mass, mean (median)": (
                        f"{100 * float(stat['mean_probability_mass']):.1f}% "
                        f"({100 * float(stat['median_probability_mass']):.1f}%)"
                    ),
                    "Total search time, mean (median)": (
                        f"{float(stat['mean_total_search_runtime_seconds']):.1f}s "
                        f"({float(stat['median_total_search_runtime_seconds']):.1f}s)"
                    ),
                    "Attempts, mean": (
                        f"{float(stat['mean_search_attempts']):.1f}"
                    ),
                }
            )
    return pd.DataFrame(rows).set_index(["Round", "Method"])


def exp6_closed_loop_plot(
    summary: Mapping[str, Any],
    *,
    model: str,
    calibration: str,
    methods: Sequence[str],
) -> Figure:
    """Four-panel execution, set-size, mass, and search-cost comparison."""
    _validate_exp6_selection(
        summary, model=model, calibration=calibration, methods=methods
    )
    efficiency_methods = [
        method
        for method in methods
        if method in {"ams", "adaptive_abstention"}
    ]
    if not efficiency_methods:
        raise ValueError("Plot requires AMS or adaptive abstention.")

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    width = 0.34
    round_specs = (
        (0, "Before observation", "0.65"),
        (1, "After observation", "#2b8cbe"),
    )

    ax = axes[0, 0]
    x = np.arange(len(methods))
    for round_index, round_label, color in round_specs:
        values = [
            100
            * float(
                summary["method_round"][
                    f"m0:{method}:k{round_index}"
                ]["success_rate"]
            )
            for method in methods
        ]
        offset = (-width / 2) if round_index == 0 else (width / 2)
        bars = ax.bar(x + offset, values, width, color=color, label=round_label)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [EXP6_METHOD_LABELS.get(method, method) for method in methods],
        rotation=12,
        ha="right",
    )
    ax.set_ylim(0, 108)
    ax.set_ylabel("Execution success (%)")
    ax.set_title("(a) Ground-truth plan execution")
    ax.grid(True, axis="y", alpha=0.25)
    legend_handles, legend_labels = ax.get_legend_handles_labels()

    def grouped_efficiency_panel(
        ax,
        field: str,
        *,
        scale: float,
        ylabel: str,
        title: str,
        value_format: str,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        xe = np.arange(len(efficiency_methods))
        for round_index, round_label, color in round_specs:
            values = [
                scale
                * float(
                    summary["method_efficiency"][
                        f"m0:{method}:k{round_index}"
                    ][field]
                )
                for method in efficiency_methods
            ]
            offset = (-width / 2) if round_index == 0 else (width / 2)
            bars = ax.bar(
                xe + offset,
                values,
                width,
                color=color,
                label=round_label,
            )
            ax.bar_label(bars, fmt=value_format, padding=2, fontsize=8)
        ax.set_xticks(xe)
        ax.set_xticklabels(
            [EXP6_METHOD_LABELS.get(method, method) for method in efficiency_methods]
        )
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

    grouped_efficiency_panel(
        axes[0, 1],
        "mean_belief_size",
        scale=1.0,
        ylabel="Complete states",
        title="(b) Selected belief-set size",
        value_format="%.1f",
    )
    grouped_efficiency_panel(
        axes[1, 0],
        "mean_probability_mass",
        scale=100.0,
        ylabel="Constrained mass (%)",
        title="(c) Selected probability mass",
        value_format="%.1f",
        ylim=(0, 108),
    )
    grouped_efficiency_panel(
        axes[1, 1],
        "mean_total_search_runtime_seconds",
        scale=1.0,
        ylabel="Planner-runtime sum (s)",
        title="(d) Total planning search effort",
        value_format="%.1f",
    )

    n = int(summary["complete_sample_conditions"])
    fig.suptitle(
        f"ALF uniform-hidden closed-loop planning — {model_label(model)} "
        f"({calibration}, paired n={n})"
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig


# ===========================================================================
# Display: Experiment 7 correlated perception errors
# ===========================================================================


def _validate_exp7_selection(
    summary: Mapping[str, Any],
    *,
    model: str,
    calibration: str,
    group_mode: str,
) -> None:
    config = summary.get("config", {})
    expected_model = model_label(model)
    if config.get("model") != expected_model:
        raise ValueError(
            f"Requested model {expected_model!r}, but the Experiment-7 summary "
            f"contains {config.get('model')!r}."
        )
    if calibration not in config.get("calibrations", []):
        raise ValueError(
            f"Calibration {calibration!r} is not present in the Experiment-7 "
            "summary."
        )
    if group_mode not in config.get("group_modes", []):
        raise ValueError(
            f"Unavailable Experiment-7 corruption block mode: {group_mode!r}."
        )


def exp7_sweep_rows(
    summary: Mapping[str, Any],
    *,
    group_mode: str,
    strength: float,
    theta: float,
    domains: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Sweep cells matching one operating point, ordered by domain then rate.

    The uncorrelated control (rate 0) is stored once per theta because every
    strength names the same model there, so it is matched on rate alone.
    """
    rows = [
        dict(cell)
        for cell in summary.get("sweep", {}).values()
        if cell.get("group_mode") == group_mode
        and float(cell.get("theta")) == float(theta)
        and (
            float(cell.get("rate")) == 0.0
            or float(cell.get("strength")) == float(strength)
        )
    ]
    if domains is not None:
        allowed = set(domains)
        rows = [row for row in rows if row.get("domain") in allowed]
    order = {domain: i for i, domain in enumerate(order_domains(PREFERRED_DOMAINS))}
    return sorted(
        rows, key=lambda row: (order.get(row["domain"], 99), float(row["rate"]))
    )


def _format_exp7_pp(stat: Mapping[str, Any]) -> str:
    if stat.get("mean") is None:
        return "—"
    return (
        f"{100 * float(stat['mean']):+.1f} pp "
        f"[{100 * float(stat['ci_low']):+.1f}, {100 * float(stat['ci_high']):+.1f}]"
    )


def exp7_distortion_table(
    summary: Mapping[str, Any],
    *,
    model: str,
    calibration: str,
    group_mode: str,
    strength: float,
    theta: float,
    domains: Sequence[str] | None = None,
) -> pd.DataFrame:
    """How far the product belief drifts from the correlated joint it approximates."""
    _validate_exp7_selection(
        summary, model=model, calibration=calibration, group_mode=group_mode
    )
    rows: list[dict[str, Any]] = []
    for cell in exp7_sweep_rows(
        summary,
        group_mode=group_mode,
        strength=strength,
        theta=theta,
        domains=domains,
    ):
        variation = cell.get("total_variation_mean")
        ratio = cell.get("size_ratio_median")
        meets = cell.get("meets_theta_rate")
        rows.append(
            {
                "Domain": str(cell["domain"]).upper(),
                "Nuisance rate π": f"{float(cell['rate']):.2f}",
                "n": int(cell["n"]),
                "Total variation": "—" if variation is None else f"{variation:.3f}",
                "Attributed − true mass": _format_exp7_pp(cell["mass_coverage_gap"]),
                "Reaches θ": "—" if meets is None else f"{100 * float(meets):.1f}%",
                "Size vs true-joint set": "—" if ratio is None else f"{ratio:.2f}×",
            }
        )
    return pd.DataFrame(rows).set_index(["Domain", "Nuisance rate π"])


def exp7_robustness_table(
    summary: Mapping[str, Any],
    *,
    model: str,
    calibration: str,
    group_mode: str,
    strength: float,
    theta: float,
    methods: Sequence[str],
    domains: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Ground-truth state coverage per method as the correlated nuisance strengthens."""
    _validate_exp7_selection(
        summary, model=model, calibration=calibration, group_mode=group_mode
    )
    rows: list[dict[str, Any]] = []
    for cell in exp7_sweep_rows(
        summary,
        group_mode=group_mode,
        strength=strength,
        theta=theta,
        domains=domains,
    ):
        row: dict[str, Any] = {
            "Domain": str(cell["domain"]).upper(),
            "Nuisance rate π": f"{float(cell['rate']):.2f}",
            "n": int(cell["n"]),
        }
        coverage = cell.get("method_truth_coverage", {})
        sizes = cell.get("method_size_median", {})
        for method in methods:
            label = EXP7_METHOD_LABELS.get(method, method)
            value = coverage.get(method)
            size = sizes.get(method)
            row[label] = (
                "—"
                if value is None
                else f"{100 * float(value):.1f}%"
                + ("" if size is None else f" (|S|={float(size):g})")
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index(["Domain", "Nuisance rate π"])


def exp7_block_mode_table(
    summary: Mapping[str, Any],
    *,
    model: str,
    calibration: str,
    group_modes: Sequence[str],
    strength: float,
    theta: float,
    domains: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Does the constraint layer absorb correlation it was never designed to model?

    Corrupting a whole mutex group is the adversarial case: the features are already
    coupled by a constraint, so conditioning cannot re-separate them.  Corrupting the
    least-decided features instead spreads the nuisance across groups.
    """
    for group_mode in group_modes:
        _validate_exp7_selection(
            summary, model=model, calibration=calibration, group_mode=group_mode
        )
    per_mode: dict[str, dict[tuple[str, float], dict[str, Any]]] = {}
    for group_mode in group_modes:
        per_mode[group_mode] = {
            (str(cell["domain"]), float(cell["rate"])): cell
            for cell in exp7_sweep_rows(
                summary,
                group_mode=group_mode,
                strength=strength,
                theta=theta,
                domains=domains,
            )
        }
    keys = sorted(
        {key for cells in per_mode.values() for key in cells},
        key=lambda key: (
            {d: i for i, d in enumerate(order_domains(PREFERRED_DOMAINS))}.get(
                key[0], 99
            ),
            key[1],
        ),
    )
    rows: list[dict[str, Any]] = []
    for domain, rate in keys:
        if rate == 0.0:
            continue
        row: dict[str, Any] = {
            "Domain": domain.upper(),
            "Nuisance rate π": f"{rate:.2f}",
        }
        for group_mode in group_modes:
            cell = per_mode[group_mode].get((domain, rate))
            label = EXP7_GROUP_MODE_LABELS.get(group_mode, group_mode)
            if cell is None:
                row[label] = "—"
                continue
            meets = cell.get("meets_theta_rate")
            row[label] = (
                _format_exp7_pp(cell["mass_coverage_gap"])
                + ("" if meets is None else f", θ met {100 * float(meets):.0f}%")
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index(["Domain", "Nuisance rate π"])


def exp7_correlated_noise_plot(
    summary: Mapping[str, Any],
    *,
    model: str,
    calibration: str,
    group_mode: str,
    strength: float,
    theta: float,
    domains: Sequence[str] | None = None,
) -> Figure:
    """Four-panel sensitivity view of the conditional-independence assumption."""
    _validate_exp7_selection(
        summary, model=model, calibration=calibration, group_mode=group_mode
    )
    rows = exp7_sweep_rows(
        summary,
        group_mode=group_mode,
        strength=strength,
        theta=theta,
        domains=domains,
    )
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(str(row["domain"]), []).append(row)
    ordered = [d for d in order_domains(PREFERRED_DOMAINS) if d in by_domain]
    colors = {"prb": "#1b9e77", "gw": "#d95f02", "alf": "#7570b3"}
    markers = {"prb": "o", "gw": "s", "alf": "^"}

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))

    def rates(cells: Sequence[Mapping[str, Any]]) -> list[float]:
        return [float(cell["rate"]) for cell in cells]

    ax = axes[0][0]
    for domain in ordered:
        cells = by_domain[domain]
        values = [cell.get("total_variation_mean") for cell in cells]
        ax.plot(
            rates(cells),
            [float("nan") if v is None else float(v) for v in values],
            marker=markers.get(domain, "o"),
            color=colors.get(domain),
            label=domain.upper(),
        )
    ax.set_ylabel("Total variation")
    ax.set_title("(a) Distance from the true joint")

    ax = axes[0][1]
    for domain in ordered:
        cells = by_domain[domain]
        means = [100 * float(cell["mass_coverage_gap"]["mean"]) for cell in cells]
        lows = [100 * float(cell["mass_coverage_gap"]["ci_low"]) for cell in cells]
        highs = [100 * float(cell["mass_coverage_gap"]["ci_high"]) for cell in cells]
        ax.plot(
            rates(cells),
            means,
            marker=markers.get(domain, "o"),
            color=colors.get(domain),
            label=domain.upper(),
        )
        ax.fill_between(
            rates(cells), lows, highs, color=colors.get(domain), alpha=0.15, linewidth=0
        )
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Attributed − true mass (pp)")
    ax.set_title("(b) Reported minus realized mass")

    ax = axes[1][0]
    for domain in ordered:
        cells = by_domain[domain]
        values = [cell.get("size_ratio_median") for cell in cells]
        ax.plot(
            rates(cells),
            [float("nan") if v is None else float(v) for v in values],
            marker=markers.get(domain, "o"),
            color=colors.get(domain),
            label=domain.upper(),
        )
    ax.axhline(1.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_ylabel("|MBSθ| ÷ |true-joint set|")
    ax.set_title("(c) Cost in set size")

    ax = axes[1][1]
    for domain in ordered:
        cells = by_domain[domain]
        for method, style in (("mbs_prevas", "-"), ("map", ":")):
            values = [
                cell.get("method_truth_coverage", {}).get(method) for cell in cells
            ]
            ax.plot(
                rates(cells),
                [
                    float("nan") if v is None else 100 * float(v)
                    for v in values
                ],
                linestyle=style,
                marker=markers.get(domain, "o") if method == "mbs_prevas" else None,
                markersize=4,
                color=colors.get(domain),
                label=(
                    f"{domain.upper()} {EXP7_METHOD_LABELS.get(method, method)}"
                    if method == "map"
                    else f"{domain.upper()} MBSθ"
                ),
            )
    ax.set_ylim(0, 105)
    ax.set_ylabel("True-state coverage (%)")
    ax.set_title("(d) True-state coverage, MBSθ vs MAP")
    # Colour already encodes the domain, so this second legend only has to say
    # which line style is which method.
    ax.legend(
        handles=[
            Line2D([], [], color="0.35", linestyle="-", marker="o", markersize=4),
            Line2D([], [], color="0.35", linestyle=":"),
        ],
        labels=["MBSθ", "MAP"],
        loc="lower left",
        fontsize=7,
        frameon=True,
        framealpha=0.85,
    )

    for row in axes:
        for ax in row:
            ax.set_xlabel("Nuisance-event rate π")
            ax.grid(True, alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=max(1, len(ordered)),
        frameon=False,
        fontsize=8,
    )
    samples = int(summary.get("samples", 0))
    fig.suptitle(
        f"Correlated perception errors — {model_label(model)} "
        f"({calibration}, {EXP7_GROUP_MODE_LABELS.get(group_mode, group_mode).lower()}, "
        f"ρ={strength:g}, θ={theta:g}, n={samples})"
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig
