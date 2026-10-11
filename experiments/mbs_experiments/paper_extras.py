"""Paper results that need more than the display modules provide.

Pure helpers over experiment ``sample_results.jsonl`` records, in the style of
:mod:`mbs_experiments.notebook_data` (pandas tables and matplotlib figures
returned to the caller; no file I/O besides :func:`load_records`):

* :func:`load_records` -- stream a results file, keeping only some records and
  fields, so that multi-GB files fit in memory.
* :func:`exp3_paired_constraint_ablation` -- paired-bootstrap differences in
  planning success between Experiment-3 methods on matched test instances
  (the constraint ablation added during the review).
* :func:`exp3_confidence_success_figure` -- paper Fig. 7: AMS confidence (the
  mass of the belief set AMS plans for) against planning success, one point
  per PRB (objects, horizon) cell, with the Pearson correlation.
* :func:`exp2_states_visited_table` -- median explored-state counts of PRaVaSS
  by domain and θ on constrained instances, next to the unconstrained-case
  bound of Proposition 1 (added during the review).

Typical notebook use::

    records = nd.load_exp3_records("OpenAI__gpt-4.1-2025-04-14")
    exp3_paired_constraint_ablation(records)
    fig, r, p = exp3_confidence_success_figure(records)

    exp2 = load_records(".../experiment_2_prevas_scaling/<model>/sample_results.jsonl")
    exp2_states_visited_table(exp2)
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy import stats

from .exp3_display import (
    EXP3_PREFIX_SEARCH_METHOD,
    EXP3_PREFIX_THETA,
    exp3_prefix_search_success_matrix,
)

# ---------------------------------------------------------------------------
# Shared record helpers
# ---------------------------------------------------------------------------

DOMAIN_ORDER = ("prb", "gw", "alf")

# The Experiment-1 fields that ``results_display.exp1_coverage_size_summary``
# reads. Keeping only these makes multi-GB result files fit in memory.
EXP1_SUMMARY_FIELDS = (
    "domain",
    "configuration",
    "sample",
    "split_relative_path",
    "calibration",
    "method",
    "theta",
    "status",
    "compact_set_size",
    "full_set_size",
    "set_size",
    "contains_truth",
    "constrained_mass",
)


def load_records(
    path,
    *,
    fields: Sequence[str] | None = None,
    calibration: str | None = None,
    methods: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Stream a ``sample_results.jsonl``, keeping only the requested records and fields.

    ``fields=None`` keeps every field. ``calibration`` and ``methods`` filter
    records before they are stored.
    """
    method_set = None if methods is None else set(methods)
    records: list[dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if calibration is not None and record.get("calibration") != calibration:
                continue
            if method_set is not None and record.get("method") not in method_set:
                continue
            if fields is not None:
                record = {field: record.get(field) for field in fields}
            records.append(record)
    return records


# Experiment-3 record statuses that count as attempts (same set as
# ``exp3_display._oracle_population`` / ``notebook_data._EXP3_STATUSES``).
EXP3_STATUSES = frozenset({"ok", "timeout", "no_plan", "unsupported", "error"})
# Experiment-3 methods whose rows carry no θ (``theta`` is null).
EXP3_POINT_METHODS = frozenset({"oracle", "map", "valid_map", "adaptive_fluent_abstention"})


def _theta_key(theta: Any) -> float | None:
    if theta is None or (isinstance(theta, float) and np.isnan(theta)):
        return None
    return round(float(theta), 10)


def _instance_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("domain"),
        record.get("configuration"),
        record.get("sample"),
        record.get("split_relative_path"),
    )


def order_domains(domains: Iterable[Any]) -> list[Any]:
    """Order domains as PRB, GW, ALF, then anything else alphabetically."""
    found = list(dict.fromkeys(domains))
    rank = {d: i for i, d in enumerate(DOMAIN_ORDER)}
    return sorted(found, key=lambda d: (rank.get(d, len(rank)), str(d)))


def _format_pp(value: float) -> str:
    # ``round`` first so a tiny negative that displays as zero prints ``+0.0``.
    return f"{round(100.0 * float(value), 1) + 0.0:+.1f}"


# ---------------------------------------------------------------------------
# Paired-bootstrap constraint ablation (Experiment 3)
# ---------------------------------------------------------------------------

EXP3_PAIRED_METHOD_LABELS: dict[str, str] = {
    "map": "MAP",
    "adaptive_fluent_abstention": "Adaptive Abstention",
    "mbs_prevas_prefix_search": "AMS-Loose (Ours)",
    "mbs_prevas_prefix_search_constraints_exact": "AMS (Ours)",
    "mbs_prevas_prefix_search_constraints_relaxed": "AMS (Ours, relaxed Z)",
}

# (left, right) pairs reported as ``left − right``.
EXP3_PAIRED_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("mbs_prevas_prefix_search_constraints_exact", "map"),
    ("mbs_prevas_prefix_search_constraints_exact", "adaptive_fluent_abstention"),
    ("mbs_prevas_prefix_search_constraints_exact", "mbs_prevas_prefix_search"),
    (
        "mbs_prevas_prefix_search_constraints_relaxed",
        "mbs_prevas_prefix_search_constraints_exact",
    ),
)

PAIRED_BOOTSTRAP_RESAMPLES = 10_000
PAIRED_BOOTSTRAP_SEED = 0


def exp3_method_theta(method: str, theta: float) -> float | None:
    """θ under which ``method`` is keyed in Experiment-3 records.

    Point methods (MAP, adaptive abstention, oracle) have no θ; every other
    method is read at ``theta`` (0.99 for the prefix-search operating point).
    """
    return None if method in EXP3_POINT_METHODS else _theta_key(theta)


def exp3_success_by_instance(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domain: str = "prb",
    methods: Sequence[str],
    theta: float = EXP3_PREFIX_THETA,
) -> pd.DataFrame:
    """Per-instance planning-success indicators for ``methods`` in one domain.

    Rows are the oracle population of ``domain`` (one row per test instance
    with an ``oracle`` record under ``calibration``), sorted by instance key
    ``(domain, configuration, sample, split_relative_path)`` so the result does
    not depend on the order of the JSONL; columns are ``methods``.  A cell is
    1.0 / 0.0 for a successful / failed record and NaN when the method has no
    record for that instance (later duplicate rows win, matching
    ``exp3_display._dedupe_records``).
    """
    wanted = {method: exp3_method_theta(method, theta) for method in methods}
    population: list[tuple[Any, ...]] = []
    seen: set[tuple[Any, ...]] = set()
    outcomes: dict[tuple[str, tuple[Any, ...]], float] = {}
    for record in records:
        if record.get("calibration") != calibration or record.get("domain") != domain:
            continue
        if record.get("status") not in EXP3_STATUSES:
            continue
        key = _instance_key(record)
        method = str(record.get("method"))
        record_theta = _theta_key(record.get("theta"))
        if method == "oracle" and record_theta is None and key not in seen:
            seen.add(key)
            population.append(key)
        if method in wanted and record_theta == wanted[method]:
            outcomes[(method, key)] = 1.0 if bool(record.get("success")) else 0.0

    population.sort()
    data = {
        method: [outcomes.get((method, key), np.nan) for key in population]
        for method in methods
    }
    frame = pd.DataFrame(data, columns=list(methods), dtype=float)
    frame.index = pd.MultiIndex.from_tuples(
        population or [], names=["domain", "configuration", "sample", "split_relative_path"]
    )
    return frame


def paired_bootstrap_indices(
    population_size: int,
    *,
    resamples: int = PAIRED_BOOTSTRAP_RESAMPLES,
    rng: np.random.Generator | int = PAIRED_BOOTSTRAP_SEED,
) -> np.ndarray:
    """``(resamples, population_size)`` matrix of with-replacement instance draws.

    One matrix is drawn per domain and shared by every contrast in that
    domain, so all contrasts are evaluated on the *same* resampled instances.
    """
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    return generator.integers(0, population_size, size=(resamples, population_size))


def paired_bootstrap_mean_ci(
    differences: Sequence[float] | np.ndarray,
    *,
    resamples: int = PAIRED_BOOTSTRAP_RESAMPLES,
    seed: int | np.random.Generator = PAIRED_BOOTSTRAP_SEED,
    indices: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Mean of paired differences with a 95% percentile-bootstrap interval.

    ``differences`` holds one entry per population instance; NaN marks an
    instance the contrast cannot be evaluated on (one side has no record), so
    the statistic is the mean over the matched instances of each resample.
    Instances are resampled with replacement ``resamples`` times -- either via
    ``indices`` (a matrix from :func:`paired_bootstrap_indices`) or a fresh
    matrix from ``numpy.random.default_rng(seed)`` -- and the interval is the
    2.5th / 97.5th percentile of the resampled means.
    """
    diffs = np.asarray(differences, dtype=float)
    matched = diffs[~np.isnan(diffs)]
    if matched.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(matched.mean())
    if matched.size == 1 or resamples <= 0:
        return mean, mean, mean
    if indices is None:
        indices = paired_bootstrap_indices(diffs.size, resamples=resamples, rng=seed)
    if matched.size == diffs.size:
        means = diffs[indices].mean(axis=1)
    else:
        with np.errstate(invalid="ignore"):
            means = np.nanmean(diffs[indices], axis=1)
    low, high = np.nanquantile(means, [0.025, 0.975])
    return mean, float(low), float(high)


def exp3_paired_contrast_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domains: Sequence[str] = ("prb", "gw"),
    contrasts: Sequence[tuple[str, str]] = EXP3_PAIRED_CONTRASTS,
    theta: float = EXP3_PREFIX_THETA,
    resamples: int = PAIRED_BOOTSTRAP_RESAMPLES,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Numeric paired contrasts of planning success (fractions, not pp).

    For every domain and ``(left, right)`` contrast the matched instances are
    the oracle-population instances where *both* methods have a record; the
    statistic is the mean of the per-instance success differences
    ``left − right`` with a paired percentile-bootstrap CI.

    Resampling scheme: a single
    ``numpy.random.default_rng(seed)`` stream; for each domain, in the order
    given, one ``(resamples, n)`` index matrix over the domain's oracle
    population (sorted by instance key) is drawn and shared by every contrast
    in that domain, so all contrasts see the same resampled instances.
    """
    records = list(records)
    methods = list(dict.fromkeys(m for pair in contrasts for m in pair))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for domain in domains:
        success = exp3_success_by_instance(
            records, calibration=calibration, domain=domain, methods=methods, theta=theta
        )
        indices = paired_bootstrap_indices(len(success), resamples=resamples, rng=rng)
        for left, right in contrasts:
            diffs = (success[left] - success[right]).to_numpy(dtype=float)
            paired = success[[left, right]].dropna()
            mean, low, high = paired_bootstrap_mean_ci(
                diffs, resamples=resamples, indices=indices
            )
            rows.append(
                {
                    "Domain": domain,
                    "Left": left,
                    "Right": right,
                    "Left Success": float(paired[left].mean()) if len(paired) else np.nan,
                    "Right Success": float(paired[right].mean()) if len(paired) else np.nan,
                    "Mean Difference": mean,
                    "CI Low": low,
                    "CI High": high,
                    "N Pairs": int(len(paired)),
                }
            )
    return pd.DataFrame(rows)


def exp3_paired_constraint_ablation(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domains: Sequence[str] = ("prb", "gw"),
    contrasts: Sequence[tuple[str, str]] = EXP3_PAIRED_CONTRASTS,
    theta: float = EXP3_PREFIX_THETA,
    resamples: int = PAIRED_BOOTSTRAP_RESAMPLES,
    seed: int = PAIRED_BOOTSTRAP_SEED,
    labels: Mapping[str, str] = EXP3_PAIRED_METHOD_LABELS,
) -> pd.DataFrame:
    """Paired contrasts of the constraint ablation, formatted as a table.

    Columns: ``Domain`` (upper-case), ``Contrast`` (``left − right`` labels),
    ``Paired difference`` (``+15.7 pp``), ``95% paired bootstrap CI``
    (``[+13.6, +17.8]``) and ``n pairs``.  ``frame.to_csv(path, index=False)``
    reproduces the shipped file.
    """
    summary = exp3_paired_contrast_summary(
        records,
        calibration=calibration,
        domains=domains,
        contrasts=contrasts,
        theta=theta,
        resamples=resamples,
        seed=seed,
    )
    rows = [
        {
            "Domain": str(row["Domain"]).upper(),
            "Contrast": (
                f"{labels.get(row['Left'], row['Left'])} − "
                f"{labels.get(row['Right'], row['Right'])}"
            ),
            "Paired difference": f"{_format_pp(row['Mean Difference'])} pp",
            "95% paired bootstrap CI": (
                f"[{_format_pp(row['CI Low'])}, {_format_pp(row['CI High'])}]"
            ),
            "n pairs": int(row["N Pairs"]),
        }
        for _, row in summary.iterrows()
    ]
    return pd.DataFrame(
        rows,
        columns=["Domain", "Contrast", "Paired difference", "95% paired bootstrap CI", "n pairs"],
    )


# ---------------------------------------------------------------------------
# Paper Fig. 7: AMS confidence vs. planning success (Experiment 3, PRB)
# ---------------------------------------------------------------------------


def exp3_confidence_success_points(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domain: str = "prb",
) -> pd.DataFrame:
    """One point per (objects, horizon) cell: mean AMS confidence vs. success.

    Thin wrapper around
    :func:`exp3_display.exp3_prefix_search_success_matrix` (method
    ``mbs_prevas_prefix_search_constraints_exact`` at θ = 0.99): returns its
    ``attrs["summary"]`` with cells lacking a success rate or a mean belief-set
    mass dropped.  Columns include ``Number of Objects``, ``Horizon``, ``N``,
    ``Success`` and ``Belief Set Mass``.
    """
    matrix = exp3_prefix_search_success_matrix(
        records, calibration=calibration, domain=domain
    )
    summary = matrix.attrs.get("summary") if not matrix.empty else None
    if summary is None or summary.empty:
        return pd.DataFrame(
            columns=["Number of Objects", "Horizon", "N", "Success", "Belief Set Mass"]
        )
    return summary.dropna(subset=["Success", "Belief Set Mass"]).reset_index(drop=True)


def exp3_confidence_success_correlation(points: pd.DataFrame) -> tuple[float, float]:
    """Pearson ``(r, p)`` between ``Belief Set Mass`` and ``Success``."""
    if len(points) < 3:
        return float("nan"), float("nan")
    result = stats.pearsonr(
        points["Belief Set Mass"].to_numpy(dtype=float),
        points["Success"].to_numpy(dtype=float),
    )
    return float(result[0]), float(result[1])


def exp3_confidence_success_figure(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domain: str = "prb",
    annotate: bool = False,
) -> tuple[Figure, float, float]:
    """Fig. 7 scatter of AMS confidence against success, plus Pearson ``r, p``.

    Reproduces the paper-era notebook cell: x = mean final belief-set mass of
    ``mbs_prevas_prefix_search_constraints_exact`` (θ = 0.99), y = success
    rate, one marker per PRB (objects, horizon) configuration, with the
    ``y = x`` reference line.  ``annotate`` adds ``r``/``p`` to the title.
    """
    points = exp3_confidence_success_points(
        records, calibration=calibration, domain=domain
    )
    r, p = exp3_confidence_success_correlation(points)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(
        points["Belief Set Mass"].to_numpy(dtype=float),
        points["Success"].to_numpy(dtype=float),
        alpha=0.75,
        s=60,
        label=f"{EXP3_PREFIX_SEARCH_METHOD} (θ={EXP3_PREFIX_THETA:.2f})",
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Subset Probability Mass")
    ax.set_ylabel("Success Rate")
    if annotate:
        ax.set_title(f"Confidence–success correlation (Pearson r = {r:.3f}, p = {p:.1e})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, r, p


# ---------------------------------------------------------------------------
# Explored-state counts on constrained instances (Experiment 2)
# ---------------------------------------------------------------------------


def exp2_unconstrained_bound(search_feature_count: int, set_size: int) -> int:
    """States PRaVaSS scores in the unconstrained case: ``1 + |F|·k``.

    With ``C = ∅`` every popped state is valid and selected, so exactly ``k``
    states are popped and each pop scores at most ``|F|`` bit-flip successors
    on top of the initial state -- the visited-set size behind Proposition 1's
    ``O(|F|k)`` bound.  Constrained runs can exceed it because invalid states
    are popped (and expanded) without being selected.
    """
    return 1 + int(search_feature_count) * int(set_size)


def exp2_states_visited_table(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domains: Sequence[str] | None = None,
    thetas: Sequence[float] | None = None,
    constrained_only: bool = True,
) -> pd.DataFrame:
    """Median PRaVaSS explored states by domain and θ next to the ``1 + |F|·k`` bound.

    Uses the solved (``status == "ok"``) ``prevas`` rows of Experiment 2 under
    ``calibration``.  With ``constrained_only`` (default) only instances whose
    search carries at-most-one constraint groups
    (``search_selected_group_count > 0``) are kept.  ``states_scored`` is the number of
    states PRaVaSS scored (pushed on the heap), ``compact_set_size`` is ``k``
    and ``search_feature_count`` is ``|F|``.

    Columns: ``Domain``, ``Theta``, ``N``, ``Median Search Features``,
    ``Median Set Size``, ``Median States Visited``,
    ``Median Unconstrained Bound``, ``Median Visited / Bound`` (per-instance
    ratio, then median) and ``Within Bound %`` (share of instances with
    ``states_scored <= 1 + |F|·k``).
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("method") != "prevas" or record.get("status") != "ok":
            continue
        if record.get("calibration") != calibration:
            continue
        if domains is not None and record.get("domain") not in set(domains):
            continue
        theta = _theta_key(record.get("theta"))
        if theta is None or (thetas is not None and theta not in {_theta_key(t) for t in thetas}):
            continue
        groups = record.get("search_selected_group_count")
        if constrained_only and not (groups is not None and int(groups) > 0):
            continue
        features = record.get("search_feature_count")
        size = record.get("compact_set_size")
        scored = record.get("states_scored")
        if features is None or size is None or scored is None:
            continue
        bound = exp2_unconstrained_bound(int(features), int(size))
        rows.append(
            {
                "Domain": record.get("domain"),
                "Theta": theta,
                "features": int(features),
                "size": int(size),
                "scored": float(scored),
                "bound": float(bound),
                "ratio": float(scored) / float(bound),
                "within": float(scored) <= float(bound),
            }
        )

    columns = [
        "Domain",
        "Theta",
        "N",
        "Median Search Features",
        "Median Set Size",
        "Median States Visited",
        "Median Unconstrained Bound",
        "Median Visited / Bound",
        "Within Bound %",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for domain in order_domains(frame["Domain"]):
        per_domain = frame[frame["Domain"] == domain]
        for theta in sorted(per_domain["Theta"].unique()):
            group = per_domain[per_domain["Theta"] == theta]
            out.append(
                {
                    "Domain": domain,
                    "Theta": float(theta),
                    "N": int(len(group)),
                    "Median Search Features": float(group["features"].median()),
                    "Median Set Size": float(group["size"].median()),
                    "Median States Visited": float(group["scored"].median()),
                    "Median Unconstrained Bound": float(group["bound"].median()),
                    "Median Visited / Bound": float(group["ratio"].median()),
                    "Within Bound %": 100.0 * float(group["within"].mean()),
                }
            )
    return pd.DataFrame(out, columns=columns)
