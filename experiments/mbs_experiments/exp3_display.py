from __future__ import annotations

"""Display helpers for Experiment 3 planning-success summaries.

The notebook uses these helpers to render the planning-success table,
planning-horizon bar chart, and prefix-search success matrix.
"""

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

EXP3_DEFAULT_THETA = 0.50
EXP3_PREFIX_THETA = 0.99
# AMS and MBSθ (Ours) use all extracted mutex groups with the exact normalizer, the primary
# normalizer of App. G.5 (variable elimination; it stays within its scope bound on every PRB
# and GW test instance). The "_constraints_relaxed" variants use the closed-form
# selected-groups fallback instead and form the constraint ablation.
EXP3_AMS_METHOD = "mbs_prevas_prefix_search_constraints_exact"
EXP3_MBS_METHOD = "mbs_prevas_constraints_exact"
# The methods of the planning-success table and figure (see _planning_method_specs).
EXP3_PLANNING_METHOD_ORDER = (
    "mbs_prevas_prefix_search",
    EXP3_AMS_METHOD,
    "random_theta",
    "adaptive_fluent_abstention",
    "map",
)
EXP3_PREFIX_SEARCH_METHOD = EXP3_AMS_METHOD
EXP3_PREFIX_SEARCH_MASS_FIELD = "prevas_prefix_search_final_constrained_mass"
EXP3_EXCLUDED_DOMAIN_HORIZONS: dict[str, set[int]] = {}#{"prb": {1}, "gw": {1}}
EXP3_HORIZON_PATTERN = re.compile(r"(?:^|_)h(?P<horizon>\d+)(?:_|$)")
EXP3_OBJECT_COUNT_PATTERN = re.compile(r"(?:^|_)n(?P<objects>\d+)(?:_|$)")


def _bootstrap_seed(label: str) -> int:
    digest = hashlib.sha256(f"exp3-display|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _bootstrap_ci(
    values: Sequence[float],
    *,
    label: str,
    resamples: int,
) -> tuple[float | None, float | None, float | None]:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return None, None, None

    arr = np.array(clean, dtype=float)
    observed = float(np.mean(arr))
    if len(arr) == 1 or resamples <= 0:
        return observed, observed, observed

    rng = np.random.default_rng(_bootstrap_seed(label))
    stats = np.empty(resamples, dtype=float)
    for i in range(resamples):
        stats[i] = float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
    low, high = np.quantile(stats, [0.025, 0.975])
    return observed, float(low), float(high)


def _bootstrap_ratio_of_means_ci(
    numerator_values: Sequence[float],
    denominator_values: Sequence[float],
    *,
    label: str,
    resamples: int,
) -> tuple[float | None, float | None, float | None]:
    if len(numerator_values) == 0 or len(denominator_values) == 0:
        return None, None, None
    if len(numerator_values) != len(denominator_values):
        return None, None, None

    numerator = np.array(numerator_values, dtype=float)
    denominator = np.array(denominator_values, dtype=float)
    denominator_mean = float(np.mean(denominator))
    if denominator_mean == 0.0:
        return None, None, None

    observed = float(np.mean(numerator) / denominator_mean)
    if len(numerator) == 1 or resamples <= 0:
        return observed, observed, observed

    rng = np.random.default_rng(_bootstrap_seed(label))
    stats = np.empty(resamples, dtype=float)
    valid = 0
    n = len(numerator)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        denominator_sample_mean = float(np.mean(denominator[idx]))
        if denominator_sample_mean == 0.0:
            stats[i] = np.nan
            continue
        stats[i] = float(np.mean(numerator[idx]) / denominator_sample_mean)
        valid += 1

    if valid == 0:
        return observed, np.nan, np.nan
    stats = stats[~np.isnan(stats)]
    low, high = np.quantile(stats, [0.025, 0.975])
    return observed, float(low), float(high)


def _format_percent_ci_value(
    value: Any,
    ci_low: Any,
    ci_high: Any,
    *,
    precision: int,
) -> str:
    def _fmt(v: Any) -> str:
        if pd.isna(v):
            return ""
        return f"{100.0 * float(v):.{precision}f}%"

    if pd.isna(value):
        return ""
    value_str = _fmt(value)
    if pd.isna(ci_low) or pd.isna(ci_high):
        return value_str
    return f"{value_str} [{_fmt(ci_low)}, {_fmt(ci_high)}]"


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _theta_key(theta: Any) -> float | None:
    if theta is None:
        return None
    return round(float(theta), 10)


def _result_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("domain"),
        record.get("configuration"),
        record.get("sample"),
        record.get("split_relative_path"),
        record.get("calibration"),
        str(record.get("method")),
        _theta_key(record.get("theta")),
    )


def _dedupe_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        latest[_result_key(record)] = dict(record)
    return list(latest.values())


def _record_index(records: Iterable[Mapping[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {_result_key(record): dict(record) for record in records}


def _oracle_population(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str,
    domains: Sequence[str],
) -> list[dict[str, Any]]:
    domain_set = set(domains)
    return [
        dict(record)
        for record in records
        if record.get("status") in {"ok", "timeout", "no_plan", "unsupported", "error"}
        and record.get("calibration") == calibration
        and record.get("domain") in domain_set
        and record.get("method") == "oracle"
        and _planning_record_allowed(record)
        and _theta_key(record.get("theta")) is None
    ]


def parse_goal_horizon(configuration: Any) -> int | None:
    """Extract H from configuration names such as h01, h06, or h04_n08."""
    if configuration is None:
        return None
    match = EXP3_HORIZON_PATTERN.search(str(configuration))
    if match is None:
        return None
    return int(match.group("horizon"))


def parse_object_count(configuration: Any) -> int | None:
    """Extract the object count from configurations such as h04_n08."""
    if configuration is None:
        return None
    match = EXP3_OBJECT_COUNT_PATTERN.search(str(configuration))
    if match is None:
        return None
    return int(match.group("objects"))


def _planning_record_allowed(record: Mapping[str, Any]) -> bool:
    """Return whether a record should contribute to the planning summaries."""
    domain = str(record.get("domain"))
    horizon = parse_goal_horizon(record.get("configuration"))
    excluded = EXP3_EXCLUDED_DOMAIN_HORIZONS.get(domain)
    return excluded is None or horizon not in excluded


def _planning_method_specs(theta: float) -> tuple[tuple[str, str, float | None], ...]:
    return (
        (
            "mbs_prevas_prefix_search",
            "AMS-Loose (Ours)",
            EXP3_PREFIX_THETA,
        ),
        (
            EXP3_AMS_METHOD,
            "AMS (Ours)",
            EXP3_PREFIX_THETA,
        ),
        (
            "random_theta",
            f"Random θ-Mass (θ={theta:.2f})",
            theta,
        ),
        # (
        #     "fluent_abstention_unconstrained",
        #     f"Feature Abstention (θ={theta:.2f})",
        #     theta,
        # ),
        (
            "adaptive_fluent_abstention",
            "Adaptive Abstention",
            None,
        ),
        ("map", "MAP", None),
    )


def _method_record_key(
    oracle_record: Mapping[str, Any],
    *,
    calibration: str,
    method_id: str,
    theta: float | None,
) -> tuple[Any, ...]:
    return (
        oracle_record.get("domain"),
        oracle_record.get("configuration"),
        oracle_record.get("sample"),
        oracle_record.get("split_relative_path"),
        calibration,
        method_id,
        _theta_key(theta),
    )


def _lookup_record(
    record_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
    oracle_record: Mapping[str, Any],
    *,
    calibration: str,
    method_id: str,
    theta: float | None,
) -> Mapping[str, Any] | None:
    return record_index.get(
        _method_record_key(
            oracle_record,
            calibration=calibration,
            method_id=method_id,
            theta=theta,
        )
    )


def _record_outcome(record: Mapping[str, Any]) -> dict[str, float]:
    return {
        "Success": 1.0 if bool(record.get("success")) else 0.0,
        "Precondition Violation": 1.0
        if bool(record.get("unsafe_first")) or bool(record.get("pre_fail"))
        else 0.0,
        "No Plan": 1.0 if bool(record.get("no_plan_fn")) else 0.0,
    }


def _missing_outcome(oracle_record: Mapping[str, Any]) -> dict[str, float]:
    return {
        "Success": 0.0,
        "Precondition Violation": 0.0,
        "No Plan": 1.0 if bool(oracle_record.get("success")) else 0.0,
    }


def _planning_method_rows_for_population(
    population: Sequence[Mapping[str, Any]],
    record_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    calibration: str,
    theta: float,
    bootstrap_resamples: int,
) -> list[dict[str, Any]]:
    specs = _planning_method_specs(theta)
    accum: dict[str, dict[str, Any]] = {}
    for method_id, label, method_theta in specs:
        accum[method_id] = {
            "Method": label,
            "Method ID": method_id,
            "Theta Value": method_theta,
            "recorded": 0,
            "success": [],
            "precondition": [],
            "no_plan": [],
            "map_overlap_num": [],
            "map_overlap_den": [],
        }

    for oracle_record in population:
        map_record = _lookup_record(
            record_index,
            oracle_record,
            calibration=calibration,
            method_id="map",
            theta=None,
        )
        map_failed = bool(
            map_record is not None
            and map_record.get("status") == "ok"
            and not bool(map_record.get("success"))
        )

        for method_id, _label, method_theta in specs:
            method_record = _lookup_record(
                record_index,
                oracle_record,
                calibration=calibration,
                method_id=method_id,
                theta=method_theta,
            )
            if method_record is None:
                outcome = _missing_outcome(oracle_record)
            else:
                accum[method_id]["recorded"] += 1
                outcome = _record_outcome(method_record)

            accum[method_id]["success"].append(outcome["Success"])
            accum[method_id]["precondition"].append(outcome["Precondition Violation"])
            accum[method_id]["no_plan"].append(outcome["No Plan"])
            accum[method_id]["map_overlap_num"].append(
                1.0 if outcome["No Plan"] and map_failed else 0.0
            )
            accum[method_id]["map_overlap_den"].append(outcome["No Plan"])

    rows: list[dict[str, Any]] = []
    n = len(population)
    for method_id, _label, _method_theta in specs:
        entry = accum[method_id]
        success, success_low, success_high = _bootstrap_ci(
            entry["success"],
            label=f"exp3|planning|{method_id}|success|{n}",
            resamples=bootstrap_resamples,
        )
        precond, precond_low, precond_high = _bootstrap_ci(
            entry["precondition"],
            label=f"exp3|planning|{method_id}|precondition|{n}",
            resamples=bootstrap_resamples,
        )
        no_plan, no_plan_low, no_plan_high = _bootstrap_ci(
            entry["no_plan"],
            label=f"exp3|planning|{method_id}|no_plan|{n}",
            resamples=bootstrap_resamples,
        )
        map_overlap, map_overlap_low, map_overlap_high = _bootstrap_ratio_of_means_ci(
            entry["map_overlap_num"],
            entry["map_overlap_den"],
            label=f"exp3|planning|{method_id}|map_overlap|{n}",
            resamples=bootstrap_resamples,
        )
        rows.append(
            {
                "Method": entry["Method"],
                "Method ID": entry["Method ID"],
                "Theta Value": entry["Theta Value"],
                "N": n,
                "Recorded": int(entry["recorded"]),
                "Missing": int(n - entry["recorded"]),
                "Success": success,
                "Success CI Low": success_low,
                "Success CI High": success_high,
                "Precondition Violation": precond,
                "Precondition Violation CI Low": precond_low,
                "Precondition Violation CI High": precond_high,
                "No Plan": no_plan,
                "No Plan CI Low": no_plan_low,
                "No Plan CI High": no_plan_high,
                "MAP-failed among no-plan": map_overlap,
                "MAP-failed among no-plan CI Low": map_overlap_low,
                "MAP-failed among no-plan CI High": map_overlap_high,
            }
        )
    return rows


def _horizon_method_rows_for_population(
    population: Sequence[Mapping[str, Any]],
    record_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    calibration: str,
    theta: float,
    bootstrap_resamples: int,
) -> list[dict[str, Any]]:
    specs = _planning_method_specs(theta)
    accum: dict[str, dict[str, Any]] = {}
    for method_id, label, method_theta in specs:
        accum[method_id] = {
            "Method": label,
            "Method ID": method_id,
            "Theta Value": method_theta,
            "recorded": 0,
            "success": [],
        }

    for oracle_record in population:
        for method_id, _label, method_theta in specs:
            method_record = _lookup_record(
                record_index,
                oracle_record,
                calibration=calibration,
                method_id=method_id,
                theta=method_theta,
            )
            if method_record is None:
                outcome = _missing_outcome(oracle_record)
            else:
                accum[method_id]["recorded"] += 1
                outcome = _record_outcome(method_record)
            accum[method_id]["success"].append(outcome["Success"])

    rows: list[dict[str, Any]] = []
    n = len(population)
    for method_id, _label, _method_theta in specs:
        entry = accum[method_id]
        success, success_low, success_high = _bootstrap_ci(
            entry["success"],
            label=f"exp3|horizon|{method_id}|success|{n}",
            resamples=bootstrap_resamples,
        )
        rows.append(
            {
                "Method": entry["Method"],
                "Method ID": entry["Method ID"],
                "Theta Value": entry["Theta Value"],
                "N": n,
                "Recorded": int(entry["recorded"]),
                "Missing": int(n - entry["recorded"]),
                "Success": success,
                "Success CI Low": success_low,
                "Success CI High": success_high,
            }
        )
    return rows


def exp3_planning_success_table(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domains: Sequence[str] = ("prb", "gw"),
    theta: float = EXP3_DEFAULT_THETA,
    bootstrap_resamples: int = 2_000,
) -> pd.DataFrame:
    """Return the Experiment 3 planning-success comparison table."""
    deduped = _dedupe_records(records)
    record_index = _record_index(deduped)
    oracle_population = _oracle_population(
        deduped,
        calibration=calibration,
        domains=domains,
    )

    summary_rows: list[dict[str, Any]] = []
    for domain in domains:
        domain_population = [r for r in oracle_population if r.get("domain") == domain]
        if not domain_population:
            continue
        for row in _planning_method_rows_for_population(
            domain_population,
            record_index,
            calibration=calibration,
            theta=theta,
            bootstrap_resamples=bootstrap_resamples,
        ):
            row["Domain"] = domain
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return pd.DataFrame()

    display = summary.copy()
    display["Success"] = display.apply(
        lambda row: _format_percent_ci_value(
            row["Success"],
            row["Success CI Low"],
            row["Success CI High"],
            precision=1,
        ),
        axis=1,
    )
    display["Precondition Violation"] = display.apply(
        lambda row: _format_percent_ci_value(
            row["Precondition Violation"],
            row["Precondition Violation CI Low"],
            row["Precondition Violation CI High"],
            precision=1,
        ),
        axis=1,
    )
    display["No Plan"] = display.apply(
        lambda row: _format_percent_ci_value(
            row["No Plan"],
            row["No Plan CI Low"],
            row["No Plan CI High"],
            precision=1,
        ),
        axis=1,
    )
    display["MAP-failed among no-plan"] = display.apply(
        lambda row: _format_percent_ci_value(
            row["MAP-failed among no-plan"],
            row["MAP-failed among no-plan CI Low"],
            row["MAP-failed among no-plan CI High"],
            precision=1,
        ),
        axis=1,
    )

    metric_columns = [
        "Success",
        "Precondition Violation",
        "No Plan",
        "MAP-failed among no-plan",
    ]
    table = display.pivot_table(
        index="Method",
        columns="Domain",
        values=metric_columns,
        aggfunc="first",
        sort=False,
    )
    table = table.swaplevel(0, 1, axis=1)
    domain_order = [domain for domain in domains if domain in summary["Domain"].unique()]
    table = table.reindex(
        columns=pd.MultiIndex.from_product(
            [domain_order, metric_columns],
            names=table.columns.names,
        )
    )
    method_order = [spec[1] for spec in _planning_method_specs(theta)]
    table = table.reindex(method_order)
    table = table.fillna("")
    table.index.name = "Method"
    return table


def exp3_planning_horizon_figure(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domains: Sequence[str] = ("prb", "gw"),
    theta: float = EXP3_DEFAULT_THETA,
    bootstrap_resamples: int = 2_000,
) -> Figure:
    """Return a grouped bar chart of planning success by goal horizon."""
    deduped = _dedupe_records(records)
    record_index = _record_index(deduped)
    oracle_population = _oracle_population(
        deduped,
        calibration=calibration,
        domains=domains,
    )

    summary_rows: list[dict[str, Any]] = []
    for domain in domains:
        domain_population = [r for r in oracle_population if r.get("domain") == domain]
        if not domain_population:
            continue
        horizons = _ordered_unique(
            horizon
            for horizon in (parse_goal_horizon(r.get("configuration")) for r in domain_population)
            if horizon is not None
        )
        for horizon in horizons:
            horizon_population = [
                r
                for r in domain_population
                if parse_goal_horizon(r.get("configuration")) == horizon
            ]
            for row in _horizon_method_rows_for_population(
                horizon_population,
                record_index,
                calibration=calibration,
                theta=theta,
                bootstrap_resamples=bootstrap_resamples,
            ):
                row["Domain"] = domain
                row["Horizon"] = int(horizon)
                summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise ValueError("No Experiment 3 planning horizon rows available")

    domain_order = [domain for domain in domains if domain in summary["Domain"].unique()]
    fig, axes = plt.subplots(
        1,
        max(len(domain_order), 1),
        figsize=(6.2 * max(len(domain_order), 1), 5.2),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    colors = sns.color_palette("tab10", n_colors=len(EXP3_PLANNING_METHOD_ORDER))
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=colors[i], edgecolor="none", label=spec[1])
        for i, spec in enumerate(_planning_method_specs(theta))
    ]

    for col_idx, domain in enumerate(domain_order):
        ax = axes[col_idx]
        domain_df = summary[summary["Domain"].eq(domain)].copy()
        horizons = sorted(int(h) for h in domain_df["Horizon"].dropna().unique())
        if not horizons:
            continue
        x = np.arange(len(horizons), dtype=float)
        width = 0.8 / len(EXP3_PLANNING_METHOD_ORDER)

        for method_idx, (method_id, label, _method_theta) in enumerate(
            _planning_method_specs(theta)
        ):
            method_df = (
                domain_df[domain_df["Method ID"].eq(method_id)]
                .set_index("Horizon")
                .reindex(horizons)
            )
            values = method_df["Success"].to_numpy(dtype=float) * 100.0
            low = method_df["Success CI Low"].to_numpy(dtype=float) * 100.0
            high = method_df["Success CI High"].to_numpy(dtype=float) * 100.0
            valid = ~np.isnan(values)
            if not valid.any():
                continue
            x_positions = x[valid] + (method_idx - (len(EXP3_PLANNING_METHOD_ORDER) - 1) / 2.0) * width
            yerr = np.vstack(
                [
                    np.maximum(values[valid] - low[valid], 0.0),
                    np.maximum(high[valid] - values[valid], 0.0),
                ]
            )
            ax.bar(
                x_positions,
                values[valid],
                width=width,
                color=colors[method_idx],
                edgecolor="none",
                yerr=yerr,
                capsize=3,
            )

        ax.set_title(domain.upper())
        ax.set_xticks(x)
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_ylim(0, 105)
        ax.set_xlabel("Goal horizon H")
        if col_idx == 0:
            ax.set_ylabel("Success rate (%)")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Planning success vs. goal horizon (θ={theta:.2f})", fontsize=14)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        fontsize="small",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    return fig


def _format_success_mass_cell(success: float | None, mass: float | None) -> str:
    if success is None or pd.isna(success):
        return ""
    success_str = f"{100.0 * float(success):.1f}%"
    if mass is None or pd.isna(mass):
        return success_str
    return f"{success_str} | {float(mass):.2f}"


def exp3_prefix_search_success_matrix(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domain: str = "prb",
) -> pd.DataFrame:
    """Return the PRB prefix-search success matrix.

    The matrix summarizes AMS (``EXP3_AMS_METHOD``) over
    number-of-objects × horizon configurations. Each cell shows the mean
    downstream success rate and the mean final belief-set constrained mass
    recorded in ``prevas_prefix_search_final_constrained_mass``.
    """
    deduped = _dedupe_records(records)
    record_index = _record_index(deduped)
    oracle_population = _oracle_population(
        deduped,
        calibration=calibration,
        domains=(domain,),
    )

    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for oracle_record in oracle_population:
        horizon = parse_goal_horizon(oracle_record.get("configuration"))
        objects = parse_object_count(oracle_record.get("configuration"))
        if horizon is None or objects is None:
            continue
        grouped.setdefault((objects, horizon), []).append(oracle_record)

    rows: list[dict[str, Any]] = []
    for (objects, horizon), population in sorted(grouped.items()):
        success_values: list[float] = []
        mass_values: list[float] = []
        for oracle_record in population:
            method_record = _lookup_record(
                record_index,
                oracle_record,
                calibration=calibration,
                method_id=EXP3_PREFIX_SEARCH_METHOD,
                theta=EXP3_PREFIX_THETA,
            )
            if method_record is None:
                success_values.append(0.0)
                mass_values.append(np.nan)
                continue
            success_values.append(1.0 if bool(method_record.get("success")) else 0.0)
            mass = method_record.get(EXP3_PREFIX_SEARCH_MASS_FIELD)
            mass_values.append(float(mass) if mass is not None and pd.notna(mass) else np.nan)

        success, success_low, success_high = _bootstrap_ci(
            success_values,
            label=f"exp3|prefix_search|{domain}|{objects}|{horizon}|success",
            resamples=2_000,
        )
        mass, mass_low, mass_high = _bootstrap_ci(
            mass_values,
            label=f"exp3|prefix_search|{domain}|{objects}|{horizon}|mass",
            resamples=2_000,
        )
        rows.append(
            {
                "Number of Objects": int(objects),
                "Horizon": int(horizon),
                "N": int(len(population)),
                "Success": success,
                "Success CI Low": success_low,
                "Success CI High": success_high,
                "Belief Set Mass": mass,
                "Belief Set Mass CI Low": mass_low,
                "Belief Set Mass CI High": mass_high,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return pd.DataFrame()

    display = summary.copy()
    display["Cell"] = display.apply(
        lambda row: _format_success_mass_cell(row["Success"], row["Belief Set Mass"]),
        axis=1,
    )

    matrix = display.pivot(
        index="Number of Objects",
        columns="Horizon",
        values="Cell",
    )
    matrix = matrix.reindex(
        index=sorted(summary["Number of Objects"].unique()),
        columns=sorted(summary["Horizon"].unique()),
    )
    matrix = matrix.fillna("")
    matrix.index.name = "Number of Objects"
    matrix.columns.name = "Horizon"
    matrix.attrs["summary"] = summary
    return matrix


exp3_adaptive_success_matrix = exp3_prefix_search_success_matrix


# ---------------------------------------------------------------------------
# Fixed-θ comparison (paper Table 6 and Fig. 13): the θ-indexed methods at one θ
# or over the θ grid, without the adaptive (prefix-search) variants.
# ---------------------------------------------------------------------------

EXP3_V2_METHOD_LABELS = {
    "mbs_prevas": "MBSθ-Loose (Ours)",
    EXP3_MBS_METHOD: "MBSθ (Ours)",
    "random_theta": "Random θ-Mass",
    "fluent_abstention_unconstrained": "Feature Abstention",
    "map": "MAP",
}
EXP3_V2_PARAMETRIC_METHODS = (
    "mbs_prevas",
    EXP3_MBS_METHOD,
    "random_theta",
    "fluent_abstention_unconstrained",
)
EXP3_V2_FIXED_METHODS = ("map",)
EXP3_V2_THETA_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def _v2_method_theta(method_id: str, theta: float) -> float | None:
    return theta if method_id in EXP3_V2_PARAMETRIC_METHODS else None


def _v2_solvable_population(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str,
    domains: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        r
        for r in _oracle_population(records, calibration=calibration, domains=domains)
        if bool(r.get("success"))
    ]


def _v2_map_strata(
    population: Sequence[Mapping[str, Any]],
    record_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    calibration: str,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split solvable instances by whether MAP's plan succeeded in the true state."""
    correct: list[Mapping[str, Any]] = []
    wrong: list[Mapping[str, Any]] = []
    for o in population:
        mr = _lookup_record(record_index, o, calibration=calibration, method_id="map", theta=None)
        if mr is None:
            # No MAP row at all — cannot stratify this instance.
            continue
        # ``status="no_plan"`` means MAP committed to its singleton state but
        # the classical planner could not find a plan from it; that is still a
        # MAP-failure outcome from the user's perspective.
        if bool(mr.get("success")):
            correct.append(o)
        else:
            wrong.append(o)
    return correct, wrong


def _v2_method_metric_outcomes(
    instances: Sequence[Mapping[str, Any]],
    record_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    calibration: str,
    method_id: str,
    theta: float | None,
    metric: str = "success",
) -> list[float]:
    """0/1 outcome per instance for the requested metric.

    ``metric="success"`` → 1.0 iff the method emitted a plan that achieves the
    goal in the true state. ``metric="plan_found"`` → 1.0 iff the method
    emitted any plan, regardless of whether it succeeds in the true state. A
    missing record or non-``ok`` status is counted as 0.0 in both cases.
    """
    outcomes: list[float] = []
    for o in instances:
        mr = _lookup_record(
            record_index,
            o,
            calibration=calibration,
            method_id=method_id,
            theta=theta,
        )
        if mr is None or mr.get("status") != "ok":
            outcomes.append(0.0)
        elif metric == "success":
            outcomes.append(1.0 if bool(mr.get("success")) else 0.0)
        elif metric == "plan_found":
            outcomes.append(1.0 if mr.get("plan_length") is not None else 0.0)
        else:
            raise ValueError(f"unknown metric: {metric!r}")
    return outcomes


# Backwards-compatible alias used by older call sites.
_v2_method_success_outcomes = _v2_method_metric_outcomes


_EXP3_V2_METRIC_STRATUM_LABELS = {
    "success": ("MAP-correct (Retention)", "MAP-wrong (Recovery)"),
    "plan_found": ("MAP-correct", "MAP-wrong"),
}


def _exp3_v2_map_stratified_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str,
    domains: Sequence[str],
    theta: float,
    bootstrap_resamples: int,
    metric: str,
) -> pd.DataFrame:
    """Generic MAP-stratified per-method summary for ``metric``."""
    deduped = _dedupe_records(records)
    record_index = _record_index(deduped)
    solvable = _v2_solvable_population(deduped, calibration=calibration, domains=domains)
    stratum_labels = _EXP3_V2_METRIC_STRATUM_LABELS[metric]

    rows: list[dict[str, Any]] = []
    for domain in domains:
        domain_pop = [r for r in solvable if r.get("domain") == domain]
        if not domain_pop:
            continue
        map_correct, map_wrong = _v2_map_strata(
            domain_pop, record_index, calibration=calibration
        )
        for method_id, label in EXP3_V2_METHOD_LABELS.items():
            method_theta = _v2_method_theta(method_id, theta)
            for stratum_label, stratum in zip(
                stratum_labels, (map_correct, map_wrong)
            ):
                outcomes = _v2_method_metric_outcomes(
                    stratum,
                    record_index,
                    calibration=calibration,
                    method_id=method_id,
                    theta=method_theta,
                    metric=metric,
                )
                mean, low, high = _bootstrap_ci(
                    outcomes,
                    label=f"exp3_v2|{metric}|{domain}|{method_id}|{theta}|{stratum_label}",
                    resamples=bootstrap_resamples,
                )
                rows.append(
                    {
                        "Domain": domain,
                        "Method": label,
                        "Method ID": method_id,
                        "Theta Value": method_theta,
                        "Stratum": stratum_label,
                        "N": len(stratum),
                        "Rate": mean,
                        "Rate CI Low": low,
                        "Rate CI High": high,
                    }
                )
    return pd.DataFrame(rows)


def exp3_recovery_retention_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domains: Sequence[str] = ("prb", "gw"),
    theta: float = EXP3_DEFAULT_THETA,
    bootstrap_resamples: int = 2_000,
) -> pd.DataFrame:
    """Per-method success rates, stratified by whether MAP solved the instance.

    Restricts to oracle-solvable instances. ``MAP-correct (Retention)`` is the
    fraction of MAP-correct instances that the method also solves (a high value
    means the method does not lose ground to MAP when the perception is good).
    ``MAP-wrong (Recovery)`` is the fraction of MAP-failed instances that the
    method nevertheless solves — a strict gain that any single-state method
    cannot deliver.
    """
    summary = _exp3_v2_map_stratified_summary(
        records,
        calibration=calibration,
        domains=domains,
        theta=theta,
        bootstrap_resamples=bootstrap_resamples,
        metric="success",
    )
    return summary.rename(
        columns={
            "Rate": "Success",
            "Rate CI Low": "Success CI Low",
            "Rate CI High": "Success CI High",
        }
    )


def exp3_plan_found_retention_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domains: Sequence[str] = ("prb", "gw"),
    theta: float = EXP3_DEFAULT_THETA,
    bootstrap_resamples: int = 2_000,
) -> pd.DataFrame:
    """Per-method **plan-found** rates, stratified by whether MAP solved the instance.

    Identical structure to :func:`exp3_recovery_retention_summary`, but each
    cell counts the fraction of stratum instances for which the method
    emitted *any* plan (``plan_length is not None``), regardless of whether
    that plan executes correctly in the true state.
    """
    summary = _exp3_v2_map_stratified_summary(
        records,
        calibration=calibration,
        domains=domains,
        theta=theta,
        bootstrap_resamples=bootstrap_resamples,
        metric="plan_found",
    )
    return summary.rename(
        columns={
            "Rate": "Plans Found",
            "Rate CI Low": "Plans Found CI Low",
            "Rate CI High": "Plans Found CI High",
        }
    )


def _exp3_v2_map_stratified_table(
    summary: pd.DataFrame,
    *,
    domains: Sequence[str],
    metric: str,
    value_col: str,
    low_col: str,
    high_col: str,
) -> pd.DataFrame:
    """Pivot a stratified summary into the standard display table layout."""
    if summary.empty:
        return pd.DataFrame()

    display = summary.copy()
    display["Cell"] = display.apply(
        lambda row: _format_percent_ci_value(
            row[value_col],
            row[low_col],
            row[high_col],
            precision=1,
        ),
        axis=1,
    )

    domain_order = [d for d in domains if d in summary["Domain"].unique()]
    stratum_order = list(_EXP3_V2_METRIC_STRATUM_LABELS[metric])
    method_order = [
        EXP3_V2_METHOD_LABELS[m]
        for m in EXP3_V2_METHOD_LABELS
        if EXP3_V2_METHOD_LABELS[m] in summary["Method"].unique()
    ]

    table = display.pivot_table(
        index="Method",
        columns=["Domain", "Stratum"],
        values="Cell",
        aggfunc="first",
        sort=False,
    )
    table = table.reindex(
        columns=pd.MultiIndex.from_product(
            [domain_order, stratum_order],
            names=table.columns.names,
        )
    )
    table = table.reindex(method_order)
    table = table.fillna("")
    table.index.name = "Method"

    n_row = (
        summary.drop_duplicates(subset=["Domain", "Stratum"])
        .pivot(index="Domain", columns="Stratum", values="N")
        .reindex(index=domain_order, columns=stratum_order)
    )
    n_label = pd.Series(
        {
            (d, s): f"n={int(n_row.loc[d, s])}"
            for d in domain_order
            for s in stratum_order
            if pd.notna(n_row.loc[d, s])
        }
    )
    n_frame = n_label.to_frame(name="(stratum size)").T
    n_frame.columns = pd.MultiIndex.from_tuples(n_frame.columns, names=table.columns.names)
    table = pd.concat([n_frame, table])
    return table


def exp3_recovery_retention_table(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domains: Sequence[str] = ("prb", "gw"),
    theta: float = EXP3_DEFAULT_THETA,
    bootstrap_resamples: int = 2_000,
) -> pd.DataFrame:
    """Display-ready table for ``exp3_recovery_retention_summary``."""
    summary = exp3_recovery_retention_summary(
        records,
        calibration=calibration,
        domains=domains,
        theta=theta,
        bootstrap_resamples=bootstrap_resamples,
    )
    return _exp3_v2_map_stratified_table(
        summary,
        domains=domains,
        metric="success",
        value_col="Success",
        low_col="Success CI Low",
        high_col="Success CI High",
    )


def exp3_plan_found_retention_table(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domains: Sequence[str] = ("prb", "gw"),
    theta: float = EXP3_DEFAULT_THETA,
    bootstrap_resamples: int = 2_000,
) -> pd.DataFrame:
    """Display-ready table for ``exp3_plan_found_retention_summary``.

    Same layout as :func:`exp3_recovery_retention_table` but each cell is the
    fraction of stratum instances for which the method emitted *any* plan,
    regardless of whether that plan succeeded in the true state.
    """
    summary = exp3_plan_found_retention_summary(
        records,
        calibration=calibration,
        domains=domains,
        theta=theta,
        bootstrap_resamples=bootstrap_resamples,
    )
    return _exp3_v2_map_stratified_table(
        summary,
        domains=domains,
        metric="plan_found",
        value_col="Plans Found",
        low_col="Plans Found CI Low",
        high_col="Plans Found CI High",
    )


def exp3_recovery_retention_figure(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domains: Sequence[str] = ("prb", "gw"),
    theta: float = EXP3_DEFAULT_THETA,
    bootstrap_resamples: int = 2_000,
) -> Figure:
    """Grouped bar chart of retention vs recovery for each method, per domain."""
    summary = exp3_recovery_retention_summary(
        records,
        calibration=calibration,
        domains=domains,
        theta=theta,
        bootstrap_resamples=bootstrap_resamples,
    )
    if summary.empty:
        raise ValueError("No Experiment 3 v2 records available")

    sns.set_theme(style="whitegrid")
    domain_order = [d for d in domains if d in summary["Domain"].unique()]
    fig, axes = plt.subplots(
        1,
        max(len(domain_order), 1),
        figsize=(6.0 * max(len(domain_order), 1), 5.0),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    methods_in_order = [
        EXP3_V2_METHOD_LABELS[m]
        for m in EXP3_V2_METHOD_LABELS
        if EXP3_V2_METHOD_LABELS[m] in summary["Method"].unique()
    ]
    strata = ["MAP-correct (Retention)", "MAP-wrong (Recovery)"]
    palette = sns.color_palette("Set2", n_colors=len(strata))

    for col_idx, domain in enumerate(domain_order):
        ax = axes[col_idx]
        domain_df = summary[summary["Domain"].eq(domain)]
        x = np.arange(len(methods_in_order), dtype=float)
        width = 0.4
        for s_idx, stratum in enumerate(strata):
            sub = (
                domain_df[domain_df["Stratum"].eq(stratum)]
                .set_index("Method")
                .reindex(methods_in_order)
            )
            values = sub["Success"].to_numpy(dtype=float) * 100.0
            low = sub["Success CI Low"].to_numpy(dtype=float) * 100.0
            high = sub["Success CI High"].to_numpy(dtype=float) * 100.0
            valid = ~np.isnan(values)
            yerr = np.zeros((2, len(values)))
            yerr[0, valid] = np.maximum(values[valid] - low[valid], 0.0)
            yerr[1, valid] = np.maximum(high[valid] - values[valid], 0.0)
            offset = (s_idx - 0.5) * width
            ax.bar(
                x + offset,
                np.where(valid, values, 0.0),
                width=width,
                color=palette[s_idx],
                yerr=yerr,
                capsize=3,
                label=stratum if col_idx == 0 else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(methods_in_order, rotation=20, ha="right")
        ax.set_title(domain.upper())
        ax.set_ylim(0, 105)
        if col_idx == 0:
            ax.set_ylabel("Success rate (%)")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"MAP-stratified planning success (θ={theta:.2f} for parametric methods)",
        fontsize=13,
    )
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize="small")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig


def exp3_recovery_vs_theta_figure(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domains: Sequence[str] = ("prb", "gw"),
    theta_grid: Sequence[float] = EXP3_V2_THETA_GRID,
) -> Figure:
    """Recovery rate (success on MAP-wrong instances) as θ is swept."""
    deduped = _dedupe_records(records)
    record_index = _record_index(deduped)
    solvable = _v2_solvable_population(deduped, calibration=calibration, domains=domains)

    sns.set_theme(style="whitegrid")
    domain_order = list(domains)
    fig, axes = plt.subplots(
        1,
        max(len(domain_order), 1),
        figsize=(6.0 * max(len(domain_order), 1), 5.0),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    palette = sns.color_palette("tab10", n_colors=len(EXP3_V2_METHOD_LABELS))
    method_colors = {label: palette[i] for i, label in enumerate(EXP3_V2_METHOD_LABELS.values())}

    for col_idx, domain in enumerate(domain_order):
        ax = axes[col_idx]
        domain_pop = [r for r in solvable if r.get("domain") == domain]
        if not domain_pop:
            continue
        _, map_wrong = _v2_map_strata(domain_pop, record_index, calibration=calibration)

        for method_id in EXP3_V2_PARAMETRIC_METHODS:
            label = EXP3_V2_METHOD_LABELS[method_id]
            xs: list[float] = []
            ys: list[float] = []
            for theta in theta_grid:
                outcomes = _v2_method_success_outcomes(
                    map_wrong,
                    record_index,
                    calibration=calibration,
                    method_id=method_id,
                    theta=theta,
                )
                if outcomes:
                    xs.append(float(theta))
                    ys.append(sum(outcomes) / len(outcomes))
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=label, color=method_colors[label])

        # MAP recovery is 0% by construction on MAP-wrong instances.
        ax.axhline(
            0.0,
            linestyle=":",
            linewidth=1.7,
            color=method_colors[EXP3_V2_METHOD_LABELS["map"]],
            label="MAP (0% by definition)",
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("User-specified confidence threshold (θ)")
        if col_idx == 0:
            ax.set_ylabel("Recovery rate on MAP-wrong instances")
        ax.set_title(f"{domain.upper()} (n={len(map_wrong)} MAP-wrong)")
        ax.grid(True, alpha=0.3)
        if col_idx == len(domain_order) - 1:
            ax.legend(loc="best", fontsize="small")

    fig.suptitle("Recovery rate vs θ — success on instances where MAP fails", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def exp3_cost_quality_figure(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domains: Sequence[str] = ("prb", "gw"),
    theta_grid: Sequence[float] = EXP3_V2_THETA_GRID,
) -> Figure:
    """Pareto frontier: planning success vs average belief-set size, per domain."""
    deduped = _dedupe_records(records)
    record_index = _record_index(deduped)
    solvable = _v2_solvable_population(deduped, calibration=calibration, domains=domains)

    sns.set_theme(style="whitegrid")
    domain_order = list(domains)
    fig, axes = plt.subplots(
        1,
        max(len(domain_order), 1),
        figsize=(6.0 * max(len(domain_order), 1), 5.0),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    palette = sns.color_palette("tab10", n_colors=len(EXP3_V2_METHOD_LABELS))
    method_colors = {label: palette[i] for i, label in enumerate(EXP3_V2_METHOD_LABELS.values())}

    for col_idx, domain in enumerate(domain_order):
        ax = axes[col_idx]
        domain_pop = [r for r in solvable if r.get("domain") == domain]
        n = len(domain_pop)
        if n == 0:
            continue

        for method_id in EXP3_V2_PARAMETRIC_METHODS:
            label = EXP3_V2_METHOD_LABELS[method_id]
            points: list[tuple[float, float, float]] = []
            for theta in theta_grid:
                sizes: list[int] = []
                succ: list[float] = []
                for o in domain_pop:
                    mr = _lookup_record(
                        record_index,
                        o,
                        calibration=calibration,
                        method_id=method_id,
                        theta=theta,
                    )
                    if mr is None or mr.get("status") != "ok":
                        succ.append(0.0)
                        continue
                    succ.append(1.0 if bool(mr.get("success")) else 0.0)
                    if mr.get("belief_size") is not None:
                        sizes.append(int(mr["belief_size"]))
                if sizes and succ:
                    avg_size = sum(sizes) / len(sizes)
                    success_rate = 100.0 * sum(succ) / len(succ)
                    points.append((avg_size, success_rate, float(theta)))
            if points:
                points.sort(key=lambda t: t[0])
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=label, color=method_colors[label])

        # MAP single point at size = 1.
        map_succ = _v2_method_success_outcomes(
            domain_pop,
            record_index,
            calibration=calibration,
            method_id="map",
            theta=None,
        )
        if map_succ:
            ax.scatter(
                [1.0],
                [100.0 * sum(map_succ) / len(map_succ)],
                marker="X",
                s=130,
                color=method_colors["MAP"],
                label="MAP",
                zorder=5,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Average belief-set size (log scale)")
        if col_idx == 0:
            ax.set_ylabel("Planning success rate (%)")
        ax.set_title(f"{domain.upper()} (n={n})")
        ax.set_ylim(0, 105)
        ax.grid(True, which="both", alpha=0.3)
        if col_idx == len(domain_order) - 1:
            ax.legend(loc="best", fontsize="small")

    fig.suptitle("Cost–quality frontier: planning success vs belief-set size", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def exp3_decision_quality_figure(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domains: Sequence[str] = ("prb", "gw"),
    theta: float = EXP3_DEFAULT_THETA,
) -> Figure:
    """Stacked bar chart of decision-outcome buckets per method, per domain."""
    deduped = _dedupe_records(records)
    record_index = _record_index(deduped)
    solvable = _v2_solvable_population(deduped, calibration=calibration, domains=domains)

    sns.set_theme(style="whitegrid")
    domain_order = list(domains)
    fig, axes = plt.subplots(
        1,
        max(len(domain_order), 1),
        figsize=(6.0 * max(len(domain_order), 1), 5.0),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    bucket_colors = {
        "Successful plan": "#2ca02c",
        "Failed commitment": "#d62728",
        "Abstained (no plan)": "#7f7f7f",
    }
    bucket_order = ["Successful plan", "Failed commitment", "Abstained (no plan)"]
    methods_order = list(EXP3_V2_METHOD_LABELS.values())

    for col_idx, domain in enumerate(domain_order):
        ax = axes[col_idx]
        domain_pop = [r for r in solvable if r.get("domain") == domain]
        n = len(domain_pop)
        if n == 0:
            continue

        method_buckets: dict[str, dict[str, float]] = {}
        for method_id, label in EXP3_V2_METHOD_LABELS.items():
            method_theta = _v2_method_theta(method_id, theta)
            counts = {b: 0 for b in bucket_order}
            for o in domain_pop:
                mr = _lookup_record(
                    record_index,
                    o,
                    calibration=calibration,
                    method_id=method_id,
                    theta=method_theta,
                )
                if mr is None or mr.get("status") != "ok" or mr.get("plan_length") is None:
                    counts["Abstained (no plan)"] += 1
                elif bool(mr.get("success")):
                    counts["Successful plan"] += 1
                else:
                    counts["Failed commitment"] += 1
            method_buckets[label] = {b: 100.0 * c / n for b, c in counts.items()}

        x = np.arange(len(methods_order), dtype=float)
        bottoms = np.zeros(len(methods_order))
        for bucket in bucket_order:
            heights = np.array([method_buckets[m][bucket] for m in methods_order])
            ax.bar(
                x,
                heights,
                width=0.6,
                bottom=bottoms,
                color=bucket_colors[bucket],
                label=bucket if col_idx == 0 else None,
                edgecolor="white",
                linewidth=0.6,
            )
            bottoms += heights
        ax.set_xticks(x)
        ax.set_xticklabels(methods_order, rotation=20, ha="right")
        ax.set_title(f"{domain.upper()} (n={n})")
        ax.set_ylim(0, 105)
        if col_idx == 0:
            ax.set_ylabel("Fraction of solvable instances (%)")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Decision-quality decomposition (θ={theta:.2f} for parametric methods)",
        fontsize=13,
    )
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize="small")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig


__all__ = [
    "EXP3_DEFAULT_THETA",
    "EXP3_PREFIX_THETA",
    "EXP3_PLANNING_METHOD_ORDER",
    "EXP3_V2_METHOD_LABELS",
    "EXP3_V2_PARAMETRIC_METHODS",
    "EXP3_V2_FIXED_METHODS",
    "EXP3_V2_THETA_GRID",
    "exp3_prefix_search_success_matrix",
    "exp3_adaptive_success_matrix",
    "exp3_planning_horizon_figure",
    "exp3_planning_success_table",
    "exp3_recovery_retention_summary",
    "exp3_recovery_retention_table",
    "exp3_plan_found_retention_summary",
    "exp3_plan_found_retention_table",
    "exp3_recovery_retention_figure",
    "exp3_recovery_vs_theta_figure",
    "exp3_cost_quality_figure",
    "exp3_decision_quality_figure",
    "parse_goal_horizon",
    "parse_object_count",
]
