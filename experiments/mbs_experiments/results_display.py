from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns



EXP1_THETA_METHODS = [
    "mbs_prevas",
    # "unconstrained_top_mass",
    "random_theta_confident",
    "fluent_abstention_unconstrained",
]
EXP1_POINT_METHODS = ["map"]#, "valid_map"]
EXP1_SIZED_POINT_METHODS = ["mbs_first", "mbs_first_constraints"]
EXP1_COMMON_SUPPORT_EXCLUDED = {"random_theta_confident"}
EXP1_METHOD_LABELS = {
    "mbs_prevas": "MBSθ (Ours)",
    # "unconstrained_top_mass": "MBS (unconstrained)",
    "random_theta_confident": "Random θ-Mass",
    "fluent_abstention_unconstrained": "Feature Abstention",
    "map": "MAP",
    # "valid_map": "Valid MAP",
    "mbs_first": "MBS-First",
    "mbs_first_constraints": "MBS-First (constraints)",
}
EXP1_PLOT_STYLES = {
    "MBSθ (Ours)": {"marker": "o"},
    # "MBS (unconstrained)": {"marker": "^"},
    "Random θ-Mass": {"marker": "v"},
    "Feature Abstention": {"marker": "P"},
    "MAP": {"linestyle": ":"},
    # "Valid MAP": {"linestyle": "--"},
    "MBS-First": {"marker": "D"},
    "MBS-First (constraints)": {"marker": "X"},
}

EXP2_PREVAS_LABEL = "PRaVaSS"
EXP2_EXHAUSTIVE_LABEL = "Exhaustive"


def _bootstrap_seed(label: str) -> int:
    digest = hashlib.sha256(f"results-display|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _bootstrap_ci(
    values: Sequence[float],
    *,
    label: str,
    statistic: str,
    resamples: int,
) -> tuple[float | None, float | None, float | None]:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return None, None, None

    arr = np.array(clean, dtype=float)
    if statistic == "mean":
        observed = float(np.mean(arr))
        stat_fn = np.mean
    elif statistic == "median":
        observed = float(np.median(arr))
        stat_fn = np.median
    else:
        raise ValueError(f"Unsupported bootstrap statistic: {statistic}")

    if len(arr) == 1 or resamples <= 0:
        return observed, observed, observed

    rng = np.random.default_rng(_bootstrap_seed(label))
    stats = np.empty(resamples, dtype=float)
    for i in range(resamples):
        stats[i] = stat_fn(rng.choice(arr, size=len(arr), replace=True))
    low, high = np.quantile(stats, [0.025, 0.975])
    return observed, float(low), float(high)


def _bootstrap_ratio_of_means_ci(
    num_values: Sequence[float],
    den_values: Sequence[float],
    *,
    label: str,
    resamples: int,
) -> tuple[float | None, float | None, float | None]:
    if len(num_values) == 0 or len(den_values) == 0 or len(num_values) != len(den_values):
        return None, None, None
    
    num_arr = np.array(num_values, dtype=float)
    den_arr = np.array(den_values, dtype=float)
    
    obs_num = np.mean(num_arr)
    obs_den = np.mean(den_arr)
    if obs_den == 0:
        return None, None, None
    observed = float(obs_num / obs_den)
    
    if len(num_arr) == 1 or resamples <= 0:
        return observed, observed, observed

    rng = np.random.default_rng(_bootstrap_seed(label))
    stats = np.empty(resamples, dtype=float)
    n = len(num_arr)
    for i in range(resamples):
        indices = rng.choice(n, size=n, replace=True)
        d = np.mean(den_arr[indices])
        stats[i] = np.mean(num_arr[indices]) / d if d != 0 else np.nan
        
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return observed, np.nan, np.nan
    low, high = np.quantile(stats, [0.025, 0.975])
    return observed, float(low), float(high)


def _instance_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("domain"),
        record.get("configuration"),
        record.get("sample"),
        record.get("split_relative_path"),
    )


def _format_theta(theta: Any) -> str:
    if theta is None or pd.isna(theta):
        return "point"
    return f"{float(theta):.2f}"


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _records_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    if "compact_set_size" in df.columns:
        set_size = df["compact_set_size"]
    elif "full_set_size" in df.columns:
        set_size = df["full_set_size"]
    else:
        set_size = pd.Series(np.nan, index=df.index)
    if "compact_set_size" in df.columns:
        set_size = set_size.fillna(df["compact_set_size"])
    if "set_size" in df.columns:
        set_size = set_size.fillna(df["set_size"])
    df = df.copy()
    df["_set_size"] = set_size.fillna(0).astype(float)
    if "contains_truth" in df.columns:
        df["_coverage"] = df["contains_truth"].fillna(False).astype(float)
    else:
        df["_coverage"] = 0.0
    if "constrained_mass" in df.columns:
        df["_mass"] = df["constrained_mass"].fillna(0.0).astype(float)
    else:
        df["_mass"] = 0.0
    return df


def exp1_coverage_size_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    bootstrap_resamples: int = 2_000,
) -> pd.DataFrame:
    df = _records_frame(records)
    columns = [
        "Domain",
        "Method",
        "Method ID",
        "Theta",
        "Theta Value",
        "N",
        "Coverage",
        "Coverage CI Low",
        "Coverage CI High",
        "Average Set Size",
        "Average Set Size CI Low",
        "Average Set Size CI High",
        "Average Mass",
        "Average Mass CI Low",
        "Average Mass CI High",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    method_ids = EXP1_THETA_METHODS + EXP1_POINT_METHODS# + EXP1_SIZED_POINT_METHODS
    df = df[
        df.get("calibration", pd.Series(index=df.index, dtype=object)).eq(calibration)
        & df.get("method", pd.Series(index=df.index, dtype=object)).isin(method_ids)
        & df.get("status", pd.Series(index=df.index, dtype=object)).eq("ok")
    ].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []

    theta_df = df[df["method"].isin(EXP1_THETA_METHODS)].copy()
    point_df = df[df["method"].isin(EXP1_POINT_METHODS)].copy()
    domain_common_support = _exp1_comparable_common_support(theta_df, point_df)
    if not theta_df.empty:
        theta_df["_instance_key"] = theta_df.apply(_instance_key, axis=1)
        for (domain, method_id), method_df in theta_df.groupby(
            ["domain", "method"], sort=False
        ):
            method_df = method_df.sort_values(["theta", "_instance_key"])
            theta_values = sorted(float(theta) for theta in method_df["theta"].unique())
            if method_id in EXP1_COMMON_SUPPORT_EXCLUDED:
                support_by_theta = {
                    theta: set(
                        method_df.loc[method_df["theta"].eq(theta), "_instance_key"]
                    )
                    for theta in theta_values
                }
                common_support = set.intersection(*support_by_theta.values())
            else:
                common_support = domain_common_support.get(domain, set())
            if not common_support:
                continue

            method_df = method_df[method_df["_instance_key"].isin(common_support)]
            method_df = method_df.drop_duplicates(
                ["theta", "_instance_key"], keep="first"
            )
            for theta, group in method_df.groupby("theta", sort=True):
                rows.append(
                    _exp1_summary_row(
                        domain=domain,
                        method_id=method_id,
                        theta=float(theta),
                        group=group,
                        bootstrap_resamples=bootstrap_resamples,
                    )
                )

    if not point_df.empty:
        point_df["_instance_key"] = point_df.apply(_instance_key, axis=1)
        point_df = point_df.sort_values(["domain", "method", "theta"]).drop_duplicates(
            ["domain", "method", "_instance_key"], keep="first"
        )
        for (domain, method_id), group in point_df.groupby(
            ["domain", "method"], sort=False
        ):
            if method_id not in EXP1_COMMON_SUPPORT_EXCLUDED:
                group = group[
                    group["_instance_key"].isin(domain_common_support.get(domain, set()))
                ]
            if group.empty:
                continue
            rows.append(
                _exp1_summary_row(
                    domain=domain,
                    method_id=method_id,
                    theta=None,
                    group=group,
                    bootstrap_resamples=bootstrap_resamples,
                )
            )

    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out

    method_rank = {label: i for i, label in enumerate(EXP1_METHOD_LABELS.values())}
    out["_method_rank"] = out["Method"].map(method_rank)
    out["_theta_rank"] = out["Theta Value"].fillna(np.inf)
    out = out.sort_values(["Domain", "_method_rank", "_theta_rank"]).drop(
        columns=["_method_rank", "_theta_rank"]
    )
    return out.reset_index(drop=True)


def _exp1_comparable_common_support(
    theta_df: pd.DataFrame, point_df: pd.DataFrame
) -> dict[str, set[tuple[Any, ...]]]:
    comparable_methods = [
        method
        for method in EXP1_THETA_METHODS + EXP1_POINT_METHODS
        if method not in EXP1_COMMON_SUPPORT_EXCLUDED
    ]
    supports_by_domain: dict[str, list[set[tuple[Any, ...]]]] = {}

    if not theta_df.empty:
        theta_df = theta_df[theta_df["method"].isin(comparable_methods)].copy()
        if not theta_df.empty:
            theta_df["_instance_key"] = theta_df.apply(_instance_key, axis=1)
            for (domain, _method, _theta), group in theta_df.groupby(
                ["domain", "method", "theta"], sort=False
            ):
                supports_by_domain.setdefault(domain, []).append(
                    set(group["_instance_key"])
                )

    if not point_df.empty:
        point_df = point_df[point_df["method"].isin(comparable_methods)].copy()
        if not point_df.empty:
            point_df["_instance_key"] = point_df.apply(_instance_key, axis=1)
            for (domain, _method, _theta), group in point_df.groupby(
                ["domain", "method", "theta"], sort=False
            ):
                supports_by_domain.setdefault(domain, []).append(
                    set(group["_instance_key"])
                )

    return {
        domain: set.intersection(*supports)
        for domain, supports in supports_by_domain.items()
        if supports
    }


def _exp1_summary_row(
    *,
    domain: str,
    method_id: str,
    theta: float | None,
    group: pd.DataFrame,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    coverage, cov_low, cov_high = _bootstrap_ci(
        group["_coverage"].tolist(),
        label=f"exp1|{domain}|{method_id}|{theta}|coverage",
        statistic="mean",
        resamples=bootstrap_resamples,
    )
    size, size_low, size_high = _bootstrap_ci(
        group["_set_size"].tolist(),
        label=f"exp1|{domain}|{method_id}|{theta}|size",
        statistic="mean",
        resamples=bootstrap_resamples,
    )
    mass, mass_low, mass_high = _bootstrap_ci(
        group["_mass"].tolist(),
        label=f"exp1|{domain}|{method_id}|{theta}|mass",
        statistic="mean",
        resamples=bootstrap_resamples,
    )
    return {
        "Domain": domain,
        "Method": EXP1_METHOD_LABELS[method_id],
        "Method ID": method_id,
        "Theta": _format_theta(theta),
        "Theta Value": theta,
        "N": int(len(group)),
        "Coverage": coverage,
        "Coverage CI Low": cov_low,
        "Coverage CI High": cov_high,
        "Average Set Size": size,
        "Average Set Size CI Low": size_low,
        "Average Set Size CI High": size_high,
        "Average Mass": mass,
        "Average Mass CI Low": mass_low,
        "Average Mass CI High": mass_high,
    }


def exp1_coverage_size_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    display_summary = summary.copy()
    display_summary["Coverage"] = display_summary.apply(
        lambda row: _format_ci_value(
            row["Coverage"],
            row["Coverage CI Low"],
            row["Coverage CI High"],
            precision=4,
        ),
        axis=1,
    )
    display_summary["Average Set Size"] = display_summary.apply(
        lambda row: _format_ci_value(
            row["Average Set Size"],
            row["Average Set Size CI Low"],
            row["Average Set Size CI High"],
            precision=2,
        ),
        axis=1,
    )
    display_summary["Average Mass"] = display_summary.apply(
        lambda row: _format_ci_value(
            row["Average Mass"],
            row["Average Mass CI Low"],
            row["Average Mass CI High"],
            precision=2,
        ),
        axis=1,
    )
    display_summary["mass"] = display_summary["Theta"].replace("point", "Point").str.replace("0.", "θ=", regex=False)
    table = display_summary.pivot_table(
        index=["Theta", "Method"],
        columns="Domain",
        values=["Coverage", "Average Set Size", "Average Mass"],
        aggfunc="first",
        sort=False,
    )
    table = table.swaplevel(0, 1, axis=1).sort_index(
        axis=1, level=0, sort_remaining=False
    )
    return table


def _format_ci_value(value: Any, ci_low: Any, ci_high: Any, *, precision: int) -> str:
    import math
    precision = min(precision, 2)
    def _fmt(val: Any) -> str:
        if pd.isna(val):
            return ""
        v = float(val)
        if v == 0:
            return f"{0:.{precision}f}"
        abs_v = abs(v)
        if abs_v >= 1e5 or (abs_v < 1e-2 and abs_v > 0):
            exp = int(math.floor(math.log10(abs_v)))
            base = v / (10**exp)
            return f"{base:.{precision}f}x10^{exp}"
        else:
            return f"{v:.{precision}f}"

    if pd.isna(value):
        return ""
    val_str = _fmt(value)
    if pd.isna(ci_low) or pd.isna(ci_high):
        return val_str
    return (
        f"{val_str} "
        f"[{_fmt(ci_low)}, {_fmt(ci_high)}]"
    )


def coverage_size_figures(summary: pd.DataFrame) -> dict[str, Figure]:
    figures: dict[str, Figure] = {}
    if summary.empty:
        return figures

    sns.set_theme(style="whitegrid")
    palette = dict(
        zip(
            EXP1_METHOD_LABELS.values(),
            sns.color_palette("tab10", n_colors=len(EXP1_METHOD_LABELS)),
            strict=False,
        )
    )
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    axes = {
        "prb": ax1,
        "gw": ax2,
        "alf": ax3,
    }


    for domain in _ordered_unique(summary["Domain"]):
        domain_df = summary[summary["Domain"].eq(domain)]
        ax = axes.get(domain)

        for method_id in EXP1_THETA_METHODS:
            if domain == 'gw' and method_id == 'random_theta_confident':
                continue  # This method times out on most GW instances
            # if domain == "prb" and method_id == "fluent_abstention_unconstrained":
            #     continue  # This method has very large sets on prb, making the plot hard to read
            
            method = EXP1_METHOD_LABELS[method_id]
            method_df = domain_df[
                domain_df["Method"].eq(method) & domain_df["Theta Value"].notna()
            ].sort_values("Theta Value")
            if method_df.empty:
                continue
            style = EXP1_PLOT_STYLES[method]
            ax.plot(
                method_df["Average Set Size"],
                method_df["Coverage"],
                marker=style["marker"],
                linewidth=1.8,
                label=method,
                color=palette[method],
            )

        # for method_id in EXP1_SIZED_POINT_METHODS:
        #     method = EXP1_METHOD_LABELS[method_id]
        #     method_df = domain_df[
        #         domain_df["Method"].eq(method) & domain_df["Theta Value"].notna()
        #     ].sort_values("Theta Value")
        #     if method_df.empty:
        #         continue
        #     style = EXP1_PLOT_STYLES[method]
        #     ax.scatter(
        #         method_df["Average Set Size"],
        #         method_df["Coverage"],
        #         marker=style["marker"],
        #         # linewidth=1.8,
        #         label=method,
        #         color=palette[method],
        #     )

        for method_id in EXP1_POINT_METHODS:
            method = EXP1_METHOD_LABELS[method_id]
            method_df = domain_df[domain_df["Method"].eq(method)]
            if method_df.empty:
                continue
            row = method_df.iloc[0]
            style = EXP1_PLOT_STYLES[method]
            ax.axhline(
                float(row["Coverage"]),
                linestyle=style["linestyle"],
                linewidth=1.7,
                color=palette[method],
                label=f"{method} coverage",
            )

        positive_sizes = domain_df.loc[
            domain_df["Average Set Size"].gt(0), "Average Set Size"
        ]
        if not positive_sizes.empty:
            ax.set_xscale("log")
            # ax.set_xlim(
            #     max(float(positive_sizes.min()) * 0.7, 1e-6),
            #     float(positive_sizes.max()) * 1.35,
            # )
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Average Set Size (log scale)")
        if domain == 'prb':
            ax.set_ylabel("True State Coverage")
        ax.set_title(f"{str(domain).upper()}")
        ax.grid(True, which="both", alpha=0.3)
        if domain == "alf":
            ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        figures[str(domain)] = fig

    return {'all': fig}


def merged_coverage_size_mass_figures(summary: pd.DataFrame) -> dict[str, Figure]:
    figures: dict[str, Figure] = {}
    if summary.empty:
        return figures

    sns.set_theme(style="whitegrid")
    palette = dict(
        zip(
            EXP1_METHOD_LABELS.values(),
            sns.color_palette("tab10", n_colors=len(EXP1_METHOD_LABELS)),
            strict=False,
        )
    )

    domains = _ordered_unique(summary["Domain"])
    if not domains:
        return figures

    fig, axes = plt.subplots(2, len(domains), figsize=(5 * len(domains), 10), sharex="col")
    if len(domains) == 1:
        axes = np.array([axes]).T

    for i, domain in enumerate(domains):
        domain_df = summary[summary["Domain"].eq(domain)]
        
        # --- Top row: Size-Mass plot ---
        ax_top = axes[0, i]
        for method_id in EXP1_THETA_METHODS:
            if domain == 'gw' and method_id == 'random_theta_confident':
                continue
            
            method = EXP1_METHOD_LABELS[method_id]
            method_df = domain_df[
                domain_df["Method"].eq(method) & domain_df["Theta Value"].notna()
            ].sort_values("Theta Value")
            if method_df.empty:
                continue
            style = EXP1_PLOT_STYLES[method]
            ax_top.plot(
                method_df["Average Set Size"],
                method_df["Average Mass"],
                marker=style.get("marker", "o"),
                linewidth=1.8,
                label=method,
                color=palette[method],
            )

        for method_id in EXP1_POINT_METHODS:
            method = EXP1_METHOD_LABELS[method_id]
            method_df = domain_df[domain_df["Method"].eq(method)]
            if method_df.empty:
                continue
            row = method_df.iloc[0]
            style = EXP1_PLOT_STYLES[method]
            ax_top.axhline(
                float(row["Average Mass"]),
                linestyle=style.get("linestyle", "--"),
                linewidth=1.7,
                color=palette[method],
                label=f"{method}",
            )

        positive_sizes = domain_df.loc[
            domain_df["Average Set Size"].gt(0), "Average Set Size"
        ]
        if not positive_sizes.empty:
            ax_top.set_xscale("log")
            
        ax_top.set_ylim(0, 1.0)
        if i == 0:
            ax_top.set_ylabel("Set Probability Mass")
        ax_top.set_title(f"{str(domain).upper()}")
        ax_top.grid(True, which="both", alpha=0.3)
        if i == 0 and domain == "prb":
            ax_top.legend(loc="best", fontsize="small")
        elif i == 0:
            ax_top.legend(loc="best", fontsize="small")

        # --- Bottom row: Coverage-Size plot ---
        ax_bot = axes[1, i]
        for method_id in EXP1_THETA_METHODS:
            if domain == 'gw' and method_id == 'random_theta_confident':
                continue
            
            method = EXP1_METHOD_LABELS[method_id]
            method_df = domain_df[
                domain_df["Method"].eq(method) & domain_df["Theta Value"].notna()
            ].sort_values("Theta Value")
            if method_df.empty:
                continue
            style = EXP1_PLOT_STYLES[method]
            ax_bot.plot(
                method_df["Average Set Size"],
                method_df["Coverage"],
                marker=style.get("marker", "o"),
                linewidth=1.8,
                label=method,
                color=palette[method],
            )

        for method_id in EXP1_POINT_METHODS:
            method = EXP1_METHOD_LABELS[method_id]
            method_df = domain_df[domain_df["Method"].eq(method)]
            if method_df.empty:
                continue
            row = method_df.iloc[0]
            style = EXP1_PLOT_STYLES[method]
            ax_bot.axhline(
                float(row["Coverage"]),
                linestyle=style.get("linestyle", "--"),
                linewidth=1.7,
                color=palette[method],
                label=f"{method} coverage",
            )

        if not positive_sizes.empty:
            ax_bot.set_xscale("log")
            
        ax_bot.set_ylim(0, 1.0)
        ax_bot.set_xlabel("Average Set Size (log scale)")
        if i == 0:
            ax_bot.set_ylabel("Coverage")
        ax_bot.grid(True, which="both", alpha=0.3)
        # Note: the legend may not be needed in both plots if they share methods, 
        # but leaving it for "alf" domain similar to the original function for safety.
        if domain == "alf":
            ax_bot.legend(loc="best", fontsize="small")

    fig.tight_layout()
    figures['all'] = fig

    return figures


def exp2_records_for_domains(
    records: Iterable[Mapping[str, Any]],
    domains: Sequence[str] | None = None,
    *,
    calibration: str = "global",
) -> list[dict[str, Any]]:
    selected = []
    domain_set = None if domains is None else set(domains)
    for record in records:
        if record.get("calibration") != calibration:
            continue
        if domain_set is not None and record.get("domain") not in domain_set:
            continue
        selected.append(dict(record))
    return selected


def exp2_performance_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration: str = "global",
    domains: Sequence[str] | None = None,
    bootstrap_resamples: int = 2_000,
) -> pd.DataFrame:
    selected = exp2_records_for_domains(records, domains, calibration=calibration)
    columns = [
        "Method",
        "Theta",
        "Theta Value",
        "Total",
        "Solved",
        "Solved %",
        "Median Runtime",
        "Runtime CI Low",
        "Runtime CI High",
        "Median States Visited",
        "States CI Low",
        "States CI High",
        "Max Search Feature Count Solved",
    ]
    if not selected:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    prevas_records = [r for r in selected if r.get("method") == "prevas"]
    theta_values = sorted(
        {
            float(r["theta"])
            for r in prevas_records
            if isinstance(r.get("theta"), int | float)
        }
    )
    for theta in theta_values:
        group = [r for r in prevas_records if float(r.get("theta")) == theta]
        ok = [r for r in group if r.get("status") == "ok"]
        runtime_values = [
            float(r["runtime_seconds"]) for r in ok if r.get("runtime_seconds") is not None
        ]
        states_values = [
            float(r["states_scored"]) for r in ok if r.get("states_scored") is not None
        ]
        rows.append(
            _exp2_summary_row(
                method=EXP2_PREVAS_LABEL,
                theta=theta,
                total=len(group),
                solved=len(ok),
                runtime_values=runtime_values,
                states_values=states_values,
                max_search_feature_count_solved=_max_search_feature_count(ok),
                bootstrap_resamples=bootstrap_resamples,
                label=f"exp2|prevas|{theta}",
            )
        )

    exhaustive = _dedupe_exhaustive(selected)
    if exhaustive:
        ok_exhaustive = [r for r in exhaustive if r.get("status") == "ok"]
        runtime_values = [
            float(r["runtime_seconds"])
            for r in ok_exhaustive
            if r.get("runtime_seconds") is not None
        ]
        states_values = [
            float(2 ** int(r["search_feature_count"]))
            for r in ok_exhaustive
            if r.get("search_feature_count") is not None
        ]
        rows.append(
            _exp2_summary_row(
                method=EXP2_EXHAUSTIVE_LABEL,
                theta=1.0,
                total=len(exhaustive),
                solved=len(ok_exhaustive),
                runtime_values=runtime_values,
                states_values=states_values,
                max_search_feature_count_solved=_max_search_feature_count(ok_exhaustive),
                bootstrap_resamples=bootstrap_resamples,
                label="exp2|exhaustive",
            )
        )

    return pd.DataFrame(rows, columns=columns)


def _exp2_summary_row(
    *,
    method: str,
    theta: float,
    total: int,
    solved: int,
    runtime_values: Sequence[float],
    states_values: Sequence[float],
    max_search_feature_count_solved: int | None,
    bootstrap_resamples: int,
    label: str,
) -> dict[str, Any]:
    runtime, runtime_low, runtime_high = _bootstrap_ci(
        runtime_values,
        label=f"{label}|runtime",
        statistic="median",
        resamples=bootstrap_resamples,
    )
    states, states_low, states_high = _bootstrap_ci(
        states_values,
        label=f"{label}|states",
        statistic="median",
        resamples=bootstrap_resamples,
    )
    return {
        "Method": method,
        "Theta": _format_theta(theta),
        "Theta Value": theta,
        "Total": int(total),
        "Solved": int(solved),
        "Solved %": 100.0 * solved / total if total else np.nan,
        "Median Runtime": runtime,
        "Runtime CI Low": runtime_low,
        "Runtime CI High": runtime_high,
        "Median States Visited": states,
        "States CI Low": states_low,
        "States CI High": states_high,
        "Max Search Feature Count Solved": max_search_feature_count_solved,
    }


def _max_search_feature_count(records: Iterable[Mapping[str, Any]]) -> int | None:
    values = [
        int(record["search_feature_count"])
        for record in records
        if record.get("search_feature_count") is not None
    ]
    if not values:
        return None
    return max(values)


def _dedupe_exhaustive(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        if record.get("method") != "exhaustive":
            continue
        key = _instance_key(record)
        if key not in deduped:
            deduped[key] = dict(record)
    return list(deduped.values())


def exp2_performance_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    display_summary = summary.copy()
    display_summary["Median Runtime"] = display_summary.apply(
        lambda row: _format_ci_value(
            row["Median Runtime"],
            row["Runtime CI Low"],
            row["Runtime CI High"],
            precision=4,
        ),
        axis=1,
    )
    display_summary["Median States Visited"] = display_summary.apply(
        lambda row: _format_ci_value(
            row["Median States Visited"],
            row["States CI Low"],
            row["States CI High"],
            precision=2,
        ),
        axis=1,
    )
    metrics = [
        "Total",
        "Solved",
        "Solved %",
        "Max Search Feature Count Solved",
        "Median Runtime",
        "Median States Visited",
    ]
    return display_summary.set_index(["Method", "Theta"])[metrics]


def cactus_figures(
    records: Iterable[Mapping[str, Any]],
    *,
    domain_groups: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Figure]:
    records = list(records)
    if domain_groups is None:
        domains = sorted({str(r["domain"]) for r in records if r.get("domain") is not None})
        domain_groups = {domain: [domain] for domain in domains}
        if {"alf", "prb", "gw"}.issubset(domains):
            domain_groups = dict(domain_groups)
            domain_groups["All Domains"] = ["alf", "prb", "gw"]

    figures: dict[str, Figure] = {}
    for label, domains in domain_groups.items():
        group_records = [r for r in records if r.get("domain") in set(domains)]
        fig, ax = plt.subplots(figsize=(6, 5))
        exhaustive_times = sorted(_positive_values(_exhaustive_runtimes(group_records)))
        if exhaustive_times:
            ax.plot(
                range(1, len(exhaustive_times) + 1),
                exhaustive_times,
                color="black",
                linestyle="--",
                linewidth=1.9,
                label=EXP2_EXHAUSTIVE_LABEL,
            )

        theta_values = _prevas_theta_values(group_records)
        colors = sns.color_palette("tab10", n_colors=max(len(theta_values), 1))
        for theta, color in zip(theta_values, colors, strict=False):
            times = sorted(
                _positive_values(
                    [
                        float(r["runtime_seconds"])
                        for r in group_records
                        if r.get("method") == "prevas"
                        and r.get("status") == "ok"
                        and r.get("runtime_seconds") is not None
                        and float(r.get("theta")) == theta
                    ]
                )
            )
            if not times:
                continue
            ax.plot(
                range(1, len(times) + 1),
                times,
                color=color,
                linewidth=1.8,
                label=f"{EXP2_PREVAS_LABEL} theta={theta:.2f}",
            )

        ax.set_yscale("log")
        ax.set_xlabel("Instances solved")
        ax.set_ylabel("Median Runtime (seconds, log scale)")
        # ax.set_title(f"{label.upper() if label != 'All Domains' else label}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        figures[label] = fig

    return figures


def sfc_runtime_figure(
    records: Iterable[Mapping[str, Any]],
    *,
    domains: Sequence[str] = ("alf", "prb"),
) -> Figure:
    return _sfc_metric_figure(
        records,
        domains=domains,
        metric="runtime",
        ylabel="Median Runtime (seconds, log scale)",
        title="",
    )


def sfc_states_figure(
    records: Iterable[Mapping[str, Any]],
    *,
    domains: Sequence[str] = ("alf", "prb"),
) -> Figure:
    return _sfc_metric_figure(
        records,
        domains=domains,
        metric="states",
        ylabel="States visited (log scale)",
        title="",
    )


def _sfc_metric_figure(
    records: Iterable[Mapping[str, Any]],
    *,
    domains: Sequence[str],
    metric: str,
    ylabel: str,
    title: str,
) -> Figure:
    domain_set = set(domains)
    records = [r for r in records if r.get("domain") in domain_set]
    fig, ax = plt.subplots(figsize=(6, 5))

    exhaustive_bins: dict[int, list[float]] = {}
    for record in _dedupe_exhaustive(records):
        if record.get("status") != "ok" or record.get("search_feature_count") is None:
            continue
        value = _record_metric_value(record, metric, exhaustive=True)
        if value is None:
            continue
        exhaustive_bins.setdefault(int(record["search_feature_count"]), []).append(value)
    _plot_sfc_bins(
        ax,
        exhaustive_bins,
        color="black",
        linestyle="--",
        label=EXP2_EXHAUSTIVE_LABEL,
    )

    theta_values = _prevas_theta_values(records)
    colors = sns.color_palette("tab10", n_colors=max(len(theta_values), 1))
    for theta, color in zip(theta_values, colors, strict=False):
        bins: dict[int, list[float]] = {}
        for record in records:
            if (
                record.get("method") != "prevas"
                or record.get("status") != "ok"
                or record.get("search_feature_count") is None
                or float(record.get("theta")) != theta
            ):
                continue
            value = _record_metric_value(record, metric, exhaustive=False)
            if value is None:
                continue
            bins.setdefault(int(record["search_feature_count"]), []).append(value)
        _plot_sfc_bins(
            ax,
            bins,
            color=color,
            linestyle="-",
            label=f"{EXP2_PREVAS_LABEL} theta={theta:.2f}",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Number of search features")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig


def _plot_sfc_bins(
    ax: plt.Axes,
    bins: Mapping[int, Sequence[float]],
    *,
    color: Any,
    linestyle: str,
    label: str,
) -> None:
    if not bins:
        return
    sfcs = sorted(bins)
    values = [float(np.median(_positive_values(bins[sfc]))) for sfc in sfcs]
    ax.plot(sfcs, values, color=color, linestyle=linestyle, linewidth=1.8, label=label)


FigureCollection: TypeAlias = Mapping[str, Figure] | Sequence[Figure] | Figure


def show_figures(figures: FigureCollection) -> FigureCollection:
    """Display figures and return the original figure object/container.

    Returning the input lets notebook callers keep access to generated figures, e.g.
    ``exp1_figures = show_figures(coverage_size_figures(exp1_summary))``.
    """
    show_inline = plt.get_backend().lower() != "agg"
    if isinstance(figures, Figure):
        plt.figure(figures.number)
        if show_inline:
            plt.show()
        plt.close(figures)
        return figures

    if isinstance(figures, Mapping):
        iterable = list(figures.values())
    else:
        iterable = list(figures)

    for fig in iterable:
        plt.figure(fig.number)
    if show_inline:
        plt.show()
    for fig in iterable:
        plt.close(fig)
    return figures


def _exhaustive_runtimes(records: Iterable[Mapping[str, Any]]) -> list[float]:
    return [
        float(r["runtime_seconds"])
        for r in _dedupe_exhaustive(records)
        if r.get("status") == "ok" and r.get("runtime_seconds") is not None
    ]


def _prevas_theta_values(records: Iterable[Mapping[str, Any]]) -> list[float]:
    return sorted(
        {
            float(r["theta"])
            for r in records
            if r.get("method") == "prevas"
            and isinstance(r.get("theta"), int | float)
        }
    )


def _record_metric_value(
    record: Mapping[str, Any], metric: str, *, exhaustive: bool
) -> float | None:
    if metric == "runtime":
        value = record.get("runtime_seconds")
    elif metric == "states":
        if exhaustive:
            sfc = record.get("search_feature_count")
            value = None if sfc is None else 2 ** int(sfc)
        else:
            value = record.get("states_scored")
    else:
        raise ValueError(f"Unsupported SFC metric: {metric}")
    if value is None or pd.isna(value):
        return None
    return max(float(value), 1e-12)


def _positive_values(values: Iterable[Any]) -> list[float]:
    return [max(float(v), 1e-12) for v in values if v is not None and pd.notna(v)]
