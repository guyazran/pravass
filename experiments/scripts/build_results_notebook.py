#!/usr/bin/env python3
"""Generate ``results.ipynb``, which renders every figure and table of the paper.

The notebook has two parts: the results of the submitted paper, rendered with
the same functions and settings as the paper, and the results added during the
review (more VLM backbones, a constraint ablation, a reliability view, a
closed-loop experiment and a correlated-noise study). Run from the experiments
directory::

    uv run scripts/build_results_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace results.ipynb
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    text = textwrap.dedent(text)
    cells.append(
        {
            "cell_type": "markdown",
            "id": f"cell-{len(cells):02d}",
            "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


def code(text: str) -> None:
    text = textwrap.dedent(text)
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": f"cell-{len(cells):02d}",
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


def build() -> None:
    """Append every cell of the notebook to ``cells``."""
    # ---------------------------------------------------------------------------
    md(
        r"""
    # Results

    Every figure and table of *Robust Task-Aware State Estimation from Pretrained
    Perception* (Azran and Keren, NeurIPS 2026), rendered from the experiment outputs in
    `experiments_data/`. Figures are saved to `figures/` as PDF and tables as CSV.

    The notebook has two parts:

    1. **Paper.** The results of Section 4 and the appendix, with GPT-4.1 as the VLM,
       rendered with the functions and settings that produced the paper.
    2. **Review additions.** Results added during the review: five more VLM backbones,
       a constraint ablation, a reliability view of θ, explored-state counts, a
       closed-loop experiment on ALFWorld, and a correlated-noise study.

    Confidence intervals are 95% bootstrap intervals over test instances with 10,000
    resamples, except the six-backbone and ablation planning tables (Wilson score
    intervals) and Experiments 6 and 7 (1,000 resamples, computed by their runners). The
    first run streams the multi-GB Experiment 1 and 3 outputs once and caches small
    summaries next to them; it can take an hour. Later runs are fast.
    """
    )

    code(
        r"""
    %matplotlib inline
    from pathlib import Path

    import pandas as pd

    from mbs_experiments import exp3_display as e3d
    from mbs_experiments import notebook_data as nd
    from mbs_experiments import paper_extras as px
    from mbs_experiments import results_display as rd

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 220)

    ROOT = Path.cwd()
    FIGURES = ROOT / "figures"
    FIGURES.mkdir(exist_ok=True)
    DATA = ROOT / "experiments_data"

    PAPER_MODEL = "OpenAI__gpt-4.1-2025-04-14"   # the paper's VLM
    CALIBRATION = "global"                       # Platt scaling of all features of a domain
    RESAMPLES = 10_000                           # bootstrap resamples for every interval
    PLANNING_THETA = 0.5                         # θ of the fixed-θ methods in Experiment 3


    def exp_records(experiment: str, model: str = PAPER_MODEL) -> Path:
        return DATA / experiment / model / "sample_results.jsonl"


    def save_table(table: pd.DataFrame, name: str, **kwargs) -> pd.DataFrame:
        table.to_csv(FIGURES / name, **kwargs)
        return table
    """
    )

    # =========================================================================== Paper
    md(
        r"""
    # Part 1: Paper

    ## Experiment 1: belief-set quality (Fig. 5, Table 3, Fig. 10)

    True-state coverage, average set size and average belief mass of each belief-set
    construction, for θ from 0.01 to 0.99. Averages are taken over the instances that every
    method solves at every θ, so the curves are comparable. Random θ-Mass uses its own
    support, since it times out on most GW instances.
    """
    )
    code(
        r"""
    EXP1_RECORDS = px.load_records(
        exp_records("experiment_1_static_quality"),
        fields=px.EXP1_SUMMARY_FIELDS,
        calibration=CALIBRATION,
    )
    EXP1_SUMMARY = rd.exp1_coverage_size_summary(
        EXP1_RECORDS, calibration=CALIBRATION, bootstrap_resamples=RESAMPLES
    )
    PAPER_EXP1_METHODS = ["MBSθ (Ours)", "Random θ-Mass", "Feature Abstention", "MAP"]
    EXP1_PAPER = EXP1_SUMMARY[EXP1_SUMMARY["Method"].isin(PAPER_EXP1_METHODS)]
    print(f"{len(EXP1_RECORDS):,} Experiment 1 records")
    """
    )

    md("### Table 3: coverage, set size and mass for every θ")
    code(
        r"""
    save_table(rd.exp1_coverage_size_table(EXP1_PAPER), "paper-table3-exp1.csv")
    """
    )

    md("### Fig. 5: coverage vs. average set size")
    code(
        r"""
    fig = rd.coverage_size_figures(EXP1_PAPER)["all"]
    nd.save(fig, "paper-fig5-coverage-size.pdf")
    nd.show(fig)
    """
    )

    md("### Fig. 10: set size vs. belief mass and coverage")
    code(
        r"""
    fig = rd.merged_coverage_size_mass_figures(EXP1_PAPER)["all"]
    nd.save(fig, "paper-fig10-size-mass-coverage.pdf")
    nd.show(fig)
    """
    )

    md(
        r"""
    ### Numbers quoted in the text

    The fraction of GW instances on which Random θ-Mass finishes within its time limit at
    every θ, and the size of Random θ-Mass relative to MBSθ at θ = 0.9 (from Table 3).
    """
    )
    code(
        r"""
    random_gw = {}
    for r in EXP1_RECORDS:
        if r["method"] == "random_theta_confident" and r["domain"] == "gw":
            ok = random_gw.get(r["split_relative_path"], True)
            random_gw[r["split_relative_path"]] = ok and r["status"] == "ok"
    finished = sum(random_gw.values()) / len(random_gw)
    print(f"Random θ-Mass finishes at every θ on {finished:.1%} of {len(random_gw)} GW instances")

    at_09 = EXP1_PAPER[EXP1_PAPER["Theta Value"] == 0.9].set_index(["Domain", "Method"])["Average Set Size"]
    for domain in ("prb", "alf"):
        ratio = at_09[(domain, "Random θ-Mass")] / at_09[(domain, "MBSθ (Ours)")]
        print(f"{domain}: Random θ-Mass is {ratio:.1f}x the size of MBSθ at θ = 0.9")
    """
    )

    md(
        r"""
    ## Experiment 2: PRaVaSS scaling (Table 4, Fig. 6, Fig. 11)

    PRaVaSS against exhaustive enumeration, with a 10 s budget per instance. Runtimes
    depend on the hardware; the counts of explored states do not.
    """
    )
    code(
        r"""
    EXP2_RECORDS = px.load_records(exp_records("experiment_2_prevas_scaling"))
    EXP2 = rd.exp2_records_for_domains(EXP2_RECORDS, calibration=CALIBRATION)
    EXP2_SUMMARY = rd.exp2_performance_summary(EXP2, calibration=CALIBRATION, bootstrap_resamples=RESAMPLES)
    print(f"{len(EXP2):,} Experiment 2 records")
    """
    )

    md("### Table 4: solved instances, largest instance solved, runtime and explored states")
    code(
        r"""
    save_table(rd.exp2_performance_table(EXP2_SUMMARY), "paper-table4-exp2.csv")
    """
    )

    md(
        r"""
    ### How many more features than exhaustive enumeration (Sec. 4, App. J)

    The largest number of search features solved at each θ, relative to exhaustive
    enumeration (the 154% and 42% figures).
    """
    )
    code(
        r"""
    features = EXP2_SUMMARY.set_index("Method")[["Theta", "Max Search Feature Count Solved"]]
    exhaustive = features.loc["Exhaustive", "Max Search Feature Count Solved"]
    for theta, count in features.drop("Exhaustive").itertuples(index=False):
        print(f"PRaVaSS θ = {theta}: {count} features, {count / exhaustive - 1:+.0%} vs. exhaustive ({exhaustive})")
    """
    )

    md("### Fig. 6: instances solved within a time budget")
    code(
        r"""
    fig = rd.cactus_figures(EXP2)["All Domains"]
    nd.save(fig, "paper-fig6-cactus.pdf")
    nd.show(fig)
    """
    )

    md("### Fig. 11: runtime and explored states by number of search features")
    code(
        r"""
    for name, fig in [("paper-fig11a-runtime-by-features.pdf", rd.sfc_runtime_figure(EXP2)),
                      ("paper-fig11b-states-by-features.pdf", rd.sfc_states_figure(EXP2))]:
        nd.save(fig, name)
        nd.show(fig)
    """
    )

    md(
        r"""
    ## Experiment 3: planning with belief sets (Tables 1, 5 and 6, Figs. 7, 12 and 13)

    A robust planner (CPOR) plans for every state of each method's belief set; success means
    that the plan reaches the goal from the true state. **AMS** searches for the largest θ
    whose belief set admits a plan. *AMS (Ours)* uses all extracted mutex groups with the
    exact normalizer, the primary normalizer of App. G.5 (it stays within its scope bound on
    every test instance); *AMS-Loose* uses only the disjoint selected groups. Adaptive Abstention searches for the largest τ whose Feature
    Abstention set admits a plan. The fixed-θ methods use θ = 0.5; *MBSθ (Ours)* and
    *MBSθ-Loose (Ours)* use the same two constraint sets as AMS and AMS-Loose.

    The labels follow Tables 1, 5 and 6. The submitted version's Figs. 12 and 13 and App. K
    call AMS-Loose "AMS" and AMS "AMS-strict" (likewise MBSθ-Loose and MBSθ-strict).
    """
    )
    code(
        r"""
    EXP3_RECORDS = nd.load_exp3_records(PAPER_MODEL)
    print(f"{len(EXP3_RECORDS):,} Experiment 3 records")
    """
    )

    md(
        r"""
    ### Tables 1 and 5: planning success

    Table 5 lists MAP, Adaptive Abstention, AMS-Loose and AMS; Table 1 omits AMS-Loose.
    """
    )
    code(
        r"""
    EXP3_SUCCESS = e3d.exp3_planning_success_table(
        EXP3_RECORDS, calibration=CALIBRATION, domains=("prb", "gw"),
        theta=PLANNING_THETA, bootstrap_resamples=RESAMPLES,
    )
    save_table(EXP3_SUCCESS, "paper-table5-exp3-success.csv")
    """
    )
    code(
        r"""
    # Relative improvements quoted in Sec. 4 and App. K.
    rates = {
        domain: EXP3_SUCCESS[(domain, "Success")].str.extract(r"^([\d.]+)%")[0].astype(float)
        for domain in ("prb", "gw")
    }
    for domain, rate in rates.items():
        for method in ("AMS (Ours)", "AMS-Loose (Ours)"):
            for baseline in ("MAP", "Adaptive Abstention"):
                print(f"{domain}: {method} vs {baseline}: {rate[method] / rate[baseline] - 1:+.1%}")
    """
    )

    md("### Fig. 7: AMS confidence vs. planning success (PRB)")
    code(
        r"""
    fig, r, p = px.exp3_confidence_success_figure(EXP3_RECORDS, calibration=CALIBRATION, domain="prb")
    print(f"Pearson r = {r:.3f}, p = {p:.1e}")
    nd.save(fig, "paper-fig7-confidence-vs-success.pdf")
    nd.show(fig)
    """
    )

    md("### Fig. 12: planning success by goal horizon")
    code(
        r"""
    fig = e3d.exp3_planning_horizon_figure(
        EXP3_RECORDS, calibration=CALIBRATION, domains=("prb", "gw"),
        theta=PLANNING_THETA, bootstrap_resamples=RESAMPLES,
    )
    nd.save(fig, "paper-fig12-success-by-horizon.pdf")
    nd.show(fig)
    """
    )

    md(
        r"""
    ### Table 6: retention and recovery

    Success on the instances where MAP is correct (retention) and where it is wrong
    (recovery), at θ = 0.5.
    """
    )
    code(
        r"""
    save_table(
        e3d.exp3_recovery_retention_table(
            EXP3_RECORDS, calibration=CALIBRATION, domains=("prb", "gw"),
            theta=PLANNING_THETA, bootstrap_resamples=RESAMPLES,
        ),
        "paper-table6-exp3-retention-recovery.csv",
    )
    """
    )

    md("### Fig. 13: recovery rate vs. θ")
    code(
        r"""
    fig = e3d.exp3_recovery_vs_theta_figure(EXP3_RECORDS, calibration=CALIBRATION, domains=("prb", "gw"))
    nd.save(fig, "paper-fig13-recovery-vs-theta.pdf")
    nd.show(fig)
    """
    )

    # =========================================================================== Review
    md(
        r"""
    # Part 2: Review additions

    The results added during the review. Tables are also saved as CSV files in `figures/`.

    The Experiment 1 and 3 summaries for all six backbones are cached next to the outputs.
    """
    )
    code(
        r"""
    EXP1 = nd.load_exp1_summary(resamples=RESAMPLES)
    EXP3 = nd.load_exp3_summary(resamples=RESAMPLES)
    print("Experiment 1 backbones:", [nd.model_label(m) for m in nd.exp1_models()])
    print("Experiment 3 backbones:", [nd.model_label(m) for m in nd.exp3_models()])
    """
    )

    md(
        r"""
    ## Six VLM backbones: belief-set quality at θ = 0.9

    Coverage [95% CI] and interquartile mean (IQM) set size. Valid MAP is the MAP state
    projected onto the states that satisfy the constraints.
    """
    )
    code(
        r"""
    THETA = 0.9
    METHODS = ["map", "valid_map", "fluent_abstention_unconstrained", "mbs_prevas"]
    DOMAINS = ["prb", "gw", "alf"]
    SIZE_COL = "IQM Compact Size"

    sel = EXP1[(EXP1["Calibration"] == CALIBRATION) & EXP1["Method"].isin(METHODS)
               & (EXP1["Theta"] == THETA) & EXP1["Domain"].isin(DOMAINS)]


    def _cell(row):
        coverage = f"{row['Coverage']:.2f} [{row['Coverage CI Low']:.2f}, {row['Coverage CI High']:.2f}]"
        return f"{coverage} | {row[SIZE_COL]:.3g}"


    tbl = sel.assign(Cell=sel.apply(_cell, axis=1)).pivot_table(
        index=["Model Label", "Method Label"], columns="Domain", values="Cell",
        aggfunc="first", observed=True,
    ).reindex(
        [(m, meth) for m in sel["Model Label"].unique() for meth in sel["Method Label"].unique()]
    ).dropna(how="all").reindex(columns=DOMAINS)
    print(f"coverage [95% CI] | belief-set size ({SIZE_COL}) at θ = {THETA}")
    save_table(tbl, "review-exp1-backbones.csv")
    """
    )

    md("### Coverage vs. set size per backbone")
    code(
        r"""
    fig = nd.exp1_model_comparison_plot(EXP1, method="mbs_prevas", calibration=CALIBRATION,
                                        domains=("prb", "gw", "alf"), size_field="compact",
                                        size_stat="iqm")
    nd.save(fig, "review-exp1-backbones-mbs.pdf")
    nd.show(fig)

    fig = nd.exp1_coverage_size_by_model_plot(
        EXP1, methods=["map", "valid_map", "fluent_abstention_unconstrained",
                       "random_theta_confident", "mbs_prevas"],
        calibration=CALIBRATION, models=None, domains=("prb", "gw", "alf"), size_field="compact",
        size_stat="iqm",
    )
    nd.save(fig, "review-exp1-backbones-all-methods.pdf")
    nd.show(fig)
    """
    )

    md(
        r"""
    ## Reliability: belief mass vs. true-state coverage

    θ bounds the probability mass that the belief attributes to the set. It equals the
    probability that the set contains the true state only insofar as the belief is
    calibrated. Each point is one θ: the average mass of the MBSθ sets against their
    empirical coverage.
    """
    )
    code(
        r"""
    fig = nd.exp1_mass_coverage_by_domain_plot(EXP1, method="mbs_prevas", calibration=CALIBRATION,
                                               models=None, domains=("prb", "gw", "alf"))
    nd.save(fig, "review-reliability.pdf")
    nd.show(fig)

    fig = nd.exp1_mass_coverage_by_model_plot(
        EXP1, methods=["map", "valid_map", "fluent_abstention_unconstrained",
                       "random_theta_confident", "mbs_prevas"],
        calibration=CALIBRATION, models=None, domains=("prb", "gw", "alf"),
    )
    nd.save(fig, "review-reliability-backbones.pdf")
    nd.show(fig)
    """
    )
    code(
        r"""
    # Largest |coverage - mass| of MBSθ for each backbone, at θ in [0.9, 0.99] and over all θ.
    mbs = EXP1[(EXP1["Method"] == "mbs_prevas") & (EXP1["Calibration"] == CALIBRATION)
               & EXP1["Domain"].isin(["prb", "gw", "alf"])].copy()
    mbs["Gap"] = mbs["Coverage"] - mbs["Avg Mass"]
    gaps = mbs.groupby("Model Label", observed=True).apply(
        lambda g: pd.Series({
            "max |gap|, θ ≥ 0.9": g.loc[g["Theta"] >= 0.9, "Gap"].abs().max(),
            "max |gap|, all θ": g["Gap"].abs().max(),
        }),
        include_groups=False,
    )
    save_table(gaps.round(3), "review-reliability-gaps.csv")
    """
    )

    md(
        r"""
    ## Explored states on constrained instances (Experiment 2)

    Median number of states that PRaVaSS explores on instances with constraints, next to
    the `1 + |F|·k` bound that Prop. 1 proves without constraints.
    """
    )
    code(
        r"""
    save_table(px.exp2_states_visited_table(EXP2_RECORDS), "review-exp2-explored-states.csv", index=False)
    """
    )

    md(
        r"""
    ## Six VLM backbones: planning success

    Success [95% CI] (instances) [fraction of planner timeouts], for every backbone. The
    oracle plans from the true state.
    """
    )
    code(
        r"""
    METHODS = ["map", "adaptive_fluent_abstention", "mbs_prevas_prefix_search",
               "mbs_prevas_prefix_search_constraints_exact", "oracle"]
    blocks = []
    for model in nd.exp3_models():
        t = nd.exp3_success_table(EXP3, model=model, calibration=CALIBRATION,
                                  methods=METHODS, theta=None, domains=("prb", "gw"))
        t.insert(0, "Model", nd.model_label(model))
        blocks.append(t)
    save_table(pd.concat(blocks), "review-exp3-backbones.csv")
    """
    )

    md(
        r"""
    ## Constraint ablation (GPT-4.1)

    AMS with the same marginals and search, varying only the constraint layer: the disjoint
    selected groups (AMS-Loose), all mutex groups with the exact normalizer (AMS), and all
    mutex groups with the closed-form relaxed normalizer of App. G.5 (relaxed Z). MAP and
    Adaptive Abstention receive the same calibrated marginals.
    """
    )
    code(
        r"""
    METHODS = ["map", "adaptive_fluent_abstention", "mbs_prevas_prefix_search",
               "mbs_prevas_prefix_search_constraints_exact",
               "mbs_prevas_prefix_search_constraints_relaxed"]
    save_table(
        nd.exp3_success_table(EXP3, model=PAPER_MODEL, calibration=CALIBRATION,
                              methods=METHODS, theta=None, domains=("prb", "gw")),
        "review-exp3-constraint-ablation.csv",
    )
    """
    )

    md(
        r"""
    ### Paired contrasts

    Differences in planning success between two methods on the same test instances, with
    95% paired-bootstrap intervals (10,000 resamples, seed 0).
    """
    )
    code(
        r"""
    save_table(
        px.exp3_paired_constraint_ablation(EXP3_RECORDS),
        "review-exp3-constraint-ablation-paired.csv", index=False,
    )
    """
    )

    md(
        r"""
    ## Closed-loop replanning on ALFWorld (Experiment 6)

    120 frozen ALFWorld tasks (115 paired after five method-independent exclusions). Every
    declared non-visible, goal-relevant feature starts at probability 0.5. Each method
    plans, one relevant feature is revealed without noise, and each method replans.
    Success means that the plan reaches the goal from the true initial state. This is a
    synthetic stress test of partial observability, not an estimate of natural VLM errors.
    """
    )
    code(
        r"""
    RUN_NAME = "uniform_hidden_main_n120_adaptive_v3"
    METHODS = ["map", "adaptive_abstention", "ams"]
    EXP6 = nd.load_exp6_uniform_hidden_summary(model=PAPER_MODEL, calibration=CALIBRATION,
                                              run_name=RUN_NAME)
    display(save_table(nd.exp6_closed_loop_success_table(EXP6, model=PAPER_MODEL,
                                                         calibration=CALIBRATION, methods=METHODS),
                       "review-exp6-success.csv"))
    display(save_table(nd.exp6_closed_loop_effect_table(EXP6, model=PAPER_MODEL,
                                                        calibration=CALIBRATION, methods=METHODS),
                       "review-exp6-effects.csv"))
    display(save_table(nd.exp6_closed_loop_efficiency_table(EXP6, model=PAPER_MODEL,
                                                            calibration=CALIBRATION,
                                                            methods=["ams", "adaptive_abstention"]),
                       "review-exp6-efficiency.csv"))
    """
    )
    code(
        r"""
    # Failures recovered and successes lost by the observation, and the oracle's success.
    transitions = pd.DataFrame(
        {method: EXP6["transitions"][f"m0:{method}"] for method in METHODS}
    ).T[["n", "recovered", "regressed", "retained_success", "failed_both"]]
    display(transitions)

    exp6_dir = DATA / "experiment_6_controlled_prior" / PAPER_MODEL / RUN_NAME
    oracle = px.load_records(exp6_dir / "sample_results.jsonl", methods=["oracle"])
    for round_index, name in ((0, "before"), (1, "after")):
        rows = [r for r in oracle if r.get("round") == round_index]
        print(f"oracle, {name} the observation: {sum(bool(r.get('success')) for r in rows)}/{len(rows)} plans succeed")
    print("analysis decision:", EXP6["analysis_decision"]["decision"],
          "| common exclusions:", EXP6["common_exclusions"])
    """
    )
    code(
        r"""
    fig = nd.exp6_closed_loop_plot(EXP6, model=PAPER_MODEL, calibration=CALIBRATION,
                                            methods=METHODS)
    nd.save(fig, "review-exp6-closed-loop.pdf")
    nd.show(fig)
    """
    )

    md(
        r"""
    ## Correlated perceptual noise (Experiment 7)

    The unconstrained belief assumes that features are independent. Here a shared nuisance
    event with rate π corrupts a block of features together, pulling each marginal toward its
    opposite with strength ρ. The true joint is a mixture of two products, so masses and total
    variation are exact. The estimator receives the mixture's exact marginals, so the only
    error left is the independence assumption. `MBSθ (true joint)` is the minimal set of the
    true joint, computed by a certified exact search. Blocks are either a whole mutex group
    (`constraint`) or the least-decided features (`uncertain`). This is a synthetic
    sensitivity analysis; Experiment 1 measures coverage against real labels.
    """
    )
    code(
        r"""
    STRENGTH = 1.0   # ρ
    THETA = 0.9
    EXP7 = nd.load_exp7_correlated_noise_summary(model=PAPER_MODEL, calibration=CALIBRATION,
                                                 run_name="main")
    display(save_table(nd.exp7_distortion_table(EXP7, model=PAPER_MODEL, calibration=CALIBRATION,
                                                group_mode="constraint", strength=STRENGTH,
                                                theta=THETA),
                       "review-exp7-distortion.csv"))
    display(save_table(nd.exp7_robustness_table(EXP7, model=PAPER_MODEL, calibration=CALIBRATION,
                                                group_mode="constraint", strength=STRENGTH,
                                                theta=THETA,
                                                methods=["mbs_prevas", "oracle_joint", "valid_map", "map"]),
                       "review-exp7-robustness.csv"))
    display(save_table(nd.exp7_block_mode_table(EXP7, model=PAPER_MODEL, calibration=CALIBRATION,
                                                group_modes=["constraint", "uncertain"],
                                                strength=STRENGTH, theta=THETA),
                       "review-exp7-block-modes.csv"))
    """
    )
    md(
        r"""
    ### Ranges over the whole sweep

    The tables above fix ρ = 1 and θ = 0.9. This summary covers both block modes, both
    strengths ρ and every nuisance rate π (including the uncorrelated control π = 0),
    separately for each θ.
    """
    )
    code(
        r"""
    cells = list(EXP7["sweep"].values())
    rows = []
    for theta in sorted({float(c["theta"]) for c in cells}):
        sel = [c for c in cells if float(c["theta"]) == theta]
        mbs = [c["method_truth_coverage"]["mbs_prevas"] for c in sel]
        rows.append({
            "θ": theta,
            "MBSθ coverage, min–max": f"{min(mbs):.1%} – {max(mbs):.1%}",
            "MAP coverage, min": f"{min(c['method_truth_coverage']['map'] for c in sel):.1%}",
            "Valid MAP coverage, min": f"{min(c['method_truth_coverage']['valid_map'] for c in sel):.1%}",
            "Total variation, max": f"{max(c['total_variation_mean'] for c in sel):.3f}",
            "|Attributed − true mass|, max": f"{100 * max(abs(c['mass_coverage_gap']['mean']) for c in sel):.1f} pp",
            "Size vs. true-joint set, max": f"{max(c['size_ratio_median'] for c in sel):.2f}×",
        })
    save_table(pd.DataFrame(rows), "review-exp7-sweep-ranges.csv", index=False)
    """
    )
    code(
        r"""
    fig = nd.exp7_correlated_noise_plot(EXP7, model=PAPER_MODEL, calibration=CALIBRATION,
                                        group_mode="constraint", strength=STRENGTH, theta=THETA)
    nd.save(fig, "review-exp7-correlated-noise.pdf")
    nd.show(fig)
    """
    )


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    build()
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
