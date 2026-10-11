from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tqdm import tqdm

from .baselines import (
    adaptive_fluent_abstention_tau_candidates,
    fluent_abstention_unconstrained,
    fluent_prediction_marginals,
)
from .data import (
    CONSTRAINTS_DIR,
    PREDICTION_DIR,
    SPLIT_DIR,
    CalibrationName,
    SampleId,
    available_calibrations,
    load_calibration_bundle,
    load_validation_sample_ids,
    model_to_path_segment,
    prediction_path,
)
from .io import load_json, write_json_atomic
from .normalizers import selected_groups_runtime_normalizer
from .states import (
    StateSet,
    constraint_satisfying_full_state_count,
    project_groups,
    visible_indices,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_TARGET = "smallest_global_tau_with_mean_constrained_mass_ge_theta"
# τ is picked on the validation split, the same split that fits Platt scaling.
CALIBRATION_SOURCE_SPLIT = "validation"
MEAN_MASS_THETA_TOLERANCE = 1e-12


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _ensure_valid_cwd() -> None:
    """Make multiprocessing spawn robust when the caller's cwd disappeared."""
    try:
        os.getcwd()
    except FileNotFoundError:
        os.chdir(PROJECT_ROOT)


def filter_ids_for_calibration(
    sample_ids: Iterable[SampleId],
    domains: Iterable[str] | None = None,
    configs: Iterable[str] | None = None,
) -> list[SampleId]:
    domain_filter = None if domains is None else set(domains)
    config_filter = None if configs is None else set(configs)
    filtered: list[SampleId] = []
    for sid in sample_ids:
        if domain_filter is not None and sid.domain not in domain_filter:
            continue
        if config_filter is not None and sid.configuration not in config_filter:
            continue
        filtered.append(sid)
    return filtered


def calibration_source_sample_ids(
    domains: Iterable[str] | None = None,
    configs: Iterable[str] | None = None,
) -> list[SampleId]:
    """Return the validation sample ids used for τ calibration."""
    return filter_ids_for_calibration(
        load_validation_sample_ids(), domains=domains, configs=configs
    )


def count_committed_full_valid_states(
    *,
    full_feature_count: int,
    selected_groups: Sequence[Sequence[int]],
    search_indices: Sequence[int],
    search_feature_count: int,
    commits: dict[int, bool],
) -> int:
    """Count full constraint-satisfying completions for committed search literals.

    Retained as a utility for callers that still want the old size-based view
    of a committed template. The current calibration target works in mass-space
    and does not rely on this count.
    """
    template = tuple(commits.get(idx) for idx in range(search_feature_count))
    return constraint_satisfying_full_state_count(
        StateSet(
            states=(),
            unconstrained_mass=0.0,
            constrained_mass=0.0,
            total_states_considered=0,
            committed_template=template,
        ),
        full_feature_count=full_feature_count,
        selected_groups=selected_groups,
        search_indices=search_indices,
    )


def _tau_sample_worker(args: tuple) -> tuple:
    (
        sid,
        calibration,
        model,
        split,
        split_data_root_str,
        prediction_root_str,
        constraints_root_str,
    ) = args
    split_data_root = Path(split_data_root_str)
    prediction_root = Path(prediction_root_str)
    constraints_root = Path(constraints_root_str)
    try:
        if calibration not in available_calibrations(model, sid.domain):
            return ("skip", sid.split_relative_path, "calibration_unavailable", None)
        if not prediction_path(sid, split, model, prediction_root).exists():
            return ("skip", sid.split_relative_path, "prediction_missing", None)
        bundle = load_calibration_bundle(
            sid,
            calibration,
            model=model,
            split=split,
            prediction_root=prediction_root,
            constraints_root=constraints_root,
            split_data_root=split_data_root,
        )
        search_idx = visible_indices(bundle.none_is_max)
        reduced_probs = tuple(bundle.probabilities[i] for i in search_idx)
        reduced_groups = project_groups(bundle.selected_groups, search_idx)
        z = selected_groups_runtime_normalizer(bundle, feature_space="visible")
        if z <= 0.0:
            return ("skip", sid.split_relative_path, "invalid_normalizer", None)
        candidates = adaptive_fluent_abstention_tau_candidates(
            fluent_prediction_marginals(reduced_probs)
        )
        payload = {
            "sample": sid.split_relative_path,
            "probabilities": reduced_probs,
            "selected_groups": reduced_groups,
            "normalizer": z,
            "tau_candidates": candidates,
        }
        return ("used", sid.split_relative_path, None, payload)
    except Exception as exc:
        return (
            "skip",
            sid.split_relative_path,
            f"{type(exc).__name__}: {exc}",
            None,
        )


def _mean_mass_curve(used_samples: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    """Mean validation constrained mass of the Feature Abstention set at every candidate τ.

    The global candidate set is the union of per-sample ``tau_candidates``, in
    ascending order. For each candidate τ, this computes each sample's
    constrained mass freshly (correct semantics — no interpolation across
    heterogeneous candidate grids) and averages. The curve does not depend on
    θ, so it is computed once per (domain, calibration).
    """
    if not used_samples:
        return []
    global_candidates = sorted(
        {float(tau) for sample in used_samples for tau in sample["tau_candidates"]}
    )
    history: list[dict[str, float]] = []
    for tau in tqdm(global_candidates, desc="mean-mass curve", leave=False):
        masses: list[float] = []
        for sample in used_samples:
            result = fluent_abstention_unconstrained(
                sample["probabilities"],
                tau,
                sample["normalizer"],
                selected_groups=sample["selected_groups"],
            )
            if result.status != "ok":
                continue
            masses.append(result.constrained_mass)
        if not masses:
            continue
        history.append({"tau": tau, "mean_mass": sum(masses) / len(masses)})
    return history


def _select_tau(
    history: Sequence[Mapping[str, float]], theta: float
) -> tuple[float | None, float | None]:
    """The smallest τ on the mean-mass curve whose mean mass meets θ, and that mass."""
    for point in history:
        if point["mean_mass"] + MEAN_MASS_THETA_TOLERANCE >= theta:
            return point["tau"], point["mean_mass"]
    return None, None


def _aggregate_global_tau(
    used_samples: Sequence[Mapping[str, Any]],
    theta: float,
) -> tuple[float | None, float | None, list[dict[str, float]]]:
    """Pick the smallest global τ candidate whose mean validation constrained mass meets θ.

    Returns the selected τ, the mean mass at it, and the full mean-mass curve
    (see :func:`_mean_mass_curve`) so callers can inspect or plot it.
    """
    history = _mean_mass_curve(used_samples)
    selected_tau, mean_at_selected = _select_tau(history, theta)
    return selected_tau, mean_at_selected, history


def _calibration_row(
    *,
    domain: str,
    calibration: CalibrationName,
    theta: float,
    used_samples: Sequence[Mapping[str, Any]],
    skipped_samples: Sequence[dict[str, str]],
    history: Sequence[Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    if history is None:
        history = _mean_mass_curve(used_samples)
    selected_tau, mean_at_selected = _select_tau(history, theta)
    return {
        "domain": domain,
        "calibration": calibration,
        "theta": theta,
        "calibration_target": CALIBRATION_TARGET,
        "selected_tau": selected_tau,
        "mean_mass_at_selected_tau": mean_at_selected,
        "mean_mass_curve": list(history),
        "used_samples": [str(s["sample"]) for s in used_samples],
        "skipped_samples": list(skipped_samples),
    }


def calibrate_tau_for_domain(
    sample_ids: Sequence[SampleId],
    domain: str,
    calibration: CalibrationName,
    *,
    model: str,
    split: str = CALIBRATION_SOURCE_SPLIT,
    thetas: Sequence[float] = (0.9,),
    split_data_root: Path = SPLIT_DIR,
    prediction_root: Path = PREDICTION_DIR,
    constraints_root: Path = CONSTRAINTS_DIR,
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    """Load each sample once, then aggregate the global τ for every θ.

    The sample-level work (loading bundles, projecting probabilities, computing
    the normalizer, building per-sample τ candidates) is θ-independent, so we
    cache the per-sample payloads and let the aggregator re-use them for every
    θ in ``thetas``.
    """
    domain_ids = [sid for sid in sample_ids if sid.domain == domain]
    split_data_root = _absolute_path(split_data_root)
    prediction_root = _absolute_path(prediction_root)
    constraints_root = _absolute_path(constraints_root)
    work_items = [
        (
            sid,
            calibration,
            model,
            split,
            str(split_data_root),
            str(prediction_root),
            str(constraints_root),
        )
        for sid in domain_ids
    ]

    desc = f"tau {domain}/{calibration}"
    if n_jobs > 1 and len(work_items) > 1:
        _ensure_valid_cwd()
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_jobs) as pool:
            iterator = pool.imap_unordered(_tau_sample_worker, work_items, chunksize=1)
            results = list(tqdm(iterator, total=len(work_items), desc=desc))
        # Restore the input order, so that the output (and the floating-point
        # sums behind the mean-mass curve) does not depend on worker timing.
        position = {sid.split_relative_path: i for i, sid in enumerate(domain_ids)}
        results.sort(key=lambda result: position[result[1]])
    else:
        results = [_tau_sample_worker(item) for item in tqdm(work_items, desc=desc)]

    used_samples: list[dict[str, Any]] = []
    skipped_samples: list[dict[str, str]] = []
    for kind, path_str, reason, payload in results:
        if kind == "skip":
            skipped_samples.append({"sample": path_str, "reason": reason})
            continue
        used_samples.append(payload)

    history = _mean_mass_curve(used_samples)
    return [
        _calibration_row(
            domain=domain,
            calibration=calibration,
            theta=theta,
            used_samples=used_samples,
            skipped_samples=skipped_samples,
            history=history,
        )
        for theta in thetas
    ]


def tau_output_path(
    output_root: Path,
    calibration: CalibrationName,
    domain: str,
) -> Path:
    """Return ``<output_root>/<calibration>/<domain>/tau_calibration.json``."""
    return output_root / calibration / domain / "tau_calibration.json"


def write_tau_calibration(
    output_root: Path,
    *,
    model: str,
    domains: Sequence[str],
    configs: Sequence[str] | None = None,
    calibrations: Sequence[CalibrationName],
    thetas: Sequence[float] = (0.9,),
    n_jobs: int | None = None,
    overwrite: bool = False,
) -> dict[tuple[CalibrationName, str], Path]:
    """Fit τ on the validation split and shard outputs by (calibration, domain).

    Writes one JSON file per (calibration, domain) at
    ``<output_root>/<calibration>/<domain>/tau_calibration.json``. Each file
    contains τ rows for every θ in this run. Returns a mapping of
    (calibration, domain) → output path covering every requested combination,
    including ones that were skipped because their file already existed.
    """
    _ensure_valid_cwd()
    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, (os.cpu_count() or 1) - 1)
    source_ids = calibration_source_sample_ids(domains=domains, configs=configs)
    if not source_ids:
        raise ValueError("No validation samples matched the requested domains/configs.")
    split = CALIBRATION_SOURCE_SPLIT
    output_root = _absolute_path(output_root)
    paths: dict[tuple[CalibrationName, str], Path] = {}
    for domain in domains:
        domain_source_samples = [
            sid.split_relative_path for sid in source_ids if sid.domain == domain
        ]
        for calibration in calibrations:
            output_path = tau_output_path(output_root, calibration, domain)
            paths[(calibration, domain)] = output_path
            if output_path.exists() and not overwrite:
                continue
            rows = calibrate_tau_for_domain(
                source_ids,
                domain,
                calibration,
                model=model,
                split=split,
                thetas=thetas,
                split_data_root=SPLIT_DIR,
                n_jobs=n_jobs,
            )
            payload = {
                "model": model,
                "calibration": calibration,
                "domain": domain,
                "split": split,
                "calibration_target": CALIBRATION_TARGET,
                "thetas": list(thetas),
                "source_samples": domain_source_samples,
                "results": rows,
            }
            write_json_atomic(output_path, payload)
    return paths


ABSTENTION_OUTPUT_DIR = Path("abstention_data")


def load_selected_tau_lookup(
    model: str,
    *,
    calibrations: Sequence[CalibrationName],
    domains: Sequence[str],
    output_root: Path = ABSTENTION_OUTPUT_DIR,
) -> dict[tuple[str, str, float], float]:
    """Merge sharded τ files into one (domain, calibration, theta) → τ lookup.

    Mirrors the sharded layout written by ``write_tau_calibration``:
    ``<output_root>/<model>/<calibration>/<domain>/tau_calibration.json``.
    Missing shards are skipped (e.g. ``lifted``/``alf`` produces a file whose
    rows all have ``selected_tau = None``, which ``selected_tau_lookup`` drops).
    """
    base = output_root / model_to_path_segment(model)
    lookup: dict[tuple[str, str, float], float] = {}
    for calibration in calibrations:
        for domain in domains:
            path = tau_output_path(base, calibration, domain)
            if not path.exists():
                continue
            lookup.update(selected_tau_lookup(load_json(path)))
    return lookup


def selected_tau_lookup(
    payload: dict[str, Any],
) -> dict[tuple[str, str, float], float]:
    """Return τ keyed by (domain, calibration, theta).

    Skips rows whose calibration produced no τ (e.g. all samples were skipped).
    """
    lookup: dict[tuple[str, str, float], float] = {}
    for result in payload.get("results", []):
        tau = result.get("selected_tau")
        theta = result.get("theta")
        if tau is None or theta is None:
            continue
        key = (
            str(result["domain"]),
            str(result["calibration"]),
            float(theta),
        )
        lookup[key] = float(tau)
    return lookup
