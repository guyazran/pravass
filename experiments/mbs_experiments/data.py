from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal

import numpy as np

# NOTE: `s3e` is imported lazily inside the three functions that need it
# (`_load_profile`, `_load_up_problem`, `calibrated_probability_from_score`),
# NOT at module top level. Importing `s3e.calibration` pulls in PyTorch and
# costs ~25-30s. Experiment 3 spawns its belief-planning workers with the
# "spawn" start method, which re-imports this module in every child process;
# a top-level s3e import therefore added ~25s of irrelevant PyTorch startup to
# *every* CPOR planning call, pushing them past `--planning-timeout` and
# manifesting as planner timeouts. Belief planning never touches calibration,
# so deferring the import keeps `import mbs_experiments.data` torch-free and
# fast. (`from __future__ import annotations` makes the PlattScalingProfile
# annotations below lazy strings, so no top-level import is needed for typing.)

from .io import load_json

if TYPE_CHECKING:
    from s3e.calibration import PlattScalingProfile

ROOT = Path(".")
PREDICTION_DIR = ROOT / "prediction_data"
CALIB_DIR = ROOT / "calib_data"
CONSTRAINTS_DIR = ROOT / "constraints_data"
SPLIT_DIR = ROOT / "split_data"
TEST_DIR = SPLIT_DIR / "test"
# The validation split fits Platt scaling (calibrate_vlm.py) and picks the
# Feature Abstention threshold τ (calibrate_tau.py).
VALIDATION_DIR = SPLIT_DIR / "validation"
TEST_SAMPLES_FILE = SPLIT_DIR / "test_samples.txt"
VALIDATION_SAMPLES_FILE = SPLIT_DIR / "validation_samples.txt"
CONDITIONAL_DOMAINS_DIR = ROOT / "conditional_domains"

CalibrationName = Literal["global", "lifted", "none"]

NO_CALIBRATION: CalibrationName = "none"
PLATT_CALIBRATIONS: tuple[CalibrationName, ...] = ("global", "lifted")

SPLIT_DIRS: dict[str, Path] = {
    "test": TEST_DIR,
    "validation": VALIDATION_DIR,
}


def model_to_path_segment(model: str) -> str:
    """Encode a model identifier as a single path segment (matches make_predictions.py)."""
    return model.replace("/", "__")


def split_dir(split: str, split_data_root: Path = SPLIT_DIR) -> Path:
    try:
        rel = SPLIT_DIRS[split].name
    except KeyError as exc:
        raise ValueError(
            f"Unknown split '{split}'. Known: {sorted(SPLIT_DIRS)}"
        ) from exc
    return split_data_root / rel


@dataclass(frozen=True)
class SampleId:
    domain: str
    configuration: str
    sample: str
    split_relative_path: str


@dataclass(frozen=True)
class CalibrationBundle:
    sample_id: SampleId
    calibration: CalibrationName
    model: str
    split: str
    prediction_path: Path
    constraints_path: Path
    sample_path: Path
    problem_path: Path
    conditional_domain_path: Path
    metadata: dict[str, Any]
    feature_names: tuple[str, ...]
    probabilities: tuple[float, ...]
    none_is_max: tuple[bool, ...]
    ground_truth: tuple[bool, ...]
    ground_truth_full: dict[str, bool]
    selected_groups: tuple[tuple[int, ...], ...]
    constraint_groups: tuple[tuple[int, ...], ...]
    normalizer_payload: dict[str, Any]


def parse_split_line(line: str) -> SampleId:
    clean = line.strip()
    parts = clean.split("/")
    if len(parts) < 3:
        raise ValueError(f"Expected '<domain>/<configuration>/<sample>', got {clean!r}")
    return SampleId(
        domain=parts[0],
        configuration=parts[1],
        sample=parts[-1],
        split_relative_path=clean,
    )


def load_split_sample_ids(path: Path) -> list[SampleId]:
    sample_ids: list[SampleId] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                sample_ids.append(parse_split_line(line))
    return sample_ids


def prediction_path(
    sample_id: SampleId,
    split: str,
    model: str,
    root: Path = PREDICTION_DIR,
) -> Path:
    return (
        root
        / split
        / model_to_path_segment(model)
        / sample_id.split_relative_path
        / "prediction.json"
    )


def constraints_path(
    sample_id: SampleId,
    split: str,
    root: Path = CONSTRAINTS_DIR,
) -> Path:
    """Return ``<root>/<split>/<sample_path>/constraints.json``.

    Constraints are written per-split by extract_constraints.py so that
    re-extracting one split never overwrites another's files.
    """
    return root / split / sample_id.split_relative_path / "constraints.json"


def sample_path(
    sample_id: SampleId,
    split: str,
    split_data_root: Path = SPLIT_DIR,
) -> Path:
    return split_dir(split, split_data_root) / sample_id.split_relative_path


def conditional_domain_path(domain: str) -> Path:
    return CONDITIONAL_DOMAINS_DIR / f"{domain}-cond-domain.pddl"


def calibration_profile_path(
    model: str,
    domain: str,
    calibration: CalibrationName,
    root: Path = CALIB_DIR,
) -> Path:
    if calibration == NO_CALIBRATION:
        raise ValueError(
            "calibration='none' uses raw probabilities and has no profile path."
        )
    return root / model_to_path_segment(model) / domain / f"platt_{calibration}.json"


def available_calibrations(
    model: str,
    domain: str,
    calib_root: Path = CALIB_DIR,
) -> tuple[CalibrationName, ...]:
    """Return the calibration scopes available for (model, domain).

    ``"none"`` is always available (raw probabilities are saved in every
    prediction.json). ``"global"``/``"lifted"`` require a fitted Platt profile
    under ``calib_data/<model>/<domain>/``.
    """
    found: list[CalibrationName] = [NO_CALIBRATION]
    for scope in PLATT_CALIBRATIONS:
        if calibration_profile_path(model, domain, scope, calib_root).exists():
            found.append(scope)
    return tuple(found)


_PROFILE_CACHE: dict[Path, PlattScalingProfile] = {}
_UP_PROBLEM_CACHE: dict[tuple[str, str], Any] = {}


def _load_profile(path: Path) -> PlattScalingProfile:
    from s3e.calibration import PlattScalingProfile  # lazy: avoids torch at import

    cache_key = path.resolve() if path.exists() else path
    if cache_key not in _PROFILE_CACHE:
        _PROFILE_CACHE[cache_key] = PlattScalingProfile.load(str(path))
    return _PROFILE_CACHE[cache_key]


def _load_up_problem(domain_pddl: str, problem_pddl: str):
    from s3e.pddl.up_utils import create_up_problem  # lazy: avoids torch at import

    key = (domain_pddl, problem_pddl)
    if key not in _UP_PROBLEM_CACHE:
        _UP_PROBLEM_CACHE[key] = create_up_problem(domain_pddl, problem_pddl)
    return _UP_PROBLEM_CACHE[key]


def calibrated_probability_from_score(
    score: float,
    *,
    profile: PlattScalingProfile,
    predicate: str,
    up_problem: Any,
) -> float:
    """Apply a saved Platt profile to a raw grouped-log-odds score.

    Falls back to the uncalibrated sigmoid of the raw score when the profile
    has no parameters for the relevant group (e.g. single-class predicates
    that were skipped during fitting).
    """
    from s3e.calibration import (  # lazy: avoids torch at import
        GLOBAL_CALIBRATION_KEY,
        apply_platt_scaling,
    )
    from s3e.pddl.up_utils import get_lifted_predicate_key

    if profile.scope == "global":
        params = profile.groups.get(GLOBAL_CALIBRATION_KEY)
    else:
        if up_problem is None:
            raise ValueError(
                "up_problem is required when applying a lifted-scope Platt profile."
            )
        key = get_lifted_predicate_key(up_problem, predicate)
        params = profile.groups.get(key)
    if params is None:
        return float(np.clip(1.0 / (1.0 + math.exp(-score)), 0.0, 1.0))
    calibrated = apply_platt_scaling(score, params)
    return float(np.clip(calibrated, 0.0, 1.0))


def project_selected_groups(
    selected_groups_json: Iterable[dict[str, Any]], feature_names: tuple[str, ...]
) -> tuple[tuple[int, ...], ...]:
    index = {name: i for i, name in enumerate(feature_names)}
    groups: list[tuple[int, ...]] = []
    for group in selected_groups_json:
        if group.get("type", "at_most_one") != "at_most_one":
            continue
        projected = tuple(sorted(index[atom] for atom in group.get("atoms", []) if atom in index))
        if len(projected) >= 2:
            groups.append(projected)
    seen: set[tuple[int, ...]] = set()
    deduped: list[tuple[int, ...]] = []
    for group in groups:
        if group not in seen:
            seen.add(group)
            deduped.append(group)
    return tuple(deduped)


def load_calibration_bundle(
    sample_id: SampleId,
    calibration: CalibrationName,
    *,
    model: str,
    split: str,
    prediction_root: Path = PREDICTION_DIR,
    constraints_root: Path = CONSTRAINTS_DIR,
    split_data_root: Path = SPLIT_DIR,
    calib_root: Path = CALIB_DIR,
) -> CalibrationBundle:
    """Load probabilities, ground truth, and constraints for one sample.

    Calibrated probabilities are computed on-the-fly from the raw score in
    prediction.json and the saved Platt profile under
    ``calib_data/<model>/<domain>/platt_<calibration>.json``.
    """
    pred_path = prediction_path(sample_id, split, model, prediction_root)
    cons_path = constraints_path(sample_id, split, constraints_root)
    sample_dir = sample_path(sample_id, split, split_data_root)
    prediction = load_json(pred_path)
    constraints = load_json(cons_path)
    predictions = prediction.get("predictions", {})
    feature_names = tuple(prediction.get("predicates") or predictions.keys())

    # Locate the PDDL files through the split layout rather than the paths
    # recorded in prediction.json, which reflect where the predictions were made.
    domain_pddl_path = str(sample_dir / "domain.pddl")
    problem_pddl_path = str(sample_dir / "problem.pddl")

    if calibration == NO_CALIBRATION:
        profile = None
        up_problem = None
    else:
        profile_path = calibration_profile_path(
            model, sample_id.domain, calibration, calib_root
        )
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Missing Platt scaling profile for model={model} "
                f"domain={sample_id.domain} calibration={calibration}: {profile_path}. "
                f"Run calibrate_vlm.py first."
            )
        profile = _load_profile(profile_path)
        up_problem = (
            _load_up_problem(domain_pddl_path, problem_pddl_path)
            if calibration == "lifted"
            else None
        )

    probabilities: list[float] = []
    none_is_max: list[bool] = []
    ground_truth: list[bool] = []
    for feature in feature_names:
        detail = predictions[feature]
        if bool(detail.get("none_is_max_raw", False)):
            probabilities.append(0.5)
        elif profile is None:
            probabilities.append(float(np.clip(float(detail["raw_probability"]), 0.0, 1.0)))
        else:
            score = float(detail["score"])
            probabilities.append(
                calibrated_probability_from_score(
                    score,
                    profile=profile,
                    predicate=feature,
                    up_problem=up_problem,
                )
            )
        none_is_max.append(bool(detail.get("none_is_max_raw", False)))
        ground_truth.append(
            bool(
                detail.get(
                    "ground_truth",
                    prediction.get("ground_truth_relevant", {}).get(feature, False),
                )
            )
        )

    return CalibrationBundle(
        sample_id=sample_id,
        calibration=calibration,
        model=model,
        split=split,
        prediction_path=pred_path,
        constraints_path=cons_path,
        sample_path=sample_dir,
        problem_path=Path(problem_pddl_path),
        conditional_domain_path=conditional_domain_path(sample_id.domain),
        metadata=dict(prediction.get("metadata", {})),
        feature_names=feature_names,
        probabilities=tuple(probabilities),
        none_is_max=tuple(none_is_max),
        ground_truth=tuple(ground_truth),
        ground_truth_full=dict(prediction.get("ground_truth_full", {})),
        selected_groups=project_selected_groups(
            constraints.get("selected_groups", []), feature_names
        ),
        constraint_groups=project_selected_groups(
            constraints.get("constraints", []), feature_names
        ),
        normalizer_payload=dict(constraints.get("normalizer", {})),
    )


def filter_sample_ids(
    sample_ids: Iterable[SampleId],
    *,
    domains: set[str] | None = None,
    configs: set[str] | None = None,
    limit: int | None = None,
) -> list[SampleId]:
    if limit is not None and limit <= 0:
        return []

    out: list[SampleId] = []
    for sample_id in sample_ids:
        if domains is not None and sample_id.domain not in domains:
            continue
        if configs is not None and sample_id.configuration not in configs:
            continue
        out.append(sample_id)
        if limit is not None and len(out) >= limit:
            break
    return out


def load_test_sample_ids() -> list[SampleId]:
    return load_split_sample_ids(TEST_SAMPLES_FILE)


def load_validation_sample_ids() -> list[SampleId]:
    return load_split_sample_ids(VALIDATION_SAMPLES_FILE)
