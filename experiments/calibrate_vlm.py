"""Fit Platt scaling of each model's VLM probabilities on the validation split.

Reads the prediction.json files that make_predictions.py wrote for the
validation split and fits a logistic recalibration of the per-feature log-odds
score: one per domain for the global scope (the paper's, Sec. 4), or one per
lifted predicate for the lifted scope. The experiments apply these profiles to
the test predictions.

Output layout (model / domain):
    calib_data/<model>/<domain>/platt_{global,lifted}.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

from s3e.calibration import PlattCalibrationSample
from s3e.semantic_state_estimator import SemanticStateEstimator
from s3e.translation import TemplateTranslator
from s3e.vlm.backend import VLMBackend, VLMOutput
from s3e.pddl.up_utils import create_up_problem, get_object_names_dict, get_pddl_strings

GEMMA4_2B_MODEL = 'google/gemma-4-E2B-it'
PREDICTION_OUTPUT_DIR = Path("prediction_data")
CALIB_OUTPUT_DIR = Path("calib_data")
SPLIT_DATA_DIR = Path("split_data")
NL_TEMPLATES_DIR = Path("nl_templates")
CALIBRATION_SPLIT = "validation"
CALIBRATION_MODE_CHOICES = ("global", "lifted", "both")


class _StubVLM(VLMBackend):
    """No-op VLM used when constructing an estimator solely to access its
    Platt-scaling fitting logic. Calibration runs from pre-computed scores
    and must not query a VLM."""

    def query(self, *args, **kwargs) -> VLMOutput:
        raise NotImplementedError(
            "Stub VLM should not be queried during calibration fitting."
        )


def model_to_path_segment(model: str) -> str:
    """Encode a model identifier as a single path segment (matches make_predictions.py)."""
    return model.replace("/", "__")


def get_prediction_root(model: str, domain: str) -> Path:
    return (
        PREDICTION_OUTPUT_DIR
        / CALIBRATION_SPLIT
        / model_to_path_segment(model)
        / domain
    )


def get_calibration_output_dir(model: str, domain: str) -> Path:
    return CALIB_OUTPUT_DIR / model_to_path_segment(model) / domain


def load_domain_translator(domain: str) -> TemplateTranslator:
    template_path = NL_TEMPLATES_DIR / f"{domain}_template.json"
    if not template_path.exists():
        raise FileNotFoundError(
            f"Missing template file for domain '{domain}': {template_path}"
        )
    with open(template_path) as f:
        templates = json.load(f)
    return TemplateTranslator(templates)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def find_prediction_files(model: str, domain: str) -> list[Path]:
    """Return all prediction.json files for a domain under the validation split."""
    prediction_root = get_prediction_root(model, domain)
    if not prediction_root.exists():
        raise FileNotFoundError(
            f"No predictions found at {prediction_root}. "
            f"Run make_predictions.py with --splits {CALIBRATION_SPLIT} "
            f"--model {model} first."
        )
    files = sorted(prediction_root.rglob("prediction.json"))
    if not files:
        raise FileNotFoundError(
            f"No prediction.json files found under {prediction_root}."
        )
    return files


def collect_calibration_samples(
    prediction_files: list[Path],
) -> tuple[list[PlattCalibrationSample], str, str]:
    """Convert prediction.json files into PlattCalibrationSamples.

    Returns the samples plus the domain.pddl and a representative
    problem.pddl path used to bootstrap the calibration estimator.
    """
    samples: list[PlattCalibrationSample] = []
    domain_path: str | None = None
    problem_path: str | None = None
    for path in tqdm(prediction_files, desc="Loading predictions"):
        payload = load_json(path)
        # The sample's PDDL files, located through the split layout rather than
        # the paths recorded in prediction.json.
        sample_dir = SPLIT_DATA_DIR / CALIBRATION_SPLIT / payload["split_relative_path"]
        if domain_path is None:
            domain_path = str(sample_dir / "domain.pddl")
        if problem_path is None:
            problem_path = str(sample_dir / "problem.pddl")
        sample_problem = str(sample_dir / "problem.pddl")
        for predicate, pred_data in payload["predictions"].items():
            samples.append(
                PlattCalibrationSample(
                    predicate=predicate,
                    score=float(pred_data["score"]),
                    label=bool(pred_data["ground_truth"]),
                    problem=sample_problem,
                )
            )
    assert domain_path is not None and problem_path is not None
    return samples, domain_path, problem_path


def build_calibration_estimator(
    domain: str,
    domain_pddl: str,
    problem_pddl: str,
) -> SemanticStateEstimator:
    """Build a SemanticStateEstimator with a stub VLM for fitting Platt scaling.

    The token groups, system prompt, and translator must match what was used
    to produce the predictions so the saved profile's metadata is consistent.
    """
    query_translator = load_domain_translator(domain)

    with open('prompts/system.md') as f:
        system_prompt = f.read()
    problem = create_up_problem(domain_pddl, problem_pddl)
    domain_str, _ = get_pddl_strings(problem)
    objects = get_object_names_dict(problem)
    objects_str = "\n".join(
        f"{key} type: {list(map(str, value))}"
        for key, value in objects.items()
    )
    system_prompt = system_prompt.format(domain=domain_str, objects=objects_str)

    with open(f'prompts/{domain}.md') as f:
        domain_instructions = f.read()

    return SemanticStateEstimator(
        domain=domain_pddl,
        problem=problem_pddl,
        vlm=_StubVLM(),
        query_translator=query_translator,
        system_prompt=system_prompt,
        additional_instructions=domain_instructions,
        null_tokens=["null", "Null", "NULL"],
    )


def resolve_scopes(calibration_mode: str) -> tuple[str, ...]:
    if calibration_mode == "both":
        return ("global", "lifted")
    return (calibration_mode,)


def calibrate_domain(
    domain: str,
    *,
    model: str,
    scopes: tuple[str, ...],
    overwrite: bool,
) -> None:
    """Fit Platt scaling profile(s) for a domain from saved predictions."""
    print(f"\n{'='*60}")
    print(f"Calibrating domain: {domain} (model={model}, scopes={list(scopes)})")
    print(f"{'='*60}")

    out_dir = get_calibration_output_dir(model, domain)
    scope_paths = {scope: out_dir / f"platt_{scope}.json" for scope in scopes}

    pending_scopes = [
        scope for scope, path in scope_paths.items()
        if overwrite or not path.exists()
    ]
    if not pending_scopes:
        print(
            f"{domain}: all requested calibrations already exist at {out_dir}, skipping."
        )
        return

    prediction_files = find_prediction_files(model, domain)
    print(f"{domain}: found {len(prediction_files)} prediction files")

    samples, domain_pddl, problem_pddl = collect_calibration_samples(prediction_files)
    print(f"{domain}: collected {len(samples)} calibration samples")

    estimator = build_calibration_estimator(domain, domain_pddl, problem_pddl)
    os.makedirs(out_dir, exist_ok=True)

    for scope in scopes:
        path = scope_paths[scope]
        if scope not in pending_scopes:
            print(f"{domain}: {scope} calibration already exists at {path}, skipping...")
            continue
        print(f"Fitting {scope} Platt scaling...")
        estimator.fit_platt_scaling_from_data(samples, scope=scope)
        estimator.save_platt_scaling(str(path))
        print(f"Saved {scope} calibration to {path}")


def list_domains_with_predictions(model: str) -> list[str]:
    """Discover domains that have at least one prediction.json under the validation split/model."""
    root = PREDICTION_OUTPUT_DIR / CALIBRATION_SPLIT / model_to_path_segment(model)
    if not root.exists():
        return []
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and any(d.rglob("prediction.json"))
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit Platt scaling profiles from precomputed VLM predictions on the "
            f"'{CALIBRATION_SPLIT}' split. Reads prediction.json files written "
            "by make_predictions.py."
        )
    )
    parser.add_argument(
        "--model",
        default=GEMMA4_2B_MODEL,
        help=(
            "Model identifier whose predictions to use. Must match the "
            "--model passed to make_predictions.py. Default: "
            f"{GEMMA4_2B_MODEL}."
        ),
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Domains to calibrate. Default: all domains with predictions found.",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=CALIBRATION_MODE_CHOICES,
        default="both",
        help=(
            "Which Platt scaling scope(s) to fit. 'both' fits global and lifted. "
            "Default: both."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Refit and overwrite existing Platt scaling profiles.",
    )
    args = parser.parse_args()

    scopes = resolve_scopes(args.calibration_mode)

    available = list_domains_with_predictions(args.model)
    if not available:
        raise FileNotFoundError(
            f"No predictions found under "
            f"{PREDICTION_OUTPUT_DIR / CALIBRATION_SPLIT / model_to_path_segment(args.model)}. "
            f"Run make_predictions.py with --splits {CALIBRATION_SPLIT} first."
        )

    domains = args.domains if args.domains is not None else available
    missing = [d for d in domains if d not in available]
    if missing:
        raise FileNotFoundError(
            f"Requested domains have no predictions for split={CALIBRATION_SPLIT} "
            f"model={args.model}: {missing}. Available: {available}."
        )

    print(f"Calibrating domains: {domains}")
    for domain in domains:
        calibrate_domain(
            domain,
            model=args.model,
            scopes=scopes,
            overwrite=args.overwrite,
        )

    print(
        "\nDone! Calibration data saved to",
        CALIB_OUTPUT_DIR / model_to_path_segment(args.model),
    )


if __name__ == "__main__":
    main()
