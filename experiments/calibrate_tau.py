#!/usr/bin/env python3
"""Calibrate the threshold τ of the Feature Abstention baseline.

For each θ, pick the smallest τ such that the Feature Abstention belief set
has a mean task-belief mass of at least θ over the validation split (paper,
Sec. 4, Experiment 1). The candidates for τ come from the confidences
max(p, 1 - p) of the validation samples' calibrated probabilities: the smallest
confidence, the midpoints between adjacent confidences, and 1. So τ depends on
the Platt profiles: run calibrate_vlm.py first. The resulting JSON drives the
Feature Abstention baseline in Experiments 1 and 3.

Output layout (model / calibration / domain):
    abstention_data/<model>/<calibration>/<domain>/tau_calibration.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mbs_experiments.abstention import write_tau_calibration
from mbs_experiments.baselines import THETA_GRID
from mbs_experiments.data import model_to_path_segment

GEMMA4_2B_MODEL = "google/gemma-4-E2B-it"
ABSTENTION_OUTPUT_DIR = Path("abstention_data")


def default_output_root(model: str) -> Path:
    return ABSTENTION_OUTPUT_DIR / model_to_path_segment(model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate τ for the Feature Abstention baseline on the validation split"
    )
    parser.add_argument(
        "--model",
        default=GEMMA4_2B_MODEL,
        help=(
            "Model identifier whose predictions and Platt profiles to use. "
            "Must match the --model passed to make_predictions.py and "
            f"calibrate_vlm.py. Default: {GEMMA4_2B_MODEL}."
        ),
    )
    parser.add_argument("--domains", nargs="+", default=("prb", "gw", "alf"))
    parser.add_argument("--configs", nargs="+")
    parser.add_argument(
        "--calibrations",
        nargs="+",
        choices=("global", "lifted", "none"),
        default=("global", "lifted", "none"),
        help=(
            "Calibration scopes to fit τ for. 'none' uses raw probabilities "
            "without Platt scaling. Default: global lifted none."
        ),
    )
    parser.add_argument(
        "--thetas",
        type=float,
        nargs="+",
        default=list(THETA_GRID),
        help="Calibrate the θ-driven τ separately for each θ (default: THETA_GRID).",
    )
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output root directory. Default: abstention_data/<model>/. "
            "One JSON file is written per (calibration, domain) at "
            "<output-root>/<calibration>/<domain>/tau_calibration.json."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Refit and overwrite existing tau_calibration.json files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or default_output_root(args.model)
    output_root.mkdir(parents=True, exist_ok=True)

    paths = write_tau_calibration(
        output_root,
        model=args.model,
        domains=tuple(sorted(args.domains)),
        configs=tuple(sorted(args.configs)) if args.configs else None,
        calibrations=tuple(args.calibrations),
        thetas=tuple(args.thetas),
        n_jobs=args.jobs,
        overwrite=args.overwrite,
    )
    print(
        f"θ-driven τ output covers {len(paths)} (calibration, domain) "
        f"combination(s) under {output_root}"
    )


if __name__ == "__main__":
    main()
