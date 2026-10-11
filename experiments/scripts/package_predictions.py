"""Package cached VLM predictions for the CPU-only reproduction path.

Running the VLMs (``make_predictions.py``) is the only step that needs GPUs or
paid API calls. This script bundles its outputs into a single archive that can
be published next to the raw data, so that others can reproduce every result on
CPU. It is meant for the authors; reproducers only download and unpack the
archive (see README.md).

The archive holds ``prediction_data/<split>/<model>/<domain>/...`` for the
paper's domains (PRB, GW, ALF), both splits, and the given models, plus a
``MANIFEST.json`` that lists every file with its SHA-256 digest.

Usage:
    uv run scripts/package_predictions.py --output predictions.zip
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_MODELS = (
    "OpenAI/gpt-4.1-2025-04-14",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-31B-it",
)
PAPER_DOMAINS = ("prb", "gw", "alf")
SPLITS = ("test", "validation")


def model_segment(model: str) -> str:
    """The directory name make_predictions.py uses for a model."""
    return model.replace("/", "__")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(prediction_dir: Path, models, domains) -> list[Path]:
    files = []
    for split in SPLITS:
        for model in models:
            for domain in domains:
                directory = prediction_dir / split / model_segment(model) / domain
                if not directory.is_dir():
                    raise SystemExit(f"missing predictions: {directory}")
                files.extend(sorted(p for p in directory.rglob("*") if p.is_file()))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prediction-dir", type=Path, default=ROOT / "prediction_data")
    parser.add_argument("--output", type=Path, default=ROOT / "predictions.zip")
    parser.add_argument("--models", nargs="+", default=list(PAPER_MODELS))
    parser.add_argument("--domains", nargs="+", default=list(PAPER_DOMAINS))
    args = parser.parse_args()

    files = collect(args.prediction_dir, args.models, args.domains)
    base = args.prediction_dir.parent
    manifest = {
        "models": args.models,
        "domains": args.domains,
        "splits": list(SPLITS),
        "files": {str(p.relative_to(base)): sha256(p) for p in files},
    }
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(base))
        archive.writestr("prediction_data/MANIFEST.json", json.dumps(manifest, indent=1, sort_keys=True))
    print(f"wrote {args.output} ({len(files)} files, sha256 {sha256(args.output)})")


if __name__ == "__main__":
    main()
