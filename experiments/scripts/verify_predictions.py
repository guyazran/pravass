"""Check unpacked predictions against the archive's MANIFEST.json.

Usage (after unzipping predictions.zip into this directory):
    uv run scripts/verify_predictions.py
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prediction-dir", type=Path, default=ROOT / "prediction_data")
    args = parser.parse_args()

    manifest_path = args.prediction_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{manifest_path} not found; unzip predictions.zip into {ROOT} first")
    manifest = json.loads(manifest_path.read_text())
    base = args.prediction_dir.parent
    bad = [
        relative
        for relative, expected in manifest["files"].items()
        if not (base / relative).is_file()
        or hashlib.sha256((base / relative).read_bytes()).hexdigest() != expected
    ]
    if bad:
        raise SystemExit(f"{len(bad)} prediction files are missing or differ, e.g. {bad[:3]}")
    print(f"all {len(manifest['files'])} prediction files verified")


if __name__ == "__main__":
    main()
