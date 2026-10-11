"""Report how often a VLM put almost no probability on any answer token.

Each prediction stores the probability mass the VLM put on its answer tokens:
"yes", "no" and "null" in their spellings. When that mass is tiny, the model's
next token was something else, and the normalized probability of the feature is
noise. This script prints, for every split, model and domain under
``prediction_data/``, the share of answers whose answer-token mass is below a
threshold. Use it to check predictions you made yourself (Step 2, Option B).

Usage:
    uv run scripts/check_answer_mass.py [--threshold 0.01] [--splits test]
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def answer_mass(detail: dict) -> float:
    return sum(float(detail.get(key) or 0.0) for key in ("raw_true_mass", "raw_false_mass", "raw_none_mass"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prediction-dir", type=Path, default=ROOT / "prediction_data")
    parser.add_argument("--splits", nargs="+", default=["test", "validation"])
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()

    print(f"share of answers with less than {args.threshold:g} probability on every answer token")
    for split in args.splits:
        split_dir = args.prediction_dir / split
        if not split_dir.is_dir():
            continue
        for model_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cells = []
            for domain_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                total = low = 0
                for path in domain_dir.rglob("prediction.json"):
                    for detail in json.loads(path.read_text())["predictions"].values():
                        total += 1
                        low += answer_mass(detail) < args.threshold
                if total:
                    cells.append(f"{domain_dir.name} {low / total:6.1%}")
            print(f"{split:10s} {model_dir.name:32s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
