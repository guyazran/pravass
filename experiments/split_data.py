"""Create the test and validation splits from raw_data.zip (paper App. F).

Each domain is split per stratum with a seeded shuffle, so the splits are
deterministic:

- PRB: 50 test + 10 validation instances per (objects N, horizon H) cell;
  the dataset holds more instances per cell, and the rest are unused.
- GW: 50 validation instances per horizon H; the rest (150 per H) are test.
- ALF: 2000 test + 1000 validation instances, sampled from all tasks.

That gives 3800 test instances (PRB 1200, GW 600, ALF 2000) and 1440
validation instances (PRB 240, GW 200, ALF 1000). The validation split fits
the Platt scaling of the VLM probabilities and picks the threshold of the
Feature Abstention baseline; the experiments run on the test split.

Outputs:
- <output_dir>/split_manifest.json
- <output_dir>/test_samples.txt
- <output_dir>/validation_samples.txt
- unless --materialization none, links (or copies) of the sample folders:
  - <output_dir>/test/<domain>/<stratum>/sample_xxxxxx
  - <output_dir>/validation/<domain>/<stratum>/sample_xxxxxx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

PRB_HORIZONS = (1, 2, 4, 6)
PRB_OBJECTS = (3, 4, 5, 6, 7, 8)
GW_HORIZONS = (1, 2, 4, 6)

PRB_TEST_PER_CELL = 50
PRB_VAL_PER_CELL = 10
PRB_USED_PER_CELL = PRB_TEST_PER_CELL + PRB_VAL_PER_CELL

GW_VAL_PER_H = 50

ALF_TEST = 2000
ALF_VAL = 1000
ALF_USED = ALF_TEST + ALF_VAL

SPLIT_NAMES = ("test", "validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split raw_data into test/validation according to the paper-adjusted rules."
    )
    parser.add_argument("--zip-path", type=Path, default=Path("raw_data.zip"))
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("raw_data"),
        help="Location of extracted raw data (auto-extracted from --zip-path if missing).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("split_data"))
    parser.add_argument("--seed", type=int, default=0, help="Deterministic split seed.")
    parser.add_argument(
        "--materialization",
        choices=("symlink", "copy", "none"),
        default="symlink",
        help="How to materialize split folders under output-dir/test and output-dir/validation.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract raw_data from the zip even if raw-dir already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-dir if it exists.",
    )
    return parser.parse_args()


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def shuffled(items: Iterable[Path], seed: int, key: str) -> list[Path]:
    arr = list(items)
    rng = random.Random(stable_seed(seed, key))
    rng.shuffle(arr)
    return arr


def sample_dirs(parent: Path) -> list[Path]:
    if not parent.exists():
        raise FileNotFoundError(f"Missing directory: {parent}")
    return sorted(
        p for p in parent.iterdir() if p.is_dir() and p.name.startswith("sample_")
    )


def ensure_raw_data_available(raw_dir: Path, zip_path: Path, force_extract: bool) -> None:
    if raw_dir.exists() and any(raw_dir.iterdir()) and not force_extract:
        print(f"[info] Using existing extracted data at: {raw_dir}")
        return

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    if raw_dir.exists() and force_extract:
        print(f"[info] Removing existing raw data dir: {raw_dir}")
        shutil.rmtree(raw_dir)

    extract_parent = raw_dir.parent
    extracted_default = extract_parent / "raw_data"

    if extracted_default.exists() and force_extract:
        shutil.rmtree(extracted_default)

    print(f"[info] Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            name
            for name in zf.namelist()
            if name.startswith("raw_data/") and "__MACOSX" not in name
        ]
        zf.extractall(path=extract_parent, members=members)

    if not extracted_default.exists():
        raise RuntimeError("Extraction finished but raw_data/ was not found in archive.")

    if extracted_default.resolve() != raw_dir.resolve():
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        extracted_default.rename(raw_dir)

    print(f"[info] Extraction complete: {raw_dir}")


def build_splits(raw_dir: Path, seed: int) -> dict:
    test: list[Path] = []
    validation: list[Path] = []
    unused: list[Path] = []

    domain_stats: dict[str, dict[str, dict[str, int]]] = {
        "prb": {},
        "gw": {},
        "alf": {},
    }

    # PRB: 60 instances per cell -> 50 test, 10 validation; the rest are unused.
    for h in PRB_HORIZONS:
        for n in PRB_OBJECTS:
            slug = f"h{h:02d}_n{n:02d}"
            stratum_dir = raw_dir / "prb" / slug
            samples = sample_dirs(stratum_dir)
            if len(samples) < PRB_USED_PER_CELL:
                raise ValueError(
                    f"PRB {slug}: expected at least {PRB_USED_PER_CELL} samples, got {len(samples)}"
                )

            shuffled_samples = shuffled(samples, seed, f"prb/{slug}")
            selected = shuffled_samples[:PRB_USED_PER_CELL]

            val_split = sorted(selected[:PRB_VAL_PER_CELL])
            test_split = sorted(selected[PRB_VAL_PER_CELL:])
            dropped = sorted(shuffled_samples[PRB_USED_PER_CELL:])

            validation.extend(val_split)
            test.extend(test_split)
            unused.extend(dropped)

            domain_stats["prb"][slug] = {
                "available": len(samples),
                "used": len(selected),
                "test": len(test_split),
                "validation": len(val_split),
                "unused": len(dropped),
            }

    # GW: 50 validation instances per horizon; all others are test.
    for h in GW_HORIZONS:
        slug = f"h{h:02d}"
        stratum_dir = raw_dir / "gw" / slug
        samples = sample_dirs(stratum_dir)
        if len(samples) <= GW_VAL_PER_H:
            raise ValueError(
                f"GW {slug}: expected more than {GW_VAL_PER_H} samples, got {len(samples)}"
            )

        shuffled_samples = shuffled(samples, seed, f"gw/{slug}")
        val_split = sorted(shuffled_samples[:GW_VAL_PER_H])
        test_split = sorted(shuffled_samples[GW_VAL_PER_H:])

        validation.extend(val_split)
        test.extend(test_split)

        domain_stats["gw"][slug] = {
            "available": len(samples),
            "used": len(samples),
            "test": len(test_split),
            "validation": len(val_split),
            "unused": 0,
        }

    # ALF: 2000 test + 1000 validation instances from all tasks.
    alf_dir = raw_dir / "alf" / "all"
    if alf_dir.exists():
        samples = sample_dirs(alf_dir)
        if len(samples) < ALF_USED:
            raise ValueError(
                f"ALF all: expected at least {ALF_USED} samples, got {len(samples)}"
            )

        shuffled_samples = shuffled(samples, seed, "alf/all")
        selected = shuffled_samples[:ALF_USED]
        val_split = sorted(selected[:ALF_VAL])
        test_split = sorted(selected[ALF_VAL:])
        dropped = sorted(shuffled_samples[ALF_USED:])

        validation.extend(val_split)
        test.extend(test_split)
        unused.extend(dropped)

        domain_stats["alf"]["all"] = {
            "available": len(samples),
            "used": len(selected),
            "test": len(test_split),
            "validation": len(val_split),
            "unused": len(dropped),
        }

    test_rel = sorted(p.relative_to(raw_dir).as_posix() for p in test)
    val_rel = sorted(p.relative_to(raw_dir).as_posix() for p in validation)
    unused_rel = sorted(p.relative_to(raw_dir).as_posix() for p in unused)

    return {
        "seed": seed,
        "rules": {
            "prb": {
                "test_per_cell": PRB_TEST_PER_CELL,
                "validation_per_cell": PRB_VAL_PER_CELL,
            },
            "gw": {"validation_per_h": GW_VAL_PER_H, "test_per_h": "remaining"},
            "alf": {"test": ALF_TEST, "validation": ALF_VAL},
        },
        "counts": {
            "test": len(test_rel),
            "validation": len(val_rel),
            "unused": len(unused_rel),
        },
        "domain_stats": domain_stats,
        "splits": {
            "test": test_rel,
            "validation": val_rel,
            "unused": unused_rel,
        },
    }


def write_split_lists(output_dir: Path, splits: dict[str, list[str]]) -> None:
    for name in SPLIT_NAMES:
        out_file = output_dir / f"{name}_samples.txt"
        out_file.write_text("\n".join(splits[name]) + "\n", encoding="utf-8")


def domain_split_counts(manifest: dict) -> dict[str, dict[str, int]]:
    domain_stats = manifest.get("domain_stats", {})
    return {
        domain: {
            name: sum(int(cell.get(name, 0)) for cell in cells.values())
            for name in SPLIT_NAMES
        }
        for domain, cells in domain_stats.items()
    }


def materialize_split(
    raw_dir: Path,
    output_dir: Path,
    split_name: str,
    rel_paths: list[str],
    mode: str,
) -> None:
    split_root = output_dir / split_name

    for rel in rel_paths:
        src = raw_dir / rel
        dst = split_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if mode == "symlink":
            target = os.path.relpath(src, start=dst.parent)
            os.symlink(target, dst, target_is_directory=True)
        elif mode == "copy":
            shutil.copytree(src, dst)
        else:
            raise ValueError(f"Unknown materialization mode: {mode}")


def main() -> None:
    args = parse_args()

    ensure_raw_data_available(args.raw_dir, args.zip_path, args.force_extract)

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {args.output_dir}\n"
                f"Re-run with --overwrite to replace it."
            )
        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_splits(args.raw_dir, args.seed)

    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    split_paths = {name: manifest["splits"][name] for name in SPLIT_NAMES}
    write_split_lists(args.output_dir, split_paths)

    if args.materialization != "none":
        print(f"[info] Materializing splits via {args.materialization} ...")
        for name in SPLIT_NAMES:
            materialize_split(
                args.raw_dir,
                args.output_dir,
                name,
                split_paths[name],
                args.materialization,
            )

    print("[done] Split creation complete.")
    print(f"       test samples:       {manifest['counts']['test']}")
    print(f"       validation samples: {manifest['counts']['validation']}")
    print(f"       unused samples:     {manifest['counts']['unused']}")

    print("       samples by domain:")
    for domain, counts in domain_split_counts(manifest).items():
        print(
            f"         - {domain}: test={counts['test']}, validation={counts['validation']}"
        )

    print(f"       manifest:           {manifest_path}")
    for name in SPLIT_NAMES:
        print(f"       {name} list: {args.output_dir / f'{name}_samples.txt'}")


if __name__ == "__main__":
    main()
