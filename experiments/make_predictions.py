import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from tqdm import tqdm

from s3e.semantic_state_estimator import SemanticStateEstimator
from s3e.translation import TemplateTranslator
from s3e.pddl.up_utils import create_up_problem, get_object_names_dict, get_pddl_strings
from s3e.vlm.backend import VLMBackend

OPENAI_MODEL = "OpenAI/gpt-4.1-2025-04-14"
GEMMA4_2B_MODEL = 'google/gemma-4-E2B-it'
SPLIT_DATA_DIR = Path("split_data")
SPLIT_CHOICES = ("test", "validation")
NL_TEMPLATES_DIR = Path("nl_templates")
PREDICTION_OUTPUT_DIR = Path("prediction_data")
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PredictionExample:
    images: list[Image.Image]
    metadata: dict[str, Any]
    image_files: list[str]
    state_dict: dict[str, bool]
    full_state_dict: dict[str, bool]
    domain: str
    configuration: str
    sample: str
    split_relative_path: str
    sample_path: str
    domain_path: str
    problem_path: str


@dataclass(frozen=True)
class PredictionResult:
    output_path: Path
    status: str


def load_domain_translator(domain: str) -> TemplateTranslator:
    """Load NL predicate templates for a domain and return a translator."""
    template_path = NL_TEMPLATES_DIR / f"{domain}_template.json"
    if not template_path.exists():
        raise FileNotFoundError(
            f"Missing template file for domain '{domain}': {template_path}"
        )
    with open(template_path) as f:
        templates = json.load(f)
    return TemplateTranslator(templates)


def get_split_root(split: str) -> Path:
    """Return the directory containing the samples for a given split."""
    return SPLIT_DATA_DIR / split


def get_split_samples_file(split: str) -> Path:
    """Return the path to the samples manifest for a given split."""
    return SPLIT_DATA_DIR / f"{split}_samples.txt"


def model_to_path_segment(model: str) -> str:
    """Encode a model identifier as a single path segment (e.g. 'OpenAI/foo' -> 'OpenAI__foo')."""
    return model.replace("/", "__")


def get_split_output_root(
    split: str,
    model: str,
    output_root: Path = PREDICTION_OUTPUT_DIR,
) -> Path:
    """Return the output subdirectory for a given split and model."""
    return output_root / split / model_to_path_segment(model)


def load_split_samples(split: str) -> dict[str, list[Path]]:
    """Load sample paths for a split, grouped by domain."""
    samples_file = get_split_samples_file(split)
    split_root = get_split_root(split)
    domain_samples: dict[str, list[Path]] = defaultdict(list)
    with open(samples_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            domain = line.split("/")[0]
            domain_samples[domain].append(split_root / line)
    return dict(domain_samples)


def get_split_relative_path(sample_dir: Path, split_root: Path) -> Path:
    """Return the sample path relative to the split root."""
    try:
        return sample_dir.relative_to(split_root)
    except ValueError as exc:
        raise ValueError(
            f"Sample directory {sample_dir} is not under split root {split_root}"
        ) from exc


def get_prediction_output_path(
    sample_dir: Path,
    split_root: Path,
    output_root: Path,
) -> Path:
    """Return the prediction JSON path that mirrors the split hierarchy."""
    relative_path = get_split_relative_path(sample_dir, split_root)
    return output_root / relative_path / "prediction.json"


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def load_sample(sample_dir: Path, split_root: Path) -> PredictionExample:
    """Load a single prediction example from a sample directory."""
    metadata = load_json(sample_dir / "metadata.json")

    image_files = list(metadata["image_files"])
    images = []
    for image_file in image_files:
        with Image.open(sample_dir / image_file) as image:
            images.append(image.copy())

    state_dict = load_json(sample_dir / "gt_state_relevant.json")
    full_state_dict = load_json(sample_dir / "gt_state.json")

    relative_path = get_split_relative_path(sample_dir, split_root)
    if len(relative_path.parts) < 3:
        raise ValueError(
            f"Expected sample path '<domain>/<configuration>/<sample>', got {relative_path}"
        )

    return PredictionExample(
        images=images,
        metadata=metadata,
        image_files=image_files,
        state_dict=state_dict,
        full_state_dict=full_state_dict,
        domain=relative_path.parts[0],
        configuration=relative_path.parts[1],
        sample=relative_path.parts[-1],
        split_relative_path=relative_path.as_posix(),
        sample_path=str(sample_dir),
        domain_path=str(sample_dir / "domain.pddl"),
        problem_path=str(sample_dir / "problem.pddl"),
    )


def get_vlm_kwargs(model: str, use_flash: bool = True) -> dict:
    """Construction kwargs for the VLM backend, by model family."""
    if model.startswith("OpenAI/"):
        return dict()
    return dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if use_flash else "sdpa",
    )


def get_inference_kwargs(model: str) -> dict:
    """Per-call inference kwargs for the estimator, by model family."""
    if model.startswith("OpenAI/"):
        return dict(max_completion_tokens=1)
    return dict(logits_to_keep=1)


def build_vlm(model: str, use_flash: bool = True) -> VLMBackend:
    """Construct the VLM backend once so model weights load a single time.

    The returned backend is shared across every per-domain/per-split
    estimator, so VRAM holds exactly one copy of the model for the run.
    """
    return SemanticStateEstimator._build_vlm_from_string(
        model, get_vlm_kwargs(model, use_flash)
    )


def build_estimator(
    domain: str,
    sample_dir: Path,
    vlm: VLMBackend,
    model: str = OPENAI_MODEL,
    batch_size: int = 1,
) -> SemanticStateEstimator:
    """Build an S3E estimator for a domain using a sample's PDDL files.

    ``vlm`` is a prebuilt backend (see :func:`build_vlm`). Passing the
    object rather than the model string makes ``SemanticStateEstimator``
    reuse the resident weights instead of reloading them; everything else
    constructed here (PDDL parse, prompt, translator) is cheap and CPU-only.
    """
    domain_pddl = str(sample_dir / "domain.pddl")
    problem_pddl = str(sample_dir / "problem.pddl")

    query_translator = load_domain_translator(domain)

    with open("prompts/system.md") as f:
        system_prompt = f.read()
    problem = create_up_problem(domain_pddl, problem_pddl)
    domain_str, _ = get_pddl_strings(problem)
    objects = get_object_names_dict(problem)
    objects_str = "\n".join(
        f"{key} type: {list(map(str, value))}"
        for key, value in objects.items()
    )
    system_prompt = system_prompt.format(domain=domain_str, objects=objects_str)

    with open(f"prompts/{domain}.md") as f:
        domain_instructions = f.read()

    return SemanticStateEstimator(
        domain=domain_pddl,
        problem=problem_pddl,
        vlm=vlm,
        query_translator=query_translator,
        system_prompt=system_prompt,
        additional_instructions=domain_instructions,
        null_tokens=["null", "Null", "NULL"],
        batch_size=batch_size,
        inference_kwargs=get_inference_kwargs(model),
    )


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and Paths returned by S3E into JSON-friendly data."""
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def detail_field(detail: Any, name: str) -> Any:
    if isinstance(detail, dict):
        return detail[name]
    return getattr(detail, name)


def build_prediction_payload(
    *,
    example: PredictionExample,
    # raw_outputs: dict[str, Any],
    details: dict[str, Any],
    model: str,
    created_at: str,
) -> dict[str, Any]:
    """Build the persisted prediction JSON payload (raw VLM outputs only)."""
    predictions: dict[str, dict[str, Any]] = {}
    for predicate in example.state_dict:
        detail = details[predicate]
        predictions[predicate] = {
            "ground_truth": example.state_dict[predicate],
            "raw_probability": detail_field(detail, "raw_probability"),
            "score": detail_field(detail, "score"),
            "raw_true_mass": detail_field(detail, "raw_true_mass"),
            "raw_false_mass": detail_field(detail, "raw_false_mass"),
            "raw_none_mass": detail_field(detail, "raw_none_mass"),
            "none_is_max_raw": detail_field(detail, "none_is_max_raw"),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "model": model,
        "domain": example.domain,
        "configuration": example.configuration,
        "sample": example.sample,
        "split_relative_path": example.split_relative_path,
        "sample_path": example.sample_path,
        "metadata": example.metadata,
        "image_files": example.image_files,
        "problem_path": example.problem_path,
        "domain_path": example.domain_path,
        "predicates": list(example.state_dict.keys()),
        "ground_truth_relevant": example.state_dict,
        "ground_truth_full": example.full_state_dict,
        # "raw_outputs": to_jsonable(raw_outputs),
        "predictions": to_jsonable(predictions),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so partial prediction files are not left behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def predict_sample(
    *,
    estimator: SemanticStateEstimator,
    domain: str,
    sample_dir: Path,
    split_root: Path,
    output_root: Path,
    overwrite: bool = False,
    model: str = OPENAI_MODEL,
    created_at: str | None = None,
) -> PredictionResult:
    """Run S3E prediction for one sample and write its prediction JSON."""
    output_path = get_prediction_output_path(sample_dir, split_root, output_root)
    if output_path.exists() and not overwrite:
        return PredictionResult(output_path=output_path, status="skipped")

    example = load_sample(sample_dir, split_root)
    if example.domain != domain:
        raise ValueError(
            f"Sample domain '{example.domain}' does not match requested domain '{domain}'"
        )
    predicates = list(example.state_dict.keys())

    estimator.swap_problem(example.domain_path, example.problem_path)
    raw_outputs = estimator.estimate_raw(example.images, predicates=predicates)
    details = estimator.prediction_details_from_raw(raw_outputs)

    payload = build_prediction_payload(
        example=example,
        # raw_outputs=raw_outputs,
        details=details,
        model=model,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
    )
    write_json_atomic(output_path, payload)
    return PredictionResult(output_path=output_path, status="written")


def predict_domain(
    domain: str,
    sample_dirs: list[Path],
    *,
    split: str,
    model: str,
    vlm: VLMBackend,
    overwrite: bool,
    limit: int | None,
    batch_size: int,
) -> None:
    """Run predictions for one domain within a given split using a given model."""
    split_root = get_split_root(split)
    output_root = get_split_output_root(split, model)

    selected_samples = sample_dirs[:limit] if limit is not None else sample_dirs
    if not selected_samples:
        print(f"[{split}] {domain}: no samples selected")
        return

    pending_samples = [
        sample_dir
        for sample_dir in selected_samples
        if overwrite
        or not get_prediction_output_path(
            sample_dir,
            split_root,
            output_root,
        ).exists()
    ]
    skipped = len(selected_samples) - len(pending_samples)
    if not pending_samples:
        print(
            f"[{split}] {domain}: all {len(selected_samples)} selected samples already predicted"
        )
        return

    print(f"\n{'=' * 60}")
    print(
        f"[{split}] Predicting domain: {domain} "
        f"({len(pending_samples)} pending, {skipped} skipped)"
    )
    print(f"{'=' * 60}")

    estimator = build_estimator(domain, pending_samples[0], vlm, model=model,
                                batch_size=batch_size)
    written = 0
    for sample_dir in tqdm(pending_samples, desc=f"[{split}] {domain} predicting"):
        result = predict_sample(
            estimator=estimator,
            domain=domain,
            sample_dir=sample_dir,
            split_root=split_root,
            output_root=output_root,
            overwrite=overwrite,
            model=model,
        )
        if result.status == "written":
            written += 1

    print(f"[{split}] {domain}: wrote {written} prediction files")


def main():
    parser = argparse.ArgumentParser(
        description="Run S3E predictions on the test and/or validation splits"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_CHOICES,
        default=["test", "validation"],
        help=(
            "Splits to predict. Outputs are written to "
            "prediction_data/<split>/<model>/."
        ),
    )
    parser.add_argument(
        "--model",
        default=GEMMA4_2B_MODEL,
        help=(
            "Model identifier passed to S3E as the VLM. The model name (with "
            "'/' replaced by '__') is used as a subdirectory under each split. "
            f"Default: {GEMMA4_2B_MODEL}."
        ),
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Domains to predict",
        default=("alf", "gw", "prb"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing prediction JSON files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of samples to process per domain",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of samples to predict in a batch. If not specified, samples "
            "are predicted one at a time. Batch prediction is more efficient for "
            "models that support it, but S3E's batch prediction API is currently "
            "experimental and may not work with all models or prompts."
        ),
    )
    parser.add_argument(
        "--use-flash",
        action="store_true",
        help=(
            "Whether to use flash attention if supported by the model and S3E. "
            "This can reduce VRAM usage and increase speed for some models, but "
            "may cause issues with others."
        ),
    )
    args = parser.parse_args()

    # Load the model once for the entire run; every per-domain/per-split
    # estimator shares this backend, so VRAM never holds more than one copy.
    vlm = build_vlm(args.model, use_flash=args.use_flash)

    for split in args.splits:
        domain_samples = load_split_samples(split)
        print(f"\n[{split}] Domains found: {list(domain_samples.keys())}")
        for domain, samples in domain_samples.items():
            print(f"  {domain}: {len(samples)} {split} samples")

        for domain, sample_dirs in domain_samples.items():
            if domain not in args.domains:
                continue
            predict_domain(
                domain,
                sample_dirs,
                split=split,
                model=args.model,
                vlm=vlm,
                overwrite=args.overwrite,
                limit=args.limit,
                batch_size=args.batch_size,
            )

    print("\nDone! Prediction data saved to", PREDICTION_OUTPUT_DIR)


if __name__ == "__main__":
    main()
