#!/usr/bin/env python3
"""Extract FD invariant-finder constraints for a split's samples.

This script is PDDL-only: it processes every sample listed in
``split_data/<split>_samples.txt`` and writes
``constraints_data/<split>/<domain>/<configuration>/<sample>/constraints.json``.
Constraints depend only on each sample's domain.pddl and problem.pddl — no
predictions, no probabilities, no normalizer. The experiments compute the
normalizers at run time (``mbs_experiments/normalizers.py``).

By default constraints are kept over the FD reachable grounded atoms. Use
``--project-to-relevant`` to intersect them with the keys of
``gt_state_relevant.json``.

The extracted constraints are grounded at-most-one groups:

* ``constraints``: all discovered/generated at-most-one groups. These groups can
  overlap and are the stronger validity constraints.
* ``selected_groups``: a pairwise-disjoint subset/cover used for the cheap
  closed-form normalizer. For FD this is the SAS+ group cover; for ALF we also
  add a generated lifted gripper-capacity group because FD finds no useful ALF
  groups in this benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from tqdm import tqdm

SPLIT_DATA_DIR = Path("split_data")
TEST_DIR = SPLIT_DATA_DIR / "test"
TEST_SAMPLES_FILE = SPLIT_DATA_DIR / "test_samples.txt"
CONSTRAINT_OUTPUT_DIR = Path("constraints_data")
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class MutexGroup:
    """A grounded at-most-one constraint over a list of atom keys."""

    atoms: Tuple[str, ...]
    source: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": "at_most_one",
            "source": self.source,
            "size": len(self.atoms),
            "atoms": list(self.atoms),
        }


@dataclass(frozen=True)
class ConstraintExample:
    metadata: dict[str, Any]
    relevant_state: dict[str, bool]
    full_state: dict[str, bool]
    domain: str
    configuration: str
    sample: str
    split_relative_path: str
    sample_path: str
    domain_path: str
    problem_path: str


@dataclass(frozen=True)
class ConstraintResult:
    output_path: Path
    status: str


# ---------------------------------------------------------------------------
# Fast Downward extraction
# ---------------------------------------------------------------------------


def _fd_translate_bin_dir() -> Path:
    """Return the bundled Fast Downward translator import directory."""
    try:
        import up_fast_downward  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Could not import up_fast_downward. Install with "
            "`uv add 'unified-planning[fast-downward]'` or run through uv."
        ) from exc

    root = Path(up_fast_downward.__file__).resolve().parent
    bin_dir = root / "downward" / "builds" / "release" / "bin"
    if not (bin_dir / "translate" / "invariant_finder.py").exists():
        raise RuntimeError(f"Fast Downward translator not found under {bin_dir}")
    return bin_dir


def _ensure_fd_on_path() -> None:
    bin_dir = str(_fd_translate_bin_dir())
    if bin_dir not in sys.path:
        sys.path.insert(0, bin_dir)


def atom_key(atom: Any) -> str:
    """Convert a FD pddl.Atom to the repo's JSON key style: pred(a,b)."""
    args = tuple(getattr(atom, "args", ()))
    return f"{atom.predicate}({','.join(args)})"


def is_auxiliary_atom_key(key: str) -> bool:
    """Heuristic filter for FD-introduced auxiliary/derived atoms."""
    pred = key.split("(", 1)[0]
    return "@" in pred or pred.startswith("new-axiom") or pred.startswith("derived!")


def _dedupe_groups(groups: Iterable[MutexGroup]) -> List[MutexGroup]:
    seen: Set[Tuple[str, ...]] = set()
    out: List[MutexGroup] = []
    for group in groups:
        atoms = tuple(sorted(dict.fromkeys(group.atoms)))
        if len(atoms) < 2:
            continue
        if atoms in seen:
            continue
        seen.add(atoms)
        out.append(MutexGroup(atoms=atoms, source=group.source))
    return out


def _filter_group_atoms(
    group: Sequence[Any],
    *,
    allowed_atoms: Optional[Set[str]] = None,
    include_auxiliary: bool = False,
) -> Tuple[str, ...]:
    atoms: List[str] = []
    for atom in group:
        key = atom_key(atom)
        if not include_auxiliary and is_auxiliary_atom_key(key):
            continue
        if allowed_atoms is not None and key not in allowed_atoms:
            continue
        atoms.append(key)
    return tuple(sorted(dict.fromkeys(atoms)))


def extract_fd_mutex_groups(
    domain: Path,
    problem: Path,
    *,
    invariant_generation_max_candidates: int = 100_000,
    invariant_generation_max_time: int = 300,
    keep_unreachable_facts: bool = False,
    include_auxiliary: bool = False,
    allowed_atoms: Optional[Set[str]] = None,
    include_fd_log: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run FD translator invariant discovery and return JSON-serializable data.

    Extraction happens before FD's goal-relevance pruning, so the groups are
    task-wide modulo FD's initial-state reachability analysis.
    """

    _ensure_fd_on_path()
    from translate import fact_groups, instantiate, normalize, pddl_parser  # type: ignore
    from translate.options import set_options  # type: ignore

    domain = domain.resolve()
    problem = problem.resolve()

    fd_args = [
        str(domain),
        str(problem),
        "--invariant-generation-max-candidates",
        str(invariant_generation_max_candidates),
        "--invariant-generation-max-time",
        str(invariant_generation_max_time),
        # These matter for full SAS emission and document the intended mode:
        # no goal-specific variable dropping.
        "--keep-unimportant-variables",
        "--skip-variable-reordering",
    ]
    if keep_unreachable_facts:
        fd_args.append("--keep-unreachable-facts")

    def run_fd_pipeline() -> Tuple[bool, Any, Set[Any], List[Any], Any, List[List[Any]], List[List[Any]], List[List[str]]]:
        set_options(fd_args)
        task = pddl_parser.open(domain_filename=str(domain), task_filename=str(problem))
        normalize.normalize(task)
        relaxed_reachable, atoms, actions, goal_list, _axioms, reachable_action_params = instantiate.explore(task)

        # Even if the problem goal has no relaxed solution, the reachable atoms,
        # grounded actions, and reachable action parameters are still useful for
        # task-state invariant extraction. FD's full translator would emit a
        # dummy unsolvable SAS task at this point, but that would throw away the
        # mutex information we need for MBS validity constraints.
        selected_groups, all_mutex_groups, translation_key = fact_groups.compute_groups(
            task, atoms, reachable_action_params, negative_in_goal=set()
        )
        return relaxed_reachable, task, atoms, actions, goal_list, selected_groups, all_mutex_groups, translation_key

    fd_log: Optional[str]
    if verbose:
        relaxed_reachable, task, atoms, actions, goal_list, selected_groups, all_mutex_groups, translation_key = run_fd_pipeline()
        fd_log = None
    else:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            relaxed_reachable, task, atoms, actions, goal_list, selected_groups, all_mutex_groups, translation_key = run_fd_pipeline()
        fd_log = buf.getvalue() if include_fd_log else None

    reachable_atom_keys = sorted(
        key
        for key in (atom_key(a) for a in atoms)
        if (include_auxiliary or not is_auxiliary_atom_key(key))
        and (allowed_atoms is None or key in allowed_atoms)
    )

    all_groups = _dedupe_groups(
        MutexGroup(
            atoms=_filter_group_atoms(g, allowed_atoms=allowed_atoms, include_auxiliary=include_auxiliary),
            source="fd_all_mutex_groups",
        )
        for g in all_mutex_groups
    )
    selected = _dedupe_groups(
        MutexGroup(
            atoms=_filter_group_atoms(g, allowed_atoms=allowed_atoms, include_auxiliary=include_auxiliary),
            source="fd_selected_sas_groups",
        )
        for g in selected_groups
    )

    result: Dict[str, Any] = {
        "domain_path": str(domain),
        "problem_path": str(problem),
        "semantics": {
            "constraint_type": "at_most_one",
            "all_mutex_groups": "All grounded FD mutex groups before SAS+ group-cover selection; may overlap.",
            "selected_groups": "Disjoint FD SAS+ group cover; weaker but factorizes for normalization.",
            "goal_filtering": "Not applied: extraction occurs before FD variable-order/unimportant-variable pruning.",
            "reachability": "FD invariants are instantiated over facts reachable from the supplied initial state unless --keep-unreachable-facts is used.",
        },
        "stats": {
            "relaxed_reachable": bool(relaxed_reachable),
            "reachable_atoms": len(reachable_atom_keys),
            "ground_actions": len(actions),
            "goal_literals": 0 if goal_list is None else len(goal_list),
            "all_mutex_groups": len(all_groups),
            "selected_mutex_groups": len(selected),
            "all_mutex_group_atoms_total": sum(len(g.atoms) for g in all_groups),
            "selected_mutex_group_atoms_total": sum(len(g.atoms) for g in selected),
        },
        "atoms": reachable_atom_keys,
        "constraints": [g.to_json() for g in all_groups],
        "selected_groups": [g.to_json() for g in selected],
        "fd_translation_key": translation_key,
    }
    if fd_log is not None:
        result["fd_log"] = fd_log
    return result


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def state_satisfies_mutex_groups(state: Mapping[str, bool], constraints: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether a Boolean state satisfies all at-most-one constraints."""
    for c in constraints:
        if c.get("type") != "at_most_one":
            raise ValueError(f"Unsupported constraint type: {c.get('type')}")
        if sum(1 for atom in c["atoms"] if state.get(atom, False)) > 1:
            return False
    return True


def parse_atom_key(key: str) -> Tuple[str, Tuple[str, ...]]:
    """Parse repo-style atom keys such as ``pred(a,b)``."""
    if "(" not in key or not key.endswith(")"):
        raise ValueError(f"Invalid atom key: {key!r}")
    pred, rest = key.split("(", 1)
    args_s = rest[:-1]
    if not args_s:
        return pred, ()
    return pred, tuple(args_s.split(","))


def _constraint_dict(atoms: Iterable[str], source: str) -> Dict[str, Any]:
    atom_tuple = tuple(sorted(dict.fromkeys(atoms)))
    return {
        "type": "at_most_one",
        "source": source,
        "size": len(atom_tuple),
        "atoms": list(atom_tuple),
    }


def _dedupe_constraint_dicts(constraints: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, ...]] = set()
    out: List[Dict[str, Any]] = []
    for constraint in constraints:
        if constraint.get("type") != "at_most_one":
            raise ValueError(f"Unsupported constraint type: {constraint.get('type')}")
        atoms = tuple(sorted(dict.fromkeys(map(str, constraint["atoms"]))))
        if len(atoms) < 2:
            continue
        if atoms in seen:
            continue
        seen.add(atoms)
        out.append({
            "type": "at_most_one",
            "source": str(constraint.get("source", "unknown")),
            "size": len(atoms),
            "atoms": list(atoms),
        })
    return out


def _refresh_constraint_stats(extraction: Dict[str, Any]) -> None:
    constraints = extraction["constraints"]
    selected_groups = extraction["selected_groups"]
    extraction["stats"].update({
        "all_mutex_groups": len(constraints),
        "selected_mutex_groups": len(selected_groups),
        "all_mutex_group_atoms_total": sum(len(g["atoms"]) for g in constraints),
        "selected_mutex_group_atoms_total": sum(len(g["atoms"]) for g in selected_groups),
    })


def add_alf_lifted_constraints(extraction: Dict[str, Any]) -> None:
    """Add conservative ALF lifted mutexes with a tractable selected-groups Z.

    FD does not recover useful ALF mutex groups for the current benchmark. A
    tempting invariant is "an item has at most one receptacle", but the ALFWorld
    state representation violates this in real data: objects may simultaneously
    be in/on multiple receptacles or supports (e.g. a pan on multiple burners, a
    mug in a coffee machine and on a counter). Similarly, many objects and
    receptacles may be visible in the stored state.

    The conservative lifted constraints below are therefore restricted to the
    gripper semantics that are supported by the action model and by the data:

    * selected_groups: at most one item is held (disjoint, product-form Z);
    * constraints-only: if an item is held, it is not in any receptacle.

    The second family overlaps through ``holds(o)`` and is therefore not added
    to ``selected_groups``. The saved selected-groups normalizer is exact and
    fast, but relaxed relative to the full constraints.
    """
    atom_keys = list(map(str, extraction.get("atoms", [])))
    held_atoms: List[str] = []
    inreceptacle_by_item: Dict[str, List[str]] = defaultdict(list)

    for atom in atom_keys:
        try:
            pred, args = parse_atom_key(atom)
        except ValueError:
            continue
        if pred == "holds" and len(args) == 1:
            held_atoms.append(atom)
        elif pred == "inreceptacle" and len(args) == 2:
            item, _receptacle = args
            inreceptacle_by_item[item].append(atom)

    selected_lifted: List[Dict[str, Any]] = []
    all_lifted: List[Dict[str, Any]] = []

    held_by_item = {}
    for held_atom in held_atoms:
        _pred, args = parse_atom_key(held_atom)
        held_by_item[args[0]] = held_atom

    if len(set(held_atoms)) >= 2:
        group = _constraint_dict(held_atoms, "alf_lifted_single_held_item")
        selected_lifted.append(group)
        all_lifted.append(group)

    for item, held_atom in sorted(held_by_item.items()):
        for rec_atom in sorted(set(inreceptacle_by_item.get(item, []))):
            all_lifted.append(
                _constraint_dict(
                    [held_atom, rec_atom],
                    "alf_lifted_held_not_in_receptacle",
                )
            )

    if not all_lifted and not selected_lifted:
        extraction.setdefault("constraint_generators", []).append({
            "name": "alf_lifted_mutexes",
            "enabled": True,
            "added_constraints": 0,
            "added_selected_groups": 0,
        })
        return

    extraction["constraints"] = _dedupe_constraint_dicts([
        *extraction["constraints"],
        *all_lifted,
    ])
    extraction["selected_groups"] = _dedupe_constraint_dicts([
        *extraction["selected_groups"],
        *selected_lifted,
    ])
    extraction.setdefault("constraint_generators", []).append({
        "name": "alf_lifted_mutexes",
        "enabled": True,
        "added_constraints": len(all_lifted),
        "added_selected_groups": len(selected_lifted),
        "selected_groups_note": "Only the disjoint global held-item mutex is added to selected_groups; held-vs-in-receptacle mutexes are constraints-only because they overlap.",
    })
    extraction["semantics"]["alf_lifted_constraints"] = (
        "Generated conservative lifted ALF mutexes for gripper capacity and held-vs-contained exclusions. "
        "The selected ALF lifted group is disjoint, so its normalizer is computed by the same closed-form product as FD selected groups."
    )
    _refresh_constraint_stats(extraction)



# ---------------------------------------------------------------------------
# Split / prediction data helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def load_allowed_atoms(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None:
        return None
    data = load_json(path)
    if isinstance(data, dict):
        return set(data.keys())
    if isinstance(data, list):
        return set(map(str, data))
    raise ValueError("--fluents-json must contain either a JSON object or a list of atom keys")


def load_test_samples(samples_file: Path = TEST_SAMPLES_FILE, test_dir: Path = TEST_DIR) -> dict[str, list[Path]]:
    """Load test sample paths grouped by domain."""
    domain_samples: dict[str, list[Path]] = defaultdict(list)
    with open(samples_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            domain = line.split("/")[0]
            domain_samples[domain].append(test_dir / line)
    return dict(domain_samples)


def get_split_relative_path(sample_dir: Path, split_root: Path = TEST_DIR) -> Path:
    """Return the sample path relative to the split root."""
    try:
        return sample_dir.relative_to(split_root)
    except ValueError as exc:
        raise ValueError(
            f"Sample directory {sample_dir} is not under split root {split_root}"
        ) from exc


def get_constraints_output_path(
    sample_dir: Path,
    split_root: Path = TEST_DIR,
    output_root: Path = CONSTRAINT_OUTPUT_DIR,
) -> Path:
    """Return the constraints JSON path: ``<output_root>/<split>/<sample_path>/constraints.json``.

    The split segment is derived from ``split_root.name`` so each split's
    constraints land in its own subtree (mirrors prediction_data layout).
    """
    relative_path = get_split_relative_path(sample_dir, split_root)
    return output_root / split_root.name / relative_path / "constraints.json"


def load_sample(sample_dir: Path, split_root: Path = TEST_DIR) -> ConstraintExample:
    """Load the non-image data needed for one sample."""
    metadata = load_json(sample_dir / "metadata.json")
    relevant_state = load_json(sample_dir / "gt_state_relevant.json")
    full_state = load_json(sample_dir / "gt_state.json")

    relative_path = get_split_relative_path(sample_dir, split_root)
    if len(relative_path.parts) < 3:
        raise ValueError(
            f"Expected sample path '<domain>/<configuration>/<sample>', got {relative_path}"
        )

    return ConstraintExample(
        metadata=metadata,
        relevant_state=relevant_state,
        full_state=full_state,
        domain=relative_path.parts[0],
        configuration=relative_path.parts[1],
        sample=relative_path.parts[-1],
        split_relative_path=relative_path.as_posix(),
        sample_path=str(sample_dir),
        domain_path=str(sample_dir / "domain.pddl"),
        problem_path=str(sample_dir / "problem.pddl"),
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so partial files are not left behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Per-sample and batch orchestration
# ---------------------------------------------------------------------------


def build_constraints_payload(
    *,
    example: ConstraintExample,
    extraction: Dict[str, Any],
    created_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "domain": example.domain,
        "configuration": example.configuration,
        "sample": example.sample,
        "split_relative_path": example.split_relative_path,
        "sample_path": example.sample_path,
        "metadata": example.metadata,
        "problem_path": example.problem_path,
        "domain_path": example.domain_path,
        "relevant_atoms": list(example.relevant_state.keys()),
        "ground_truth_relevant": example.relevant_state,
        "ground_truth_full": example.full_state,
        **extraction,
    }


def _set_ground_fluent(problem: Any, fluent_name: str, args: Sequence[Any], value: bool) -> None:
    fluent = problem.fluent(fluent_name)
    fnode = problem.environment.expression_manager.FluentExp(fluent, tuple(args))
    problem.set_initial_value(fnode, bool(value))


def _has_user_type(problem: Any, type_name: str) -> bool:
    try:
        problem.user_type(type_name)
        return True
    except Exception:
        return False


def _has_fluent(problem: Any, fluent_name: str) -> bool:
    try:
        problem.fluent(fluent_name)
        return True
    except Exception:
        return False


def _write_fd_seed_problem_for_constraints(example: ConstraintExample) -> Path | None:
    """Write a deterministic seed problem that exposes structural invariants to FD.

    FD invariant generation is tied to facts/actions reachable from the supplied
    initial state. Some benchmark problem files intentionally contain only a
    partial relevant state, so objects absent from that state can be pruned before
    FD ever emits their mutex groups. For domains where that happens, build a
    synthetic but label-independent initial state from the object universe. This
    uses only the domain/problem schema and object names, never the hidden ground
    truth, so the extracted constraints remain structural rather than instance-
    label leakage.
    """
    from tempfile import NamedTemporaryFile
    import re

    from unified_planning.io import PDDLReader, PDDLWriter

    reader = PDDLReader()
    problem = reader.parse_problem(example.domain_path, example.problem_path)
    seeded = False

    # Grocery World: place every item on a canonical table and make the hand empty.
    if (
        _has_user_type(problem, "item")
        and _has_user_type(problem, "table")
        and _has_fluent(problem, "on-table")
        and _has_fluent(problem, "robot-gripping")
        and _has_fluent(problem, "robot-gripper-empty")
    ):
        items = list(problem.objects(problem.user_type("item")))
        tables = list(problem.objects(problem.user_type("table")))
        if items and tables:
            for item in items:
                for table in tables:
                    _set_ground_fluent(problem, "on-table", (item, table), False)
                _set_ground_fluent(problem, "robot-gripping", (item,), False)
            _set_ground_fluent(problem, "robot-gripper-empty", (), True)
            canonical_table = sorted(tables, key=lambda obj: obj.name)[0]
            for item in items:
                _set_ground_fluent(problem, "on-table", (item, canonical_table), True)
            seeded = True

    # PRB/block stacking: put every shape separately on the table and clear it.
    elif (
        _has_user_type(problem, "shape")
        and _has_user_type(problem, "block")
        and _has_fluent(problem, "is-directly-on-table")
        and _has_fluent(problem, "is-clear-on-top")
        and _has_fluent(problem, "shape-on-top-of-block")
    ):
        shapes = list(problem.objects(problem.user_type("shape")))
        blocks = list(problem.objects(problem.user_type("block")))
        if shapes:
            for shape in shapes:
                _set_ground_fluent(problem, "is-directly-on-table", (shape,), True)
                _set_ground_fluent(problem, "is-clear-on-top", (shape,), True)
                for block in blocks:
                    _set_ground_fluent(problem, "shape-on-top-of-block", (shape, block), False)
            seeded = True

    if not seeded:
        return None

    with NamedTemporaryFile("w", suffix=".pddl", delete=False) as f:
        temp_path = Path(f.name)
    writer = PDDLWriter(problem)
    writer.write_problem(str(temp_path))

    # Some UP versions write a normalized domain name. Restore the exact domain
    # name from the source PDDL so FD's parser accepts the temporary problem.
    content = temp_path.read_text()
    domain_text = Path(example.domain_path).read_text()
    match = re.search(r"\(domain\s+([^\s\)]+)\)", domain_text, flags=re.IGNORECASE)
    if match:
        content = re.sub(
            r"\(:domain\s+[^\s\)]+\)",
            f"(:domain {match.group(1)})",
            content,
            count=1,
            flags=re.IGNORECASE,
        )
        temp_path.write_text(content)
    return temp_path


def extract_constraints_sample(
    *,
    sample_dir: Path,
    split_root: Path = TEST_DIR,
    output_root: Path = CONSTRAINT_OUTPUT_DIR,
    invariant_generation_max_candidates: int = 100_000,
    invariant_generation_max_time: int = 300,
    keep_unreachable_facts: bool = False,
    include_auxiliary: bool = False,
    include_fd_log: bool = False,
    project_to_relevant: bool = False,
    enable_alf_lifted_constraints: bool = True,
    overwrite: bool = False,
) -> ConstraintResult:
    """Extract PDDL-only constraints for one sample."""
    output_path = get_constraints_output_path(sample_dir, split_root, output_root)
    if output_path.exists() and not overwrite:
        return ConstraintResult(output_path=output_path, status="skipped")

    example = load_sample(sample_dir, split_root)
    allowed_atoms = set(example.relevant_state.keys()) if project_to_relevant else None
    temp_prob_path = _write_fd_seed_problem_for_constraints(example)
    used_fd_seed = temp_prob_path is not None
    fd_problem_path = temp_prob_path or Path(example.problem_path)
    try:
        extraction = extract_fd_mutex_groups(
            Path(example.domain_path),
            fd_problem_path,
            invariant_generation_max_candidates=invariant_generation_max_candidates,
            invariant_generation_max_time=invariant_generation_max_time,
            keep_unreachable_facts=keep_unreachable_facts,
            include_auxiliary=include_auxiliary,
            allowed_atoms=allowed_atoms,
            include_fd_log=include_fd_log,
        )
    finally:
        if temp_prob_path is not None:
            temp_prob_path.unlink(missing_ok=True)
    extraction["projection"] = {
        "project_to_relevant": project_to_relevant,
        "relevant_source": "gt_state_relevant.json",
        "fd_seed_problem": "synthetic_complete_initial_state" if used_fd_seed else "original_problem",
        "note": "When false, constraints are extracted over FD reachable atoms rather than intersected with gt_state_relevant.json. For domains with partial initial states, a deterministic synthetic complete seed state is used so FD exposes object-level structural invariants without reading the hidden ground-truth labels.",
    }
    extraction.setdefault("constraint_generators", []).append({
        "name": "fast_downward_invariant_finder",
        "enabled": True,
    })
    if example.domain == "alf" and enable_alf_lifted_constraints:
        add_alf_lifted_constraints(extraction)
    elif example.domain == "alf":
        extraction.setdefault("constraint_generators", []).append({
            "name": "alf_lifted_mutexes",
            "enabled": False,
        })

    payload = build_constraints_payload(
        example=example,
        extraction=extraction,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_atomic(output_path, payload)
    return ConstraintResult(output_path=output_path, status="written")


def extract_constraints_domain(
    domain: str,
    sample_dirs: list[Path],
    *,
    split_root: Path,
    output_root: Path,
    overwrite: bool,
    limit: int | None,
    invariant_generation_max_candidates: int,
    invariant_generation_max_time: int,
    keep_unreachable_facts: bool,
    include_auxiliary: bool,
    include_fd_log: bool,
    project_to_relevant: bool,
    enable_alf_lifted_constraints: bool,
) -> None:
    """Extract constraints for one domain."""
    selected_samples = sample_dirs[:limit] if limit is not None else sample_dirs
    if not selected_samples:
        print(f"{domain}: no samples selected")
        return

    pending_samples = [
        sample_dir
        for sample_dir in selected_samples
        if overwrite or not get_constraints_output_path(sample_dir, split_root, output_root).exists()
    ]
    skipped = len(selected_samples) - len(pending_samples)
    if not pending_samples:
        print(f"{domain}: all {len(selected_samples)} selected samples already extracted")
        return

    print(f"\n{'=' * 60}")
    print(
        f"Extracting constraints for domain: {domain} "
        f"({len(pending_samples)} pending, {skipped} skipped)"
    )
    print(f"{'=' * 60}")

    written = 0
    failed = 0
    for sample_dir in tqdm(pending_samples, desc=f"{domain} constraints"):
        try:
            result = extract_constraints_sample(
                sample_dir=sample_dir,
                split_root=split_root,
                output_root=output_root,
                invariant_generation_max_candidates=invariant_generation_max_candidates,
                invariant_generation_max_time=invariant_generation_max_time,
                keep_unreachable_facts=keep_unreachable_facts,
                include_auxiliary=include_auxiliary,
                include_fd_log=include_fd_log,
                project_to_relevant=project_to_relevant,
                enable_alf_lifted_constraints=enable_alf_lifted_constraints,
                overwrite=overwrite,
            )
            if result.status == "written":
                written += 1
        except Exception as exc:
            failed += 1
            print(f"\nERROR extracting {sample_dir}: {exc}", file=sys.stderr)

    print(f"{domain}: wrote {written} constraints files, failed {failed}")
    if failed:
        raise RuntimeError(f"{domain}: failed on {failed} samples")


def run_batch(args: argparse.Namespace) -> None:
    for split in args.splits:
        split_root = args.split_data_dir / split
        samples_file = args.split_data_dir / f"{split}_samples.txt"
        if not samples_file.exists():
            raise FileNotFoundError(
                f"Missing samples manifest for split '{split}': {samples_file}"
            )

        domain_samples = load_test_samples(samples_file=samples_file, test_dir=split_root)
        print(f"\n[{split}] Domains found: {list(domain_samples.keys())}")
        for domain, samples in domain_samples.items():
            print(f"  {domain}: {len(samples)} {split} samples")

        for domain, sample_dirs in domain_samples.items():
            if domain not in args.domains:
                continue
            extract_constraints_domain(
                domain,
                sample_dirs,
                split_root=split_root,
                output_root=args.output_root,
                overwrite=args.overwrite,
                limit=args.limit,
                invariant_generation_max_candidates=args.max_candidates,
                invariant_generation_max_time=args.max_time,
                keep_unreachable_facts=args.keep_unreachable_facts,
                include_auxiliary=args.include_auxiliary,
                include_fd_log=args.include_fd_log,
                project_to_relevant=args.project_to_relevant,
                enable_alf_lifted_constraints=not args.no_alf_lifted_constraints,
            )

    print("\nDone! Constraint data saved to", args.output_root)


def run_single(args: argparse.Namespace) -> None:
    allowed_atoms = load_allowed_atoms(args.fluents_json)
    extraction = extract_fd_mutex_groups(
        args.domain,
        args.problem,
        invariant_generation_max_candidates=args.max_candidates,
        invariant_generation_max_time=args.max_time,
        keep_unreachable_facts=args.keep_unreachable_facts,
        include_auxiliary=args.include_auxiliary,
        allowed_atoms=allowed_atoms,
        include_fd_log=args.include_fd_log,
        verbose=args.verbose,
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **extraction,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Fast Downward invariant constraints (PDDL-only) for a split's samples. "
        "If DOMAIN and PROBLEM are supplied, run on a single PDDL task instead."
    )
    parser.add_argument("domain", nargs="?", type=Path, help="Optional single-mode PDDL domain file")
    parser.add_argument("problem", nargs="?", type=Path, help="Optional single-mode PDDL problem file")

    # Batch-style arguments.
    parser.add_argument("--split-data-dir", type=Path, default=SPLIT_DATA_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("test", "validation"),
        default=("test", "validation"),
        help=(
            "Splits to process; each split gets its own tree under the "
            "output root. Default: both splits."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=CONSTRAINT_OUTPUT_DIR)
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Domains to process in batch mode",
        default=("alf", "gw", "prb"),
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing constraints JSON files")
    parser.add_argument("--limit", type=int, help="Maximum number of samples to process per domain")
    parser.add_argument(
        "--project-to-relevant",
        action="store_true",
        help="Project extracted constraints to the keys of gt_state_relevant.json. By default constraints are kept over FD reachable atoms.",
    )
    parser.add_argument(
        "--no-alf-lifted-constraints",
        action="store_true",
        help="Disable the generated lifted ALF mutexes; keep only Fast Downward invariant-finder output.",
    )

    # FD extraction arguments.
    parser.add_argument(
        "--fluents-json",
        type=Path,
        help="Single-mode JSON object/list limiting constraints to a projected fluent set",
    )
    parser.add_argument("--include-auxiliary", action="store_true", help="Keep FD-introduced auxiliary/axiom atoms")
    parser.add_argument("--keep-unreachable-facts", action="store_true", help="Ask FD to keep facts unreachable from the supplied initial state")
    parser.add_argument("--max-candidates", type=int, default=100_000, help="FD invariant-generation candidate limit")
    parser.add_argument("--max-time", type=int, default=300, help="FD invariant-generation time limit in seconds")
    parser.add_argument("--include-fd-log", action="store_true", help="Persist the FD translator log in each JSON")
    parser.add_argument("--verbose", action="store_true", help="Single-mode: print FD translator log live")
    parser.add_argument("--output", "-o", type=Path, help="Single-mode output JSON path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.domain is None) ^ (args.problem is None):
        parser.error("DOMAIN and PROBLEM must be supplied together for single mode")
    if args.domain is not None and args.problem is not None:
        run_single(args)
    else:
        run_batch(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
