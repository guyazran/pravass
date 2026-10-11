"""Closed-world projection of a PDDL problem's initial state."""
from __future__ import annotations

import re
from collections.abc import Sequence

FACT_RE = re.compile(r"\(([a-z][a-z0-9_-]*)\s*([^()]*)\)")


def pddl_initial_truth(problem_text: str) -> frozenset[str]:
    """Return normalized atoms explicitly true in the PDDL ``:init`` section."""
    init_start = problem_text.find("(:init")
    if init_start < 0:
        raise ValueError("PDDL problem has no :init section")
    goal_start = problem_text.find("(:goal", init_start)
    if goal_start < 0:
        raise ValueError("PDDL problem has no :goal section after :init")
    init_text = problem_text[init_start:goal_start]
    atoms: set[str] = set()
    for name, raw_args in FACT_RE.findall(init_text):
        if name in {"not", "and"} or "?" in raw_args:
            continue
        args = tuple(raw_args.split())
        atoms.add(f"{name}({','.join(args)})")
    return frozenset(atoms)


def pddl_ground_truth(
    feature_names: Sequence[str],
    problem_text: str,
) -> dict[str, bool]:
    """Project closed-world PDDL truth onto an ordered modeled vocabulary."""
    true_atoms = pddl_initial_truth(problem_text)
    return {str(atom): str(atom) in true_atoms for atom in feature_names}
