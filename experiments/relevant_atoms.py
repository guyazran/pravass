from typing import Dict, Set, Tuple

from unified_planning.shortcuts import Problem, InstantaneousAction


NON_VISIBLE_FLUENTS = {
    "isclean",
    "ishot",
    "iscool",
    "isvisibleobject",
    "isvisiblereceptacle",
}


def find_dynamic_fluent_symbols(problem: Problem) -> Set[str]:
    """Return fluent symbol names that appear as effect targets in any action.
    Fluents NOT in this set are static (fixed from the initial state)."""
    fve = problem.environment.free_vars_extractor
    dynamic: Set[str] = set()
    for a in problem.actions:
        for eff in a.effects:
            for atom in fve.get(eff.fluent):
                dynamic.add(atom.fluent().name)
    return dynamic


def find_goal_relevant_fluent_symbols(problem: Problem) -> Set[str]:
    """
    Compute the set of fluent symbol names that are backward-relevant
    to achieving the goals of the given unified_planning Problem.

    Works on lifted action schemas — no grounding required.

    Algorithm:
    1. Seed with fluent symbols appearing in the goals.
    2. Repeatedly: if an action schema has any relevant fluent symbol in
       its effect targets, mark all fluent symbols in its preconditions
       and effect conditions as relevant.
    3. Stop when a fixpoint is reached (no new symbols added).
    """
    env = problem.environment
    fve = env.free_vars_extractor

    def extract_fluent_names(*expressions) -> Set[str]:
        names: Set[str] = set()
        for e in expressions:
            if e is None:
                continue
            for atom in fve.get(e):
                names.add(atom.fluent().name)
        return names

    # Seed with fluent symbols from goals
    relevant: Set[str] = extract_fluent_names(*problem.goals)

    changed = True
    while changed:
        changed = False
        for a in problem.actions:
            # Check if any effect target mentions a relevant fluent symbol
            effect_symbols: Set[str] = set()
            for eff in a.effects:
                effect_symbols |= extract_fluent_names(eff.fluent)

            if not effect_symbols & relevant:
                continue

            # Action is relevant — collect symbols from preconditions and
            # effect conditions
            new_symbols: Set[str] = set()
            if isinstance(a, InstantaneousAction):
                new_symbols |= extract_fluent_names(*a.preconditions)
                for eff in a.effects:
                    new_symbols |= extract_fluent_names(eff.condition)

            new_symbols -= relevant
            if new_symbols:
                relevant |= new_symbols
                changed = True

    return relevant


# Maps a dynamic predicate to the static predicate that gates it:
# only items where the static predicate is true can have the dynamic one.
UNARY_PREDICATE_REQUIREMENTS = {
    "issliced": "sliceable",
    "holds": "pickupable",
    "istoggled": "toggleable",
}


def build_type_constraint_lookups(
    gt_state: Dict[str, bool],
) -> Tuple[Set[Tuple[str, str]], Set[str]]:
    """Parse static predicates from gt_state and return:
    - valid_inreceptacle_pairs: set of (item, receptacle) pairs allowed by cancontain
    - valid_unary: dict mapping predicate name -> set of items that satisfy
      the required static predicate (e.g. holds -> items where pickupable is true)
    """
    obj_type: Dict[str, Set[str]] = {}
    rec_type: Dict[str, Set[str]] = {}
    can_contain: Set[Tuple[str, str]] = set()
    # items where a given static predicate is true
    static_true: Dict[str, Set[str]] = {}

    for k, v in gt_state.items():
        if not v:
            continue
        name, args_str = k.split("(", 1)
        args = args_str.rstrip(")").split(",")
        if name == "objecttype":
            obj_type.setdefault(args[0], set()).add(args[1])
        elif name == "receptacletype":
            rec_type.setdefault(args[0], set()).add(args[1])
        elif name == "cancontain":
            can_contain.add((args[0], args[1]))
        else:
            static_true.setdefault(name, set()).add(args[0])

    # Build valid inreceptacle pairs
    valid_inreceptacle: Set[Tuple[str, str]] = set()
    for item, otypes in obj_type.items():
        for rec, rtypes in rec_type.items():
            for rt in rtypes:
                for ot in otypes:
                    if (rt, ot) in can_contain:
                        valid_inreceptacle.add((item, rec))
                        break
                else:
                    continue
                break

    # Build valid items per unary predicate
    valid_unary: Dict[str, Set[str]] = {}
    for pred, required in UNARY_PREDICATE_REQUIREMENTS.items():
        valid_unary[pred] = static_true.get(required, set())

    return valid_inreceptacle, valid_unary


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from unified_planning.io import PDDLReader

    parser = argparse.ArgumentParser(
        description="Filter gt_state.json to only goal-relevant fluents."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("split_data"),
        help="Root directory containing sample subdirectories (default: split_data)",
    )
    parser.add_argument(
        "--output-name", type=str, default="gt_state_relevant.json",
        help="Output file name written in each sample directory (default: gt_state_relevant.json)",
    )
    parser.add_argument(
        "--include-static", action="store_true",
        help="Include static predicates (those never modified by any action)",
    )
    parser.add_argument(
        "--include-non-visible", action="store_true",
        help="Include non-visible fluents (e.g. isclean, ishot, iscool)",
    )
    parser.add_argument(
        "--visible-only", action="store_true",
        help="Only keep predicates whose arguments are all currently visible (alf domain only)",
    )
    parser.add_argument(
        "--domains", nargs="+",
        help="Domains to process (default: all found in input dir)",
        default=('alf', 'gw', 'prb'),
    )
    cli_args = parser.parse_args()

    # Collect all sample dirs and figure out their domain name
    samples = []
    for split_dir in sorted(cli_args.input_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for sample_dir in sorted(split_dir.rglob("sample_*")):
            if not sample_dir.is_dir():
                continue
            gt_state_path = sample_dir / "gt_state.json"
            domain_path = sample_dir / "domain.pddl"
            problem_path = sample_dir / "problem.pddl"
            if not (gt_state_path.exists() and domain_path.exists() and problem_path.exists()):
                continue
            # domain is the first directory after the split dir (e.g. alf, gw, prb)
            relative = sample_dir.relative_to(split_dir)
            domain_name = relative.parts[0]
            if domain_name not in cli_args.domains:
                continue
            samples.append((sample_dir, domain_name))

    from tqdm import tqdm
    pbar = tqdm(samples, desc="Starting")

    for sample_dir, domain_name in pbar:
        pbar.set_description(domain_name)

        with open(sample_dir / "gt_state.json") as f:
            gt_state = json.load(f)

        reader = PDDLReader()
        problem = reader.parse_problem(
            str(sample_dir / "domain.pddl"),
            str(sample_dir / "problem.pddl"),
        )

        relevant_symbols = find_goal_relevant_fluent_symbols(problem)
        if not cli_args.include_static:
            relevant_symbols &= find_dynamic_fluent_symbols(problem)
        if not cli_args.include_non_visible:
            relevant_symbols -= NON_VISIBLE_FLUENTS

        valid_inreceptacle, valid_unary = build_type_constraint_lookups(gt_state)

        apply_visible_only = cli_args.visible_only and domain_name.lower() == "alf"

        # Collect visible objects/receptacles if needed (alf domain only)
        visible: Set[str] = set()
        if apply_visible_only:
            for k, v in gt_state.items():
                if not v:
                    continue
                name, args_str = k.split("(", 1)
                if name in ("isvisibleobject", "isvisiblereceptacle"):
                    visible.add(args_str.rstrip(")"))

        filtered_state = {}
        for k, v in gt_state.items():
            name, args_str = k.split("(", 1)
            if name not in relevant_symbols:
                continue
            pred_args = args_str.rstrip(")").split(",")
            if name == "inreceptacle":
                if (pred_args[0], pred_args[1]) not in valid_inreceptacle:
                    continue
            elif name in valid_unary:
                if pred_args[0] not in valid_unary[name]:
                    continue
            if apply_visible_only and name != "holdsany":
                if not all(a in visible for a in pred_args):
                    continue
            filtered_state[k] = v

        out_path = sample_dir / cli_args.output_name
        with open(out_path, "w") as f:
            json.dump(filtered_state, f, indent=2)
