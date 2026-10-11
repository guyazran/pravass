from __future__ import annotations

import re
from typing import Any, Sequence

from mbs_experiments.states import (
    _component_exact_at_most_one_normalizer,
    at_most_one_group_probability,
    clamp_probability,
    exact_at_most_one_normalizer,
    project_groups,
    selected_groups_normalizer,
    visible_indices,
)

ATOM_RE = re.compile(r"^([^()]+)\((.*)\)$")


def _parse_atom(atom: str) -> tuple[str, tuple[str, ...]]:
    match = ATOM_RE.match(atom)
    if match is None:
        raise ValueError(f"Invalid atom: {atom!r}")
    raw_args = match.group(2)
    args = tuple(part for part in raw_args.split(",") if part)
    return match.group(1), args


def _exactly_one_group_probability(
    probabilities: Sequence[float], group: Sequence[int]
) -> float:
    if not group:
        return 0.0
    total = 0.0
    for true_idx in group:
        term = clamp_probability(float(probabilities[true_idx]))
        for idx in group:
            if idx != true_idx:
                term *= 1.0 - clamp_probability(float(probabilities[idx]))
        total += term
    return total


def selected_groups_product_normalizer(
    probabilities: Sequence[float],
    selected_groups: Sequence[Sequence[int]],
    *,
    exactly_one: bool = False,
) -> float:
    probabilities = tuple(clamp_probability(float(q)) for q in probabilities)

    if not exactly_one:
        return selected_groups_normalizer(probabilities, selected_groups)

    z = 1.0
    for group in selected_groups:
        z *= _exactly_one_group_probability(probabilities, group)
        if z == 0.0:
            return 0.0
    return z


def implied_selected_groups_normalizer(
    probabilities: Sequence[float],
    selected_groups: Sequence[Sequence[int]],
    constraint_groups: Sequence[Sequence[int]],
) -> float:
    """Relaxed normalizer over disjoint selected groups under exactly-one constraints.

    A selected group that is itself one of the exactly-one ``constraint_groups``
    keeps exactly-one semantics. A strict subset of one is implied to be only
    at-most-one, since all its features are false when the true one lies outside
    it, so it contributes its at-most-one mass. Every state that satisfies the
    constraint groups then satisfies the selected groups under these semantics,
    which keeps the result an upper bound on the exact normalizer (paper App. G.5).
    """
    probabilities = tuple(clamp_probability(float(q)) for q in probabilities)
    whole_groups = {frozenset(group) for group in constraint_groups}
    z = 1.0
    for group in selected_groups:
        if frozenset(group) in whole_groups:
            z *= _exactly_one_group_probability(probabilities, group)
        elif len(group) >= 2:
            z *= at_most_one_group_probability(group, probabilities)
        if z == 0.0:
            return 0.0
    return z


def _project_groups_preserving_singletons(
    selected_groups: Sequence[Sequence[int]],
    visible_idx: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    remap = {orig: new for new, orig in enumerate(visible_idx)}
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    for group in selected_groups:
        new_group = tuple(sorted(remap[i] for i in group if i in remap))
        if new_group and new_group not in seen:
            seen.add(new_group)
            out.append(new_group)
    return tuple(out)


def selected_groups_runtime_normalizer(
    bundle: Any,
    *,
    feature_space: str = "visible",
    exactly_one: bool = False,
) -> float:
    probabilities = tuple(bundle.probabilities)
    selected_groups = tuple(tuple(group) for group in bundle.selected_groups)

    if feature_space == "full":
        return selected_groups_product_normalizer(
            probabilities, selected_groups, exactly_one=exactly_one
        )
    if feature_space != "visible":
        raise ValueError(f"Unknown feature_space: {feature_space!r}")

    visible_idx = visible_indices(bundle.none_is_max)
    reduced_probs = tuple(probabilities[i] for i in visible_idx)

    if exactly_one:
        hidden_idx = {
            i for i, is_hidden in enumerate(bundle.none_is_max) if bool(is_hidden)
        }
        if any(any(i in hidden_idx for i in group) for group in selected_groups):
            raise ValueError(
                "exactly_one visible normalizer cannot drop none_is_max atoms"
            )
        reduced_groups = _project_groups_preserving_singletons(
            selected_groups, visible_idx
        )
    else:
        reduced_groups = project_groups(selected_groups, visible_idx)

    return selected_groups_product_normalizer(
        reduced_probs, reduced_groups, exactly_one=exactly_one
    )


def _prb_exact_one_normalizer(
    probabilities: tuple[float, ...],
    groups: tuple[tuple[int, ...], ...],
    feature_names: tuple[str, ...],
) -> tuple[float, dict[str, Any]]:
    """Exact Z for PRB stack constraints using a matching-style DP."""
    if not groups:
        return 1.0, {
            "method": "prb_matching_dp",
            "constraint_semantics": "exactly_one",
            "shapes": 0,
            "blocks": 0,
            "states": 1,
            "max_intermediate_scope": 0,
        }

    support_groups: dict[str, tuple[int, dict[str, int]]] = {}
    clear_groups: dict[str, tuple[int, dict[str, int]]] = {}
    used_groups = 0
    for group in groups:
        atoms = [feature_names[i] for i in group]
        parsed = [_parse_atom(atom) for atom in atoms]
        table_atoms = [
            i
            for i, (pred, args) in zip(group, parsed)
            if pred == "is-directly-on-table" and len(args) == 1
        ]
        clear_atoms = [
            i
            for i, (pred, args) in zip(group, parsed)
            if pred == "is-clear-on-top" and len(args) == 1
        ]
        top_atoms = [
            (i, args[0], args[1])
            for i, (pred, args) in zip(group, parsed)
            if pred == "shape-on-top-of-block" and len(args) == 2
        ]
        if len(table_atoms) == 1 and not clear_atoms and top_atoms:
            shapes = {shape for _i, shape, _block in top_atoms}
            if len(shapes) != 1:
                raise ValueError("PRB support group mixes shapes")
            shape = next(iter(shapes))
            support_groups[shape] = (
                table_atoms[0],
                {block: i for i, _shape, block in top_atoms},
            )
            used_groups += 1
        elif len(clear_atoms) == 1 and not table_atoms and top_atoms:
            blocks = {block for _i, _shape, block in top_atoms}
            if len(blocks) != 1:
                raise ValueError("PRB clear group mixes support blocks")
            block = next(iter(blocks))
            clear_groups[block] = (
                clear_atoms[0],
                {shape: i for i, shape, _block in top_atoms},
            )
            used_groups += 1
        else:
            raise ValueError("Unrecognized PRB exact-one constraint structure")
    if used_groups != len(groups) or not support_groups:
        raise ValueError("Incomplete PRB exact-one constraint decomposition")

    shapes = sorted(support_groups)
    blocks = sorted(clear_groups)
    block_pos = {block: i for i, block in enumerate(blocks)}
    clear_false_factor = {
        block: 1.0 - probabilities[clear_groups[block][0]] for block in blocks
    }
    clear_true_factor = {
        block: probabilities[clear_groups[block][0]] for block in blocks
    }

    dp: dict[int, float] = {0: 1.0}
    for shape in shapes:
        table_idx, top_by_block = support_groups[shape]
        false_top_product = 1.0
        for idx in top_by_block.values():
            false_top_product *= 1.0 - probabilities[idx]
        q_table = probabilities[table_idx]
        options: list[tuple[str | None, float]] = [
            (None, q_table * false_top_product)
        ]
        for block, idx in top_by_block.items():
            q_top = probabilities[idx]
            false_term = 1.0 - q_top
            if false_term == 0.0:
                support_weight = (1.0 - q_table) * q_top
                for other_block, other_idx in top_by_block.items():
                    if other_block != block:
                        support_weight *= 1.0 - probabilities[other_idx]
            else:
                support_weight = (
                    (1.0 - q_table) * false_top_product * q_top / false_term
                )
            options.append((block, support_weight))

        next_dp: dict[int, float] = {}
        for mask, prefix_weight in dp.items():
            for block, support_weight in options:
                if block is None or block not in block_pos:
                    next_mask = mask
                    occupancy_weight = 1.0
                else:
                    bit = 1 << block_pos[block]
                    if mask & bit:
                        continue
                    next_mask = mask | bit
                    occupancy_weight = clear_false_factor[block]
                next_dp[next_mask] = next_dp.get(next_mask, 0.0) + (
                    prefix_weight * support_weight * occupancy_weight
                )
        dp = next_dp

    z = 0.0
    full_mask = (1 << len(blocks)) - 1
    for mask, weight in dp.items():
        free_clear_weight = 1.0
        for block in blocks:
            bit = 1 << block_pos[block]
            if not (mask & bit):
                free_clear_weight *= clear_true_factor[block]
        z += weight * free_clear_weight
    return z, {
        "method": "prb_matching_dp",
        "constraint_semantics": "exactly_one",
        "shapes": len(shapes),
        "blocks": len(blocks),
        "states": len(dp),
        "max_intermediate_scope": len(blocks),
        "full_mask": full_mask,
    }


def _constraint_components_including_singletons(
    groups: Sequence[Sequence[int]],
) -> list[tuple[set[int], list[tuple[int, ...]]]]:
    var_to_groups: dict[int, list[int]] = {}
    normalized_groups = [tuple(sorted(dict.fromkeys(group))) for group in groups]
    for i, group in enumerate(normalized_groups):
        for var in group:
            var_to_groups.setdefault(var, []).append(i)

    visited: set[int] = set()
    components: list[tuple[set[int], list[tuple[int, ...]]]] = []
    for start in range(len(normalized_groups)):
        if start in visited:
            continue
        queue = [start]
        visited.add(start)
        comp_vars: set[int] = set()
        comp_groups: list[tuple[int, ...]] = []
        while queue:
            i = queue.pop()
            group = normalized_groups[i]
            comp_groups.append(group)
            for var in group:
                comp_vars.add(var)
                for j in var_to_groups.get(var, []):
                    if j not in visited:
                        visited.add(j)
                        queue.append(j)
        components.append((comp_vars, comp_groups))
    return components


def _exact_one_normalizer_including_singletons(
    probabilities: Sequence[float],
    groups: Sequence[Sequence[int]],
    *,
    max_intermediate_scope: int,
) -> tuple[float, dict[str, Any]]:
    normalized = [tuple(sorted(dict.fromkeys(group))) for group in groups]
    if any(not group for group in normalized):
        return 0.0, {
            "method": "variable_elimination",
            "constraint_semantics": "exactly_one",
            "components": [],
            "max_intermediate_scope": 0,
            "empty_exact_one_constraints": True,
        }
    if not normalized:
        return 1.0, {
            "method": "variable_elimination",
            "constraint_semantics": "exactly_one",
            "components": [],
            "max_intermediate_scope": 0,
        }

    z = 1.0
    components_info: list[dict[str, Any]] = []
    max_scope_seen = 0
    for comp_vars, comp_groups in _constraint_components_including_singletons(
        normalized
    ):
        comp_z, info = _component_exact_at_most_one_normalizer(
            probabilities,
            comp_groups,
            max_intermediate_scope=max_intermediate_scope,
            exactly_one=True,
        )
        z *= comp_z
        max_scope_seen = max(max_scope_seen, int(info["max_intermediate_scope"]))
        components_info.append(
            info | {"normalizer": comp_z, "variables": len(comp_vars)}
        )
    return z, {
        "method": "variable_elimination",
        "constraint_semantics": "exactly_one",
        "components": components_info,
        "max_intermediate_scope": max_scope_seen,
    }


def exact_constraints_runtime_normalizer(
    probabilities: Sequence[float],
    constraint_groups: Sequence[Sequence[int]],
    *,
    feature_names: Sequence[str] | None = None,
    domain: str | None = None,
    exactly_one: bool = False,
    max_intermediate_scope: int = 24,
) -> tuple[float, dict[str, Any]]:
    if domain == "prb" and exactly_one:
        if feature_names is None:
            raise ValueError("feature_names are required for PRB exactly_one normalizer")

        return _prb_exact_one_normalizer(
            tuple(float(q) for q in probabilities),
            tuple(tuple(group) for group in constraint_groups),
            tuple(feature_names),
        )

    if exactly_one:
        return _exact_one_normalizer_including_singletons(
            probabilities,
            groups=constraint_groups,
            max_intermediate_scope=max_intermediate_scope,
        )

    return exact_at_most_one_normalizer(
        probabilities,
        constraint_groups,
        max_intermediate_scope=max_intermediate_scope,
        exactly_one=exactly_one,
    )
