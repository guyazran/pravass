from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence, Any, Mapping

State = tuple[bool, ...]
CommittedTemplate = tuple[bool | None, ...]


@dataclass(frozen=True)
class StateSet:
    states: tuple[State, ...]
    unconstrained_mass: float
    constrained_mass: float
    total_states_considered: int
    status: str = "ok"
    # When set, the StateSet represents *all* completions of `committed_template`:
    # each entry is True/False (committed) or None (wildcard). `states` is then a
    # preview slice (often empty) — use the helpers below to query size, test
    # membership, or materialize completions.
    committed_template: CommittedTemplate | None = None


def state_set_size(state_set: StateSet) -> int:
    template = state_set.committed_template
    if template is None:
        return len(state_set.states)
    wildcard_count = sum(1 for v in template if v is None)
    return 1 << wildcard_count


def state_set_contains(state_set: StateSet, candidate: State) -> bool:
    template = state_set.committed_template
    if template is None:
        return candidate in set(state_set.states)
    if len(candidate) != len(template):
        return False
    return all(t is None or t == c for c, t in zip(candidate, template))


def iter_state_set(state_set: StateSet, limit: int | None = None) -> Iterator[State]:
    """Lazily yield states from a StateSet — at most `limit` if provided."""
    template = state_set.committed_template
    if template is None:
        for i, state in enumerate(state_set.states):
            if limit is not None and i >= limit:
                return
            yield state
        return
    wildcard_indices = [i for i, v in enumerate(template) if v is None]
    base = [bool(v) if v is not None else False for v in template]
    total = 1 << len(wildcard_indices)
    if limit is not None:
        total = min(total, limit)
    for combo in range(total):
        state = list(base)
        for j, idx in enumerate(wildcard_indices):
            state[idx] = bool((combo >> j) & 1)
        yield tuple(state)


def clamp_probability(q: float) -> float:
    if not math.isfinite(q) or q < 0.0 or q > 1.0:
        raise ValueError(f"Probability outside [0, 1]: {q}")
    return q


def unconstrained_probability(state: State, probabilities: Sequence[float]) -> float:
    if len(state) != len(probabilities):
        raise ValueError("State/probability length mismatch")
    p = 1.0
    for value, q_raw in zip(state, probabilities):
        q = clamp_probability(float(q_raw))
        p *= q if value else 1.0 - q
        if p == 0.0:
            break
    return p


def log_unconstrained_probability(state: State, probabilities: Sequence[float]) -> float:
    if len(state) != len(probabilities):
        raise ValueError("State/probability length mismatch")
    total = 0.0
    for value, q_raw in zip(state, probabilities):
        q = clamp_probability(float(q_raw))
        term = q if value else 1.0 - q
        if term == 0.0:
            return -math.inf
        total += math.log(term)
    return total


def flip_log_delta(current_value: bool, q_raw: float) -> float:
    """Return a conservative -inf delta for impossible current assignments."""
    q = clamp_probability(float(q_raw))
    old_term = q if current_value else 1.0 - q
    new_term = 1.0 - q if current_value else q
    if old_term == 0.0:
        return -math.inf
    if new_term == 0.0:
        return -math.inf
    return math.log(new_term) - math.log(old_term)


def is_valid_state(
    state: State,
    selected_groups: Sequence[Sequence[int]],
    *,
    exactly_one: bool = False,
) -> bool:
    for group in selected_groups:
        true_count = 0
        for idx in group:
            if state[idx]:
                true_count += 1
                if true_count > 1:
                    return False
        if exactly_one and true_count == 0:
            return False
    return True


def at_most_one_group_probability(
    group: Sequence[int], probabilities: Sequence[float]
) -> float:
    false_product = 1.0
    for idx in group:
        false_product *= 1.0 - clamp_probability(float(probabilities[idx]))
    total = false_product
    for idx in group:
        q = clamp_probability(float(probabilities[idx]))
        false_term = 1.0 - q
        if false_term == 0.0:
            term = q
            for other in group:
                if other != idx:
                    term *= 1.0 - clamp_probability(float(probabilities[other]))
        else:
            term = false_product * q / false_term
        total += term
    return total


def selected_groups_normalizer(
    probabilities: Sequence[float], selected_groups: Sequence[Sequence[int]]
) -> float:
    z = 1.0
    for group in selected_groups:
        if len(group) >= 2:
            z *= at_most_one_group_probability(group, probabilities)
    return z


def conditional_at_most_one_normalizer(
    probabilities: Sequence[float],
    selected_groups: Sequence[Sequence[int]],
    commits: Mapping[int, bool],
) -> float:
    """Probability mass of valid completions consistent with ``commits``.

    Returns Σ over states s of unconstrained_probability(s, probabilities), where
    s ranges over assignments that match ``commits`` on committed indices and
    satisfy the at-most-one ``selected_groups``. Assumes pairwise-disjoint
    groups, like ``selected_groups_normalizer``. Uncommitted lonely features
    integrate out to a factor of 1; committed lonely features contribute
    q_i (true) or 1-q_i (false). Conflicts (≥2 committed-true in the same
    group) yield 0.
    """
    grouped: set[int] = set()
    z = 1.0
    for group in selected_groups:
        if len(group) < 2:
            continue
        grouped.update(group)
        committed_true: list[int] = []
        committed_false: list[int] = []
        uncommitted: list[int] = []
        for idx in group:
            if idx in commits:
                if commits[idx]:
                    committed_true.append(idx)
                else:
                    committed_false.append(idx)
            else:
                uncommitted.append(idx)
        if len(committed_true) > 1:
            return 0.0
        false_factor = 1.0
        for idx in committed_false:
            false_factor *= 1.0 - clamp_probability(float(probabilities[idx]))
        if committed_true:
            i_star = committed_true[0]
            true_factor = clamp_probability(float(probabilities[i_star]))
            for idx in uncommitted:
                true_factor *= 1.0 - clamp_probability(float(probabilities[idx]))
            z *= true_factor * false_factor
        else:
            z *= false_factor * at_most_one_group_probability(uncommitted, probabilities)
        if z == 0.0:
            return 0.0
    for idx, value in commits.items():
        if idx in grouped:
            continue
        q = clamp_probability(float(probabilities[idx]))
        z *= q if value else 1.0 - q
        if z == 0.0:
            return 0.0
    return z


@dataclass(frozen=True)
class _Factor:
    scope: tuple[int, ...]
    table: tuple[float, ...]


def _make_unary_factor(var: int, q: float) -> _Factor:
    return _Factor((var,), (1.0 - q, q))


def _make_at_most_one_factor(group: Sequence[int], exactly_one: bool = False) -> _Factor:
    scope = tuple(sorted(dict.fromkeys(group)))
    if exactly_one:
        table = tuple(
            1.0 if assignment.bit_count() == 1 else 0.0
            for assignment in range(1 << len(scope))
        )
    else:
        table = tuple(
            1.0 if assignment.bit_count() <= 1 else 0.0
            for assignment in range(1 << len(scope))
        )
    return _Factor(scope, table)


def _project_assignment(
    assignment: int, union_scope: Sequence[int], factor_scope: Sequence[int]
) -> int:
    union_pos = {var: i for i, var in enumerate(union_scope)}
    index = 0
    for fpos, var in enumerate(factor_scope):
        if assignment & (1 << union_pos[var]):
            index |= 1 << fpos
    return index


def _drop_assignment_bit(assignment: int, drop_pos: int, union_size: int) -> int:
    out = 0
    out_pos = 0
    for pos in range(union_size):
        if pos == drop_pos:
            continue
        if assignment & (1 << pos):
            out |= 1 << out_pos
        out_pos += 1
    return out


def _combine_and_sum_out(factors: Sequence[_Factor], var: int) -> tuple[_Factor, int]:
    union_scope = tuple(sorted(set().union(*(f.scope for f in factors))))
    if var not in union_scope:
        raise ValueError(f"Variable {var!r} not present in collected factors")
    union_size = len(union_scope)
    drop_pos = union_scope.index(var)
    new_scope = tuple(v for v in union_scope if v != var)
    new_table = [0.0] * (1 << len(new_scope))
    projections = [
        [
            _project_assignment(assignment, union_scope, factor.scope)
            for assignment in range(1 << union_size)
        ]
        for factor in factors
    ]
    dropped = [
        _drop_assignment_bit(assignment, drop_pos, union_size)
        for assignment in range(1 << union_size)
    ]
    for assignment in range(1 << union_size):
        value = 1.0
        for factor, projection in zip(factors, projections):
            value *= factor.table[projection[assignment]]
            if value == 0.0:
                break
        if value:
            new_table[dropped[assignment]] += value
    return _Factor(new_scope, tuple(new_table)), union_size


def _constraint_components(
    groups: Sequence[Sequence[int]],
) -> list[tuple[set[int], list[tuple[int, ...]]]]:
    var_to_groups: dict[int, list[int]] = {}
    normalized_groups = [tuple(sorted(dict.fromkeys(group))) for group in groups if len(group) >= 2]
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


def _component_exact_at_most_one_normalizer(
    probabilities: Sequence[float],
    groups: Sequence[Sequence[int]],
    *,
    max_intermediate_scope: int,
    exactly_one: bool = False,
) -> tuple[float, dict[str, Any]]:
    variables = sorted(set().union(*(set(group) for group in groups)))
    factors: list[_Factor] = [
        _make_unary_factor(var, clamp_probability(float(probabilities[var])))
        for var in variables
    ]
    factors.extend(_make_at_most_one_factor(group, exactly_one=exactly_one) for group in groups)
    remaining = set(variables)
    max_scope_seen = 0
    eliminated = 0
    while remaining:
        best_var = None
        best_scope_size = None
        for var in sorted(remaining):
            involved = [factor for factor in factors if var in factor.scope]
            scope_size = len(set().union(*(factor.scope for factor in involved)))
            if best_scope_size is None or scope_size < best_scope_size:
                best_var = var
                best_scope_size = scope_size
        assert best_var is not None and best_scope_size is not None
        if best_scope_size > max_intermediate_scope:
            raise RuntimeError(
                f"VE intermediate scope would have {best_scope_size} variables, "
                f"exceeding limit {max_intermediate_scope}"
            )
        involved = [factor for factor in factors if best_var in factor.scope]
        not_involved = [factor for factor in factors if best_var not in factor.scope]
        new_factor, scope_seen = _combine_and_sum_out(involved, best_var)
        max_scope_seen = max(max_scope_seen, scope_seen)
        factors = not_involved
        if new_factor.scope or new_factor.table[0] != 1.0:
            factors.append(new_factor)
        remaining.remove(best_var)
        eliminated += 1

    z = 1.0
    for factor in factors:
        if factor.scope:
            raise AssertionError(f"Unexpected non-constant factor: {factor.scope}")
        z *= factor.table[0]
    return z, {
        "variables": len(variables),
        "constraints": len(groups),
        "eliminated_variables": eliminated,
        "max_intermediate_scope": max_scope_seen,
    }


def exact_at_most_one_normalizer(
    probabilities: Sequence[float],
    groups: Sequence[Sequence[int]],
    *,
    max_intermediate_scope: int = 24,
    exactly_one: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Compute exact probability of satisfying possibly-overlapping groups.

    By default each group is interpreted as an at-most-one mutex. With
    ``exactly_one=True``, every group must contain exactly one true atom. This is
    weighted model counting by variable elimination; it is exact but can be
    exponentially expensive in the induced width, so ``max_intermediate_scope``
    prevents accidental blowups.
    """
    normalized = [tuple(sorted(dict.fromkeys(group))) for group in groups if len(group) >= 2]
    if not normalized:
        return 1.0, {
            "method": "variable_elimination",
            "constraint_semantics": "exactly_one" if exactly_one else "at_most_one",
            "components": [],
            "max_intermediate_scope": 0,
        }
    z = 1.0
    components_info: list[dict[str, Any]] = []
    max_scope_seen = 0
    for comp_vars, comp_groups in _constraint_components(normalized):
        comp_z, info = _component_exact_at_most_one_normalizer(
            probabilities,
            comp_groups,
            max_intermediate_scope=max_intermediate_scope,
            exactly_one=exactly_one,
        )
        z *= comp_z
        max_scope_seen = max(max_scope_seen, int(info["max_intermediate_scope"]))
        components_info.append(info | {"normalizer": comp_z, "variables": len(comp_vars)})
    return z, {
        "method": "variable_elimination",
        "constraint_semantics": "exactly_one" if exactly_one else "at_most_one",
        "components": components_info,
        "max_intermediate_scope": max_scope_seen,
    }


def all_states(feature_count: int) -> Iterable[State]:
    for values in itertools.product((False, True), repeat=feature_count):
        yield tuple(values)


def visible_indices(none_is_max: Sequence[bool]) -> tuple[int, ...]:
    return tuple(i for i, flag in enumerate(none_is_max) if not flag)


def null_indices(none_is_max: Sequence[bool]) -> tuple[int, ...]:
    return tuple(i for i, flag in enumerate(none_is_max) if flag)


def project_state(state: Sequence[bool], indices: Sequence[int]) -> State:
    return tuple(bool(state[i]) for i in indices)


def project_feature_names(
    feature_names: Sequence[str], indices: Sequence[int]
) -> tuple[str, ...]:
    return tuple(feature_names[i] for i in indices)


def logical_full_state_count(compact_count: int, omitted_feature_count: int) -> int:
    if compact_count < 0:
        raise ValueError(f"compact_count must be non-negative, got {compact_count}")
    if omitted_feature_count < 0:
        raise ValueError(
            f"omitted_feature_count must be non-negative, got {omitted_feature_count}"
        )
    return compact_count * (1 << omitted_feature_count)


@dataclass(frozen=True)
class _DisjointExtensionGroup:
    reduced_positions: tuple[int, ...]
    omitted_count: int


def _normalized_constraint_groups(
    selected_groups: Sequence[Sequence[int]], full_feature_count: int
) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    for group in selected_groups:
        normalized = tuple(sorted(dict.fromkeys(int(idx) for idx in group)))
        if len(normalized) < 2:
            continue
        for idx in normalized:
            if idx < 0 or idx >= full_feature_count:
                raise ValueError(
                    f"constraint index {idx} outside feature range [0, {full_feature_count})"
                )
        groups.append(normalized)
    return tuple(groups)


def _groups_are_pairwise_disjoint(groups: Sequence[Sequence[int]]) -> bool:
    seen: set[int] = set()
    for group in groups:
        for idx in group:
            if idx in seen:
                return False
            seen.add(idx)
    return True


def _count_partial_assignments_satisfying_groups(
    fixed_values: Mapping[int, bool],
    *,
    full_feature_count: int,
    groups: Sequence[Sequence[int]],
) -> int:
    """Exact fallback for overlapping at-most-one groups.

    Current experiment-1 selected groups are pairwise disjoint, so the fast path
    below is used in practice. This fallback keeps the public helper exact for
    small overlapping components instead of silently over-counting.
    """
    grouped_vars = set().union(*(set(group) for group in groups)) if groups else set()
    free_outside_group_count = sum(
        1
        for idx in range(full_feature_count)
        if idx not in fixed_values and idx not in grouped_vars
    )
    total = 1 << free_outside_group_count
    for component_vars, component_groups in _constraint_components(groups):
        free_vars = tuple(sorted(v for v in component_vars if v not in fixed_values))
        if len(free_vars) > 24:
            raise RuntimeError(
                "overlapping constraint component has too many free variables "
                f"to count exactly ({len(free_vars)} > 24)"
            )
        free_pos = {var: pos for pos, var in enumerate(free_vars)}
        component_count = 0
        for assignment in range(1 << len(free_vars)):
            valid = True
            for group in component_groups:
                true_count = 0
                for var in group:
                    if var in fixed_values:
                        value = fixed_values[var]
                    else:
                        value = bool(assignment & (1 << free_pos[var]))
                    if value:
                        true_count += 1
                        if true_count > 1:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                component_count += 1
        total *= component_count
        if total == 0:
            break
    return total


def constraint_satisfying_full_state_count(
    state_set: StateSet,
    *,
    full_feature_count: int,
    selected_groups: Sequence[Sequence[int]],
    search_indices: Sequence[int],
) -> int:
    """Count full feature-space assignments represented by a reduced StateSet.

    ``state_set`` is defined over ``search_indices`` (the non-null/search
    features). The remaining full-feature indices are treated as unassigned null
    fluents. This returns the number of full assignments whose projection is in
    the reduced set and that satisfy the full at-most-one ``selected_groups``.
    """
    search_indices = tuple(int(idx) for idx in search_indices)
    if len(search_indices) != len(set(search_indices)):
        raise ValueError("search_indices must be unique")
    for idx in search_indices:
        if idx < 0 or idx >= full_feature_count:
            raise ValueError(
                f"search index {idx} outside feature range [0, {full_feature_count})"
            )
    groups = _normalized_constraint_groups(selected_groups, full_feature_count)

    if not _groups_are_pairwise_disjoint(groups):
        if state_set.committed_template is not None:
            if len(state_set.committed_template) != len(search_indices):
                raise ValueError("committed template/search index length mismatch")
            fixed_values = {
                full_idx: bool(value)
                for full_idx, value in zip(search_indices, state_set.committed_template)
                if value is not None
            }
            return _count_partial_assignments_satisfying_groups(
                fixed_values,
                full_feature_count=full_feature_count,
                groups=groups,
            )
        total = 0
        for state in state_set.states:
            if len(state) != len(search_indices):
                raise ValueError("state/search index length mismatch")
            fixed_values = {
                full_idx: bool(value)
                for full_idx, value in zip(search_indices, state)
            }
            total += _count_partial_assignments_satisfying_groups(
                fixed_values,
                full_feature_count=full_feature_count,
                groups=groups,
            )
        return total

    search_pos = {full_idx: pos for pos, full_idx in enumerate(search_indices)}
    search_set = set(search_indices)
    grouped_full_indices = (
        set().union(*(set(group) for group in groups)) if groups else set()
    )
    omitted_grouped_indices = grouped_full_indices - search_set
    omitted_count = full_feature_count - len(search_indices)
    omitted_ungrouped_count = omitted_count - len(omitted_grouped_indices)
    group_infos = tuple(
        _DisjointExtensionGroup(
            reduced_positions=tuple(
                search_pos[idx] for idx in group if idx in search_pos
            ),
            omitted_count=sum(1 for idx in group if idx not in search_set),
        )
        for group in groups
    )

    template = state_set.committed_template
    if template is not None:
        if len(template) != len(search_indices):
            raise ValueError("committed template/search index length mismatch")
        total = 1 << omitted_ungrouped_count
        grouped_search_indices = grouped_full_indices & search_set
        total *= 1 << sum(
            1
            for full_idx, value in zip(search_indices, template)
            if full_idx not in grouped_search_indices and value is None
        )
        for info in group_infos:
            committed_true = 0
            wildcard_count = info.omitted_count
            for pos in info.reduced_positions:
                value = template[pos]
                if value is None:
                    wildcard_count += 1
                elif value:
                    committed_true += 1
                    if committed_true > 1:
                        return 0
            if committed_true == 0:
                total *= 1 + wildcard_count
        return total

    total = 0
    base = 1 << omitted_ungrouped_count
    for state in state_set.states:
        if len(state) != len(search_indices):
            raise ValueError("state/search index length mismatch")
        count = base
        for info in group_infos:
            true_count = 0
            for pos in info.reduced_positions:
                if state[pos]:
                    true_count += 1
                    if true_count > 1:
                        count = 0
                        break
            if count == 0:
                break
            if true_count == 0:
                count *= 1 + info.omitted_count
        total += count
    return total


def project_groups(
    selected_groups: Sequence[Sequence[int]], visible_idx: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    remap = {orig: new for new, orig in enumerate(visible_idx)}
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    for group in selected_groups:
        new_group = tuple(sorted(remap[i] for i in group if i in remap))
        if len(new_group) >= 2 and new_group not in seen:
            seen.add(new_group)
            out.append(new_group)
    return tuple(out)


def truth_state_from_ground_truth(ground_truth: Sequence[bool]) -> State:
    return tuple(bool(v) for v in ground_truth)


def state_to_true_fluents(state: State, feature_names: Sequence[str]) -> list[str]:
    if len(state) != len(feature_names):
        raise ValueError("State/feature length mismatch")
    return [name for name, value in zip(feature_names, state) if value]


def state_to_bitstring(state: State) -> str:
    return "".join("1" if value else "0" for value in state)


def preview_state_set(
    state_set: StateSet, feature_names: Sequence[str], limit: int
) -> list[dict[str, object]]:
    preview: list[dict[str, object]] = []
    for state in iter_state_set(state_set, limit=limit):
        preview.append(
            {
                "bitstring": state_to_bitstring(state),
                "true_fluents": state_to_true_fluents(state, feature_names),
            }
        )
    return preview
