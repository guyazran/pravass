"""Pure utilities for Experiment 6 controlled-prior construction.

The functions in this module deliberately separate structural preparation from
label orientation.  ``extract_candidate_units`` reads only task structure and
the symbolic visibility observation; ground-truth values enter later through
``orient_treatment``.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pravass import (
    FunctionConstraint,
    InfeasibleConstraintError,
    TaskBelief,
    UnconstrainedBelief,
    UnitNormalizer,
    iter_pravass,
)

from .closed_loop import PROB_EPS, OrderedBelief, planning_result_found
from .planning import PlanningResult
from .search import feature_names as indexed_feature_names
from .states import State, is_valid_state, unconstrained_probability

PROTOCOL_VERSION = "exp6-controlled-prior-v1"
PRIMARY_RELEVANCE_VERSION = "strict-goal-hidden-v2"
REVISED_RELEVANCE_VERSION = "typed-action-support-v2"
QUERY_POLICY = "seeded_uniform_relevant"

ATOM_RE = re.compile(r"^([^()]+)\((.*)\)$")
FACT_RE = re.compile(r"\(([a-z][a-z0-9_-]*)\s*([^()]*)\)")
INSTANCE_RE = re.compile(r"\b([a-z][a-z0-9_-]*_\d+)\b")


def parse_atom(atom: str) -> tuple[str, tuple[str, ...]]:
    match = ATOM_RE.fullmatch(atom)
    if match is None:
        raise ValueError(f"invalid atom: {atom!r}")
    args = tuple(part.strip() for part in match.group(2).split(",") if part.strip())
    return match.group(1), args


def stable_rank(*parts: object) -> str:
    payload = "\x1f".join(map(str, parts))
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class CandidateUnit:
    """A binary fluent or one exactly-one categorical fluent group."""

    unit_id: str
    kind: str
    atoms: tuple[str, ...]
    visibility_rule: str
    relevance_rule: str

    def __post_init__(self) -> None:
        if self.kind not in {"binary", "categorical"}:
            raise ValueError(f"unsupported unit kind: {self.kind}")
        if self.kind == "binary" and len(self.atoms) != 1:
            raise ValueError("binary units require one atom")
        if self.kind == "categorical" and len(self.atoms) < 2:
            raise ValueError("categorical units require at least two atoms")
        if len(set(self.atoms)) != len(self.atoms):
            raise ValueError("candidate-unit atoms must be unique")

    @property
    def arity(self) -> int:
        return 2 if self.kind == "binary" else len(self.atoms)

    def as_json(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "kind": self.kind,
            "atoms": list(self.atoms),
            "arity": self.arity,
            "visibility_rule": self.visibility_rule,
            "relevance_rule": self.relevance_rule,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> CandidateUnit:
        return cls(
            unit_id=str(payload["unit_id"]),
            kind=str(payload["kind"]),
            atoms=tuple(map(str, payload["atoms"])),
            visibility_rule=str(payload["visibility_rule"]),
            relevance_rule=str(payload["relevance_rule"]),
        )


@dataclass(frozen=True)
class FrozenTreatment:
    corruption_count: int
    corrupted_unit_ids: tuple[str, ...]
    queried_unit_id: str
    corruption_mask_fingerprint: str

    def as_json(self) -> dict[str, Any]:
        return {
            "corruption_count": self.corruption_count,
            "corrupted_unit_ids": list(self.corrupted_unit_ids),
            "queried_unit_id": self.queried_unit_id,
            "corruption_mask_fingerprint": self.corruption_mask_fingerprint,
        }


@dataclass(frozen=True)
class OrientedTreatment:
    ground_truth_assignments: tuple[tuple[str, bool], ...]
    decoy_assignments: tuple[tuple[str, bool], ...]
    sensor_state_by_unit: tuple[tuple[str, str], ...]
    probabilities: tuple[float, ...]
    query_hit_corruption: bool
    remaining_corrupted_units: int


def _goal_section(problem_text: str) -> str:
    marker = problem_text.rfind("(:goal")
    return problem_text[marker:] if marker >= 0 else ""


def _facts(problem_text: str, predicate: str) -> tuple[tuple[str, ...], ...]:
    found: list[tuple[str, ...]] = []
    for name, raw_args in FACT_RE.findall(problem_text):
        if name != predicate or "?" in raw_args:
            continue
        args = tuple(raw_args.split())
        if args:
            found.append(args)
    return tuple(found)


def _pack_units(
    units: Iterable[CandidateUnit],
    *,
    sample_id: str,
    sample_seed: int,
    max_units: int,
    max_states: int,
) -> tuple[CandidateUnit, ...]:
    """Frozen, label-blind rank-and-pack under the explicit-state capacity."""
    ranked = sorted(
        units,
        key=lambda unit: (
            stable_rank(
                PROTOCOL_VERSION,
                "candidate",
                sample_seed,
                sample_id,
                unit.unit_id,
            ),
            unit.unit_id,
        ),
    )
    selected: list[CandidateUnit] = []
    state_count = 1
    occupied_atoms: set[str] = set()
    for unit in ranked:
        if len(selected) >= max_units:
            break
        if occupied_atoms.intersection(unit.atoms):
            continue
        if state_count * unit.arity > max_states:
            continue
        selected.append(unit)
        occupied_atoms.update(unit.atoms)
        state_count *= unit.arity
    return tuple(selected)


def extract_candidate_units(
    problem_text: str,
    *,
    sample_id: str,
    sample_seed: int,
    relevance_version: str,
    max_units: int = 6,
    max_states: int = 64,
) -> tuple[CandidateUnit, ...]:
    """Extract hidden goal/action-relevant units without reading hidden labels.

    Visibility is operationalized from the symbolic initial observation:

    * object fluents are visible only when the object has ``isvisibleobject``;
    * receptacle fluents are visible only when it has ``isvisiblereceptacle``;
    * location of a non-visible goal object is a hidden categorical unit.

    The strict definition uses only hidden goal-object locations and hidden
    goal predicates/receptacles.  The documented Phase-A revision additionally
    includes hidden openable receptacles that are type-compatible with a goal
    object and therefore occur in grounded pickup/put action support.
    """
    if relevance_version not in {
        PRIMARY_RELEVANCE_VERSION,
        REVISED_RELEVANCE_VERSION,
    }:
        raise ValueError(f"unknown relevance version: {relevance_version}")

    goal_text = _goal_section(problem_text)
    object_type = {args[0]: args[1] for args in _facts(problem_text, "objecttype")}
    receptacle_type = {
        args[0]: args[1] for args in _facts(problem_text, "receptacletype")
    }
    can_contain = set(_facts(problem_text, "cancontain"))
    visible_objects = {args[0] for args in _facts(problem_text, "isvisibleobject")}
    # The gripper state is directly known to the agent even when the held
    # object is outside the camera observation. It is therefore part of the
    # operational visibility rule, not a held-out perception label.
    held_objects = {args[0] for args in _facts(problem_text, "holds")}
    visible_receptacles = {
        args[0] for args in _facts(problem_text, "isvisiblereceptacle")
    }
    openable = {args[0] for args in _facts(problem_text, "openable")}
    declared_predicates = {
        match.group(1)
        for match in re.finditer(r"\(([a-z][a-z0-9_-]*)\s+[?a-z]", problem_text)
    }
    goal_objects = tuple(
        obj
        for obj in dict.fromkeys(INSTANCE_RE.findall(goal_text))
        if obj in object_type
    )
    goal_receptacles = {
        arg
        for _predicate, raw_args in FACT_RE.findall(goal_text)
        for arg in raw_args.split()
        if arg in receptacle_type
    }

    candidates: dict[str, CandidateUnit] = {}
    for obj in goal_objects:
        compatible = tuple(
            sorted(
                receptacle
                for receptacle, rtype in receptacle_type.items()
                if (rtype, object_type[obj]) in can_contain
            )
        )
        object_hidden = obj not in visible_objects and obj not in held_objects
        if object_hidden and len(compatible) >= 2:
            unit = CandidateUnit(
                f"location({obj})",
                "categorical",
                tuple(f"inreceptacle({obj},{rec})" for rec in compatible),
                "object absent from isvisibleobject in initial symbolic observation",
                "goal object location supports pickup and goal achievement",
            )
            candidates[unit.unit_id] = unit

        relevant_receptacles = set(goal_receptacles)
        if object_hidden:
            relevant_receptacles.update(compatible)
        if relevance_version == REVISED_RELEVANCE_VERSION:
            relevant_receptacles.update(compatible)
        for receptacle in sorted(relevant_receptacles & openable):
            if receptacle in visible_receptacles:
                continue
            unit = CandidateUnit(
                f"opened({receptacle})",
                "binary",
                (f"opened({receptacle})",),
                "receptacle absent from isvisiblereceptacle in initial symbolic observation",
                (
                    "hidden openable receptacle is a grounded action support for "
                    "a goal-object pickup/put"
                ),
            )
            candidates[unit.unit_id] = unit

        if object_hidden:
            for predicate in ("issliced", "isclean", "ishot", "iscool", "istoggled"):
                if predicate not in declared_predicates or f"({predicate} {obj}" not in goal_text:
                    continue
                unit = CandidateUnit(
                    f"{predicate}({obj})",
                    "binary",
                    (f"{predicate}({obj})",),
                    "goal object absent from isvisibleobject in initial symbolic observation",
                    "fluent appears directly in the goal",
                )
                candidates[unit.unit_id] = unit

    return _pack_units(
        candidates.values(),
        sample_id=sample_id,
        sample_seed=sample_seed,
        max_units=max_units,
        max_states=max_states,
    )


def controlled_universe_size(units: Sequence[CandidateUnit]) -> int:
    return math.prod(unit.arity for unit in units)


def freeze_treatment(
    units: Sequence[CandidateUnit],
    *,
    sample_id: str,
    corruption_count: int,
    corruption_seed: int,
    query_seed: int,
) -> FrozenTreatment:
    if not 0 <= corruption_count <= len(units):
        raise ValueError("corruption count exceeds available units")
    ranked = sorted(
        units,
        key=lambda unit: (
            stable_rank(
                PROTOCOL_VERSION,
                "corruption",
                sample_id,
                corruption_count,
                corruption_seed,
                unit.unit_id,
            ),
            unit.unit_id,
        ),
    )
    corrupted = tuple(unit.unit_id for unit in ranked[:corruption_count])
    # Deliberately contains neither corruption_count nor corruption_seed.
    queried = min(
        units,
        key=lambda unit: (
            stable_rank(
                PROTOCOL_VERSION,
                "query",
                sample_id,
                query_seed,
                unit.unit_id,
            ),
            unit.unit_id,
        ),
    ).unit_id
    fingerprint = hashlib.sha256(
        "\n".join(corrupted).encode()
    ).hexdigest()[:16]
    return FrozenTreatment(
        corruption_count,
        corrupted,
        queried,
        fingerprint,
    )


def _unit_truth(
    unit: CandidateUnit,
    ground_truth: Mapping[str, bool],
) -> tuple[bool, ...]:
    values = tuple(bool(ground_truth.get(atom, False)) for atom in unit.atoms)
    if unit.kind == "categorical" and sum(values) != 1:
        raise ValueError(
            f"{unit.unit_id} requires exactly one true ground-truth member; got {values}"
        )
    return values


def orient_treatment(
    feature_names: Sequence[str],
    units: Sequence[CandidateUnit],
    treatment: FrozenTreatment,
    ground_truth: Mapping[str, bool],
    *,
    sensor_confidence: float,
    sample_id: str,
    corruption_seed: int,
) -> OrientedTreatment:
    """Use labels only after masks/query are frozen to orient sensor states."""
    if not 0.5 < sensor_confidence < 1.0:
        raise ValueError("sensor confidence must be in (0.5, 1)")
    index = {name: i for i, name in enumerate(feature_names)}
    probabilities = [PROB_EPS] * len(feature_names)
    ground_truth_assignments: dict[str, bool] = {}
    decoy_assignments: dict[str, bool] = {}
    sensor_states: list[tuple[str, str]] = []
    corrupted = set(treatment.corrupted_unit_ids)

    for unit in units:
        truth = _unit_truth(unit, ground_truth)
        ground_truth_assignments.update(zip(unit.atoms, truth))
        if unit.kind == "binary":
            decoy = (not truth[0],)
            selected = decoy if unit.unit_id in corrupted else truth
            probabilities[index[unit.atoms[0]]] = (
                sensor_confidence if selected[0] else 1.0 - sensor_confidence
            )
            sensor_states.append((unit.unit_id, str(bool(selected[0])).lower()))
        else:
            true_index = truth.index(True)
            alternatives = [i for i in range(len(unit.atoms)) if i != true_index]
            decoy_index = min(
                alternatives,
                key=lambda i: (
                    stable_rank(
                        PROTOCOL_VERSION,
                        "decoy",
                        sample_id,
                        corruption_seed,
                        unit.unit_id,
                        unit.atoms[i],
                    ),
                    unit.atoms[i],
                ),
            )
            decoy = tuple(i == decoy_index for i in range(len(unit.atoms)))
            selected_index = decoy_index if unit.unit_id in corrupted else true_index
            remainder = (1.0 - sensor_confidence) / (len(unit.atoms) - 1)
            for i, atom in enumerate(unit.atoms):
                probabilities[index[atom]] = (
                    sensor_confidence if i == selected_index else remainder
                )
            sensor_states.append((unit.unit_id, unit.atoms[selected_index]))
        decoy_assignments.update(zip(unit.atoms, decoy))

    return OrientedTreatment(
        tuple(sorted(ground_truth_assignments.items())),
        tuple(sorted(decoy_assignments.items())),
        tuple(sensor_states),
        tuple(probabilities),
        treatment.queried_unit_id in corrupted,
        treatment.corruption_count
        - int(treatment.queried_unit_id in corrupted),
    )


def controlled_probabilities(
    feature_names: Sequence[str],
    base_state: State,
    units: Sequence[CandidateUnit],
    oriented: OrientedTreatment,
    *,
    eps: float = PROB_EPS,
) -> tuple[float, ...]:
    if len(feature_names) != len(base_state):
        raise ValueError("feature/base-state length mismatch")
    controlled_atoms = {atom for unit in units for atom in unit.atoms}
    probabilities = [
        (1.0 - eps if value else eps)
        for value in base_state
    ]
    for i, atom in enumerate(feature_names):
        if atom in controlled_atoms:
            probabilities[i] = oriented.probabilities[i]
    return tuple(probabilities)


def uniform_hidden_probabilities(
    feature_names: Sequence[str],
    base_state: State,
    units: Sequence[CandidateUnit],
    *,
    eps: float = PROB_EPS,
) -> tuple[float, ...]:
    """Assign p=0.5 to every declared hidden-unit atom.

    Coordinates outside the explicitly declared hidden, goal/action-relevant
    vocabulary remain clamped to the common natural MAP. Categorical units are
    subsequently conditioned on their shared exactly-one constraint, yielding
    a uniform distribution over their legal alternatives.
    """
    if len(feature_names) != len(base_state):
        raise ValueError("feature/base-state length mismatch")
    controlled_atoms = {atom for unit in units for atom in unit.atoms}
    return tuple(
        0.5 if atom in controlled_atoms else (1.0 - eps if value else eps)
        for atom, value in zip(feature_names, base_state)
    )


def update_group_observation(
    probabilities: Sequence[float],
    feature_names: Sequence[str],
    units: Sequence[CandidateUnit],
    queried_unit_id: str,
    ground_truth: Mapping[str, bool],
    *,
    eps: float = PROB_EPS,
) -> tuple[float, ...]:
    updated = list(map(float, probabilities))
    index = {name: i for i, name in enumerate(feature_names)}
    unit = next(unit for unit in units if unit.unit_id == queried_unit_id)
    truth = _unit_truth(unit, ground_truth)
    if unit.kind == "binary":
        updated[index[unit.atoms[0]]] = 1.0 - eps if truth[0] else eps
    else:
        true_index = truth.index(True)
        remainder = eps / (len(unit.atoms) - 1)
        for i, atom in enumerate(unit.atoms):
            updated[index[atom]] = 1.0 - eps if i == true_index else remainder
    return tuple(updated)


def enumerate_controlled_ordering(
    probabilities: Sequence[float],
    feature_names: Sequence[str],
    base_state: State,
    units: Sequence[CandidateUnit],
    constraint_groups: Sequence[Sequence[int]],
    *,
    max_states: int,
) -> OrderedBelief:
    """Rank the complete valid controlled state universe with PRaVaSS.

    PRaVaSS (the pravass package) searches over the units' atoms, with every
    other feature fixed to ``base_state``; a categorical unit has exactly one
    true atom. It yields every valid state, most probable first, so the exact
    normalizer is the sum of their masses.
    """
    start = time.perf_counter()
    index = {name: i for i, name in enumerate(feature_names)}
    atoms = tuple(dict.fromkeys(index[atom] for unit in units for atom in unit.atoms))
    names = indexed_feature_names(len(atoms))
    name_of = dict(zip(atoms, names))
    categorical = [
        frozenset(name_of[index[atom]] for atom in unit.atoms)
        for unit in units
        if unit.kind != "binary"
    ]

    def complete(assignment: frozenset[str]) -> State:
        state = list(base_state)
        for atom, name in name_of.items():
            state[atom] = name in assignment
        return tuple(state)

    def valid(assignment: frozenset[str]) -> bool:
        if any(len(group & assignment) != 1 for group in categorical):
            return False
        return is_valid_state(complete(assignment), constraint_groups, exactly_one=False)

    belief = TaskBelief(
        UnconstrainedBelief({name: float(probabilities[atom]) for atom, name in name_of.items()}),
        FunctionConstraint(valid),
        normalizer=UnitNormalizer(),
    )
    search = iter_pravass(belief, max_states=None)
    scored: list[tuple[float, State]] = []
    try:
        for assignment in search:
            candidate = complete(assignment)
            scored.append((unconstrained_probability(candidate, probabilities), candidate))
            if len(scored) > max_states:
                raise ValueError(
                    f"controlled universe has more than {max_states} states, cap is {max_states}"
                )
    except InfeasibleConstraintError:
        raise ValueError("controlled universe contains no valid state") from None
    normalizer = sum(mass for mass, _state in scored)
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("non-positive exact controlled normalizer")
    cumulative: list[float] = []
    running = 0.0
    for mass, _state in scored:
        running += mass / normalizer
        cumulative.append(min(1.0, running))
    cumulative[-1] = 1.0
    return OrderedBelief(
        states=tuple(state for _mass, state in scored),
        state_unconstrained_masses=tuple(mass for mass, _state in scored),
        cumulative_constrained_masses=tuple(cumulative),
        normalizer=normalizer,
        status="ok",
        target_mass=1.0,
        runtime_seconds=time.perf_counter() - start,
        states_popped=search.examined,
        states_scored=search.explored,
        invalid_states_skipped=search.examined - len(scored),
    )


def abstention_completions(
    ordering: OrderedBelief,
    probabilities: Sequence[float],
    feature_names: Sequence[str],
    units: Sequence[CandidateUnit],
    *,
    threshold: float,
    max_states: int,
) -> tuple[tuple[State, ...], int, int]:
    """Materialize all valid completions of a frozen feature threshold."""
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("abstention threshold must be in [0.5, 1]")
    index = {name: i for i, name in enumerate(feature_names)}
    committed: dict[str, tuple[bool, ...]] = {}
    abstained = 0
    logical_count = 1
    for unit in units:
        atom_probabilities = tuple(probabilities[index[atom]] for atom in unit.atoms)
        if unit.kind == "binary":
            confidence = max(atom_probabilities[0], 1.0 - atom_probabilities[0])
            assignment = (atom_probabilities[0] >= 0.5,)
        else:
            selected = max(range(len(unit.atoms)), key=lambda i: atom_probabilities[i])
            confidence = atom_probabilities[selected]
            assignment = tuple(i == selected for i in range(len(unit.atoms)))
        if confidence >= threshold:
            committed[unit.unit_id] = assignment
        else:
            abstained += 1
            logical_count *= unit.arity
    if logical_count > max_states:
        raise ValueError(
            f"abstention logical completion count {logical_count} exceeds {max_states}"
        )
    selected_states: list[State] = []
    for state in ordering.states:
        keep = True
        for unit in units:
            assignment = committed.get(unit.unit_id)
            if assignment is None:
                continue
            if tuple(state[index[atom]] for atom in unit.atoms) != assignment:
                keep = False
                break
        if keep:
            selected_states.append(state)
    if not selected_states:
        raise ValueError("abstention commitments have no valid completion")
    return tuple(selected_states), abstained, logical_count


def abstention_completions_by_count(
    ordering: OrderedBelief,
    probabilities: Sequence[float],
    feature_names: Sequence[str],
    units: Sequence[CandidateUnit],
    *,
    abstained_count: int,
    max_states: int,
) -> tuple[tuple[State, ...], tuple[str, ...], int]:
    """Materialize a nested, count-indexed abstention candidate.

    Units are released from least to most confident.  Every unit that remains
    committed is projected from the *common joint MAP state* rather than
    independently argmaxed.  This distinction matters for probability ties:
    ``abstained_count=0`` is therefore exactly the MAP singleton used by the
    MAP baseline, even for uniform categorical beliefs.
    """
    if not 0 <= abstained_count <= len(units):
        raise ValueError(
            "abstained_count must be between zero and the number of units"
        )
    if not ordering.states:
        raise ValueError("cannot abstain over an empty ordering")
    index = {name: i for i, name in enumerate(feature_names)}
    ranked: list[tuple[float, str, CandidateUnit]] = []
    for unit in units:
        atom_probabilities = tuple(probabilities[index[atom]] for atom in unit.atoms)
        if unit.kind == "binary":
            confidence = max(atom_probabilities[0], 1.0 - atom_probabilities[0])
        else:
            confidence = max(atom_probabilities)
        ranked.append((confidence, unit.unit_id, unit))
    ranked.sort(key=lambda item: (item[0], item[1]))
    abstained_units = tuple(
        item[2] for item in ranked[:abstained_count]
    )
    abstained_ids = tuple(unit.unit_id for unit in abstained_units)
    abstained_id_set = set(abstained_ids)

    logical_count = 1
    for unit in abstained_units:
        logical_count *= unit.arity
    if logical_count > max_states:
        raise ValueError(
            f"abstention logical completion count {logical_count} exceeds {max_states}"
        )

    map_state = ordering.states[0]
    selected_states: list[State] = []
    for state in ordering.states:
        if all(
            unit.unit_id in abstained_id_set
            or tuple(state[index[atom]] for atom in unit.atoms)
            == tuple(map_state[index[atom]] for atom in unit.atoms)
            for unit in units
        ):
            selected_states.append(state)
    if not selected_states:
        raise ValueError("abstention commitments have no valid completion")
    if abstained_count == 0 and tuple(selected_states) != ordering.states[:1]:
        raise AssertionError("zero-abstention candidate is not the MAP singleton")
    return tuple(selected_states), abstained_ids, logical_count


@dataclass(frozen=True)
class AdaptiveAbstentionAttempt:
    abstained_count: int
    completions: tuple[State, ...]
    abstained_unit_ids: tuple[str, ...]
    logical_completion_count: int
    planning_result: PlanningResult


@dataclass(frozen=True)
class AdaptiveAbstentionResult:
    selected: AdaptiveAbstentionAttempt
    attempts: tuple[AdaptiveAbstentionAttempt, ...]


def search_adaptive_abstention(
    ordering: OrderedBelief,
    probabilities: Sequence[float],
    feature_names: Sequence[str],
    units: Sequence[CandidateUnit],
    plan_candidate: Callable[[tuple[State, ...]], PlanningResult],
    *,
    singleton_result: PlanningResult,
    max_states: int,
) -> AdaptiveAbstentionResult:
    """Find the largest established-plannable nested abstention candidate."""
    cache: dict[int, AdaptiveAbstentionAttempt] = {}

    def evaluate(count: int) -> AdaptiveAbstentionAttempt:
        cached = cache.get(count)
        if cached is not None:
            return cached
        completions, abstained_ids, logical_count = (
            abstention_completions_by_count(
                ordering,
                probabilities,
                feature_names,
                units,
                abstained_count=count,
                max_states=max_states,
            )
        )
        result = singleton_result if count == 0 else plan_candidate(completions)
        attempt = AdaptiveAbstentionAttempt(
            abstained_count=count,
            completions=completions,
            abstained_unit_ids=abstained_ids,
            logical_completion_count=logical_count,
            planning_result=result,
        )
        cache[count] = attempt
        return attempt

    best = evaluate(0)
    if best.completions != ordering.states[:1]:
        raise AssertionError("adaptive zero-abstention candidate differs from MAP")
    if planning_result_found(singleton_result) and units:
        full_count = len(units)
        full = evaluate(full_count)
        if planning_result_found(full.planning_result):
            best = full
        else:
            lo, hi = 0, full_count
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                candidate = evaluate(mid)
                if planning_result_found(candidate.planning_result):
                    lo = mid
                    best = candidate
                else:
                    hi = mid
    return AdaptiveAbstentionResult(
        selected=best,
        attempts=tuple(cache[count] for count in sorted(cache)),
    )


def project_constraint_groups(
    raw_constraints: Iterable[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    index = {name: i for i, name in enumerate(feature_names)}
    groups: list[tuple[int, ...]] = []
    for raw in raw_constraints:
        if raw.get("type", "at_most_one") != "at_most_one":
            continue
        group = tuple(
            sorted(
                index[atom]
                for atom in raw.get("atoms", ())
                if atom in index
            )
        )
        if len(group) >= 2 and group not in groups:
            groups.append(group)
    return tuple(groups)
