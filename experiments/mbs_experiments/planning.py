from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_mod
import re
import resource
import shutil
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import unified_planning.environment as up_environment
from unified_planning.io import PDDLReader
from unified_planning.model.contingent import ContingentProblem
from unified_planning.plans import ActionInstance
from unified_planning.shortcuts import OneshotPlanner, SequentialSimulator

from .states import State

ATOM_RE = re.compile(r"^([^()]+)\((.*)\)$")


@dataclass(frozen=True)
class PlanningResult:
    status: str
    planner_status: str | None
    runtime_seconds: float
    plan_length: int | None
    actions: tuple[str, ...]
    success: bool
    unsafe_first: bool
    pre_fail: bool
    no_plan_fn: bool
    error_message: str | None = None


def parse_atom(atom: str) -> tuple[str, tuple[str, ...]]:
    match = ATOM_RE.match(atom)
    if not match:
        raise ValueError(f"Unsupported atom format: {atom}")
    pred = match.group(1)
    raw_args = match.group(2)
    args = tuple(part for part in raw_args.split(",") if part)
    return pred, args


def atom_to_fnode(problem, atom: str):
    pred, args = parse_atom(atom)
    fluent = problem.fluent(pred)
    em = problem.environment.expression_manager
    obj_exprs = tuple(em.ObjectExp(problem.object(arg)) for arg in args)
    return em.FluentExp(fluent, obj_exprs)


def atoms_to_fnodes(problem, atoms: Sequence[str]) -> set[Any]:
    return {atom_to_fnode(problem, atom) for atom in atoms}


def _coerce_names_to_plain_str(problem) -> None:
    """Replace str-subclass names (pddl.custom_types.name) with plain ``str``.

    The pddl-package parser backend returns RegexConstrainedString names, which
    pythonnet refuses to convert to System.String, breaking the CPOR converter
    (``Domain(problem.name)`` → "No method matches given arguments"). Equality
    and hashing are value-based, so in-place coercion is safe for all lookups.
    """
    problem._name = str(problem._name)
    for user_type in problem.user_types:
        user_type._name = str(user_type._name)
    for obj in problem.all_objects:
        obj._name = str(obj._name)
    for fluent in problem.fluents:
        fluent._name = str(fluent._name)
    for action in problem.actions:
        action._name = str(action._name)
        for parameter in action.parameters:
            parameter._name = str(parameter._name)


def parse_classical_problem(domain_path: Path, problem_path: Path):
    reader = PDDLReader()
    problem = reader.parse_problem(str(domain_path), str(problem_path))
    _coerce_names_to_plain_str(problem)
    return problem


def _add_initial_constraint_groups(
    contingent: ContingentProblem,
    constraint_groups: Sequence[Sequence[str]],
    *,
    exactly_one: bool = False,
) -> None:
    """Add lifted initial-state mutex/exact-one constraints to a contingent problem."""
    em = contingent.environment.expression_manager
    seen_groups: set[tuple[str, ...]] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for raw_group in constraint_groups:
        group = tuple(dict.fromkeys(str(atom) for atom in raw_group))
        if len(group) < 2:
            if exactly_one and len(group) == 1:
                fluent_exp = atom_to_fnode(contingent, group[0])
                contingent.add_oneof_initial_constraint([fluent_exp])
            continue
        canonical_group = tuple(sorted(group))
        if exactly_one and canonical_group not in seen_groups:
            seen_groups.add(canonical_group)
            contingent.add_oneof_initial_constraint(
                [atom_to_fnode(contingent, atom) for atom in group]
            )
            continue
        for i, left_atom in enumerate(group):
            left = atom_to_fnode(contingent, left_atom)
            for right_atom in group[i + 1 :]:
                pair = tuple(sorted((left_atom, right_atom)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                right = atom_to_fnode(contingent, right_atom)
                contingent.add_or_initial_constraint([em.Not(left), em.Not(right)])


def clone_as_contingent_with_belief(
    base_problem,
    feature_names: Sequence[str],
    belief_states: Sequence[State],
    unknown_feature_names: Sequence[str] = (),
    initial_constraint_groups: Sequence[Sequence[str]] = (),
    initial_constraints_exactly_one: bool = False,
    exactly_one_initial_constraint_groups: Sequence[Sequence[str]] = (),
    at_least_one_initial_constraint_groups: Sequence[Sequence[str]] = (),
) -> ContingentProblem:
    contingent = ContingentProblem(base_problem.name, base_problem.environment)
    contingent._fluents = base_problem._fluents[:]
    contingent._actions = [action.clone() for action in base_problem._actions]
    contingent._user_types = base_problem._user_types[:]
    contingent._user_types_hierarchy = base_problem._user_types_hierarchy.copy()
    contingent._objects = base_problem._objects[:]
    contingent._timed_effects = {
        t: [effect.clone() for effect in effects]
        for t, effects in base_problem._timed_effects.items()
    }
    contingent._timed_goals = {
        interval: [goal for goal in goals]
        for interval, goals in base_problem._timed_goals.items()
    }
    contingent._goals = base_problem._goals[:]
    contingent._metrics = list(base_problem._metrics)
    contingent._initial_defaults = base_problem._initial_defaults.copy()
    contingent._fluents_defaults = base_problem._fluents_defaults.copy()

    feature_fluents = atoms_to_fnodes(contingent, feature_names)
    unknown_fluents = atoms_to_fnodes(contingent, unknown_feature_names)
    overlap = feature_fluents & unknown_fluents
    if overlap:
        overlap_text = ", ".join(sorted(map(str, overlap)))
        raise ValueError(f"Fluents cannot be both known-belief and unknown: {overlap_text}")
    uncertain_fluents = feature_fluents | unknown_fluents
    
    for fluent_exp, value in base_problem.initial_values.items():
        if fluent_exp in uncertain_fluents:
            continue
        contingent.set_initial_value(fluent_exp, value)

    em = contingent.environment.expression_manager
    for fluent_exp in sorted(unknown_fluents, key=str):
        contingent.add_unknown_initial_constraint(fluent_exp)

    _add_initial_constraint_groups(
        contingent,
        initial_constraint_groups,
        exactly_one=initial_constraints_exactly_one,
    )
    # Groups with exactly-one semantics regardless of the global flag, e.g.
    # candidate-location groups ("the object is in exactly one receptacle").
    _add_initial_constraint_groups(
        contingent,
        exactly_one_initial_constraint_groups,
        exactly_one=True,
    )
    # Disjunctive at-least-one groups, e.g. candidate locations of an object
    # that legitimately occupies several receptacles at once.
    for raw_group in at_least_one_initial_constraint_groups:
        atoms = tuple(dict.fromkeys(str(atom) for atom in raw_group))
        if not atoms:
            continue
        contingent.add_or_initial_constraint(
            [atom_to_fnode(contingent, atom) for atom in atoms]
        )

    if not belief_states:
        return contingent

    unique_state_assignments = []
    seen_signatures = set()
    for state in belief_states:
        assignments = {}
        for atom, value in zip(feature_names, state):
            fluent_exp = atom_to_fnode(contingent, atom)
            assignments[fluent_exp] = value
            
        signature = tuple((str(f), v) for f, v in sorted(assignments.items(), key=lambda item: str(item[0])))
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            unique_state_assignments.append(assignments)

    shared_fluents = None
    for assignments in unique_state_assignments:
        state_fluents = set(assignments.keys())
        if shared_fluents is None:
            shared_fluents = state_fluents
        else:
            shared_fluents &= state_fluents
    
    shared_assignments = {}
    if shared_fluents is not None:
        for fluent_exp in sorted(shared_fluents, key=str):
            values = {assignments[fluent_exp] for assignments in unique_state_assignments}
            if len(values) == 1:
                shared_value = next(iter(values))
                contingent.set_initial_value(fluent_exp, shared_value)
                shared_assignments[fluent_exp] = shared_value

    residual_state_assignments = []
    residual_fluents = set()
    for assignments in unique_state_assignments:
        residual_assignments = {
            fluent_exp: value
            for fluent_exp, value in assignments.items()
            if fluent_exp not in shared_assignments
        }
        residual_state_assignments.append(residual_assignments)
        residual_fluents.update(residual_assignments.keys())

    if not any(residual_state_assignments):
        return contingent

    for fluent_exp in residual_fluents:
        contingent._hidden_fluents.add(fluent_exp)
        contingent._hidden_fluents.add(em.Not(fluent_exp))

    from unified_planning.model import Fluent
    existing_names = {fluent.name for fluent in contingent.fluents}
    case_tags = []
    next_index = 0
    for _ in range(len(residual_state_assignments)):
        while True:
            candidate_name = f"possible_initial_state_case_{next_index}"
            next_index += 1
            if candidate_name not in existing_names:
                existing_names.add(candidate_name)
                break
        case_fluent = Fluent(candidate_name, environment=contingent.environment)
        contingent.add_fluent(case_fluent, default_initial_value=False)
        case_tags.append(em.FluentExp(case_fluent))

    contingent.add_oneof_initial_constraint(case_tags)

    for case_tag, assignments in zip(case_tags, residual_state_assignments):
        for fluent_exp, value in sorted(assignments.items(), key=lambda item: str(item[0])):
            literal = fluent_exp if value else em.Not(fluent_exp)
            contingent.add_or_initial_constraint([em.Not(case_tag), literal])

    return contingent


_MONO_LIB_CANDIDATES = (
    "/opt/homebrew/lib/libmonosgen-2.0.dylib",
    "/usr/local/lib/libmonosgen-2.0.dylib",
    "/usr/lib/x86_64-linux-gnu/libmonosgen-2.0.so.1",
    "/usr/lib/libmonosgen-2.0.so.1",
)


def configure_pythonnet_runtime() -> None:
    try:
        import pythonnet  # type: ignore
    except ImportError:
        return
    try:
        if pythonnet.get_runtime_info() is not None:
            return
    except Exception:
        pass
    libmono = os.environ.get("PYTHONNET_MONO_LIBMONO")
    if not libmono:
        for candidate in _MONO_LIB_CANDIDATES:
            if Path(candidate).exists():
                libmono = candidate
                break
    if not libmono and shutil.which("mono"):
        mono_prefix = Path(shutil.which("mono")).resolve().parent.parent
        for ext in (".dylib", ".so", ".so.1"):
            candidate = mono_prefix / "lib" / f"libmonosgen-2.0{ext}"
            if candidate.exists():
                libmono = str(candidate)
                break
    try:
        if libmono:
            pythonnet.load("mono", libmono=libmono)
        else:
            pythonnet.load("mono")
    except Exception:
        pass


def install_up_cpor_compatibility_shims() -> None:
    configure_pythonnet_runtime()
    try:
        import unified_planning.engines.mixins.action_selector  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        action_selector_mod = types.ModuleType(
            "unified_planning.engines.mixins.action_selector"
        )

        class ActionSelectorMixin:
            pass

        action_selector_mod.ActionSelectorMixin = ActionSelectorMixin
        sys.modules[
            "unified_planning.engines.mixins.action_selector"
        ] = action_selector_mod

    try:
        import unified_planning.model.contingent.contingent_problem  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        contingent_pkg = types.ModuleType("unified_planning.model.contingent")
        contingent_pkg.__path__ = []
        sys.modules["unified_planning.model.contingent"] = contingent_pkg
        import unified_planning.model.contingent as contingent_problem_mod

        sys.modules[
            "unified_planning.model.contingent.contingent_problem"
        ] = contingent_problem_mod


def register_cpor_meta_engine() -> None:
    install_up_cpor_compatibility_shims()
    env = up_environment.get_environment()
    env.credits_stream = None
    meta_engines = getattr(env.factory, "_meta_engines", {})
    if "MetaCPORPlanning" not in meta_engines:
        env.factory.add_meta_engine(
            "MetaCPORPlanning", "up_cpor.engine", "CPORMetaEngineImpl"
        )


def solve_classical(problem, timeout_seconds: float) -> tuple[str, Any, float]:
    up_environment.get_environment().credits_stream = None
    start = time.perf_counter()
    with OneshotPlanner(name="fast-downward") as planner:
        result = planner.solve(problem, timeout=timeout_seconds)
    return str(result.status), result.plan, time.perf_counter() - start


def solve_conformant(problem: ContingentProblem, timeout_seconds: float) -> tuple[str, Any, float]:
    register_cpor_meta_engine()
    start = time.perf_counter()
    with OneshotPlanner(name="MetaCPORPlanning[fast-downward]") as planner:
        result = planner.solve(problem, timeout=timeout_seconds)
    return str(result.status), result.plan, time.perf_counter() - start


def flatten_plan_actions(plan: Any) -> tuple[ActionInstance, ...]:
    if plan is None:
        return ()
    if hasattr(plan, "actions"):
        return tuple(plan.actions)
    if hasattr(plan, "root_node"):
        actions: list[ActionInstance] = []
        node = plan.root_node
        visited = set()
        while node is not None and id(node) not in visited:
            visited.add(id(node))
            if hasattr(node, "action_instance") and node.action_instance is not None:
                actions.append(node.action_instance)
            elif hasattr(node, "action") and node.action is not None:
                actions.append(node.action)
            children = []
            if hasattr(node, "children"):
                raw_children = node.children
                if isinstance(raw_children, dict):
                    children = list(raw_children.values())
                else:
                    children = [child[1] if isinstance(child, tuple) and len(child) == 2 else child for child in raw_children]
            node = children[0] if children else None
        return tuple(actions)
    return ()


def execute_plan(problem, actions: Sequence[ActionInstance]) -> tuple[bool, bool, bool]:
    with SequentialSimulator(problem=problem) as simulator:
        state = simulator.get_initial_state()
        unsafe_first = False
        pre_fail = False
        if not actions:
            return bool(simulator.is_goal(state)), unsafe_first, pre_fail
        for idx, action in enumerate(actions):
            if not simulator.is_applicable(state, action):
                if idx == 0:
                    unsafe_first = True
                else:
                    pre_fail = True
                return False, unsafe_first, pre_fail
            state = simulator.apply(state, action)
        return bool(simulator.is_goal(state)), unsafe_first, pre_fail


def _goal_fluent_expressions(expr) -> set[Any]:
    fluents: set[Any] = set()
    if expr.is_fluent_exp():
        fluents.add(expr)
        return fluents
    for arg in getattr(expr, "args", ()):  # recursively handles And/Not/etc.
        fluents.update(_goal_fluent_expressions(arg))
    return fluents


def _goals_reference_any(problem, fluent_exps: set[Any]) -> bool:
    if not fluent_exps:
        return False
    for goal in problem.goals:
        if _goal_fluent_expressions(goal) & fluent_exps:
            return True
    return False


def _belief_state_initial_problem(
    base_problem,
    feature_names: Sequence[str],
    belief_state: State,
):
    problem = base_problem.clone()
    em = problem.environment.expression_manager
    for atom, value in zip(feature_names, belief_state):
        fluent_exp = atom_to_fnode(problem, atom)
        problem.set_initial_value(fluent_exp, em.Bool(bool(value)))
    return problem


def _actions_solve_all_explicit_belief_states(
    base_problem,
    feature_names: Sequence[str],
    belief_states: Sequence[State],
    actions: Sequence[ActionInstance],
) -> bool:
    for belief_state in belief_states:
        problem = _belief_state_initial_problem(base_problem, feature_names, belief_state)
        success, _unsafe_first, _pre_fail = execute_plan(problem, actions)
        if not success:
            return False
    return True


def find_classical_candidate_conformant_plan(
    base_problem,
    feature_names: Sequence[str],
    belief_states: Sequence[State],
    timeout_seconds: float,
) -> tuple[str | None, tuple[ActionInstance, ...] | None]:
    """Try classical plans for explicit states and keep one valid for all states.

    This is an incomplete but sound shortcut for fully explicit small beliefs:
    any returned plan is checked by simulation against every explicit belief
    state before it is accepted as conformant. It mitigates CPOR false negatives
    on small deterministic belief sets without changing the conformant-planning
    semantics.
    """
    unique_states = tuple(dict.fromkeys(tuple(state) for state in belief_states))
    if not unique_states:
        return None, None
    per_state_timeout = max(1.0, float(timeout_seconds) / max(1, len(unique_states)))
    seen_action_signatures: set[tuple[str, ...]] = set()
    last_status: str | None = None
    for belief_state in unique_states:
        plan_problem = _belief_state_initial_problem(
            base_problem, feature_names, belief_state
        )
        planner_status, plan, _runtime = solve_classical(plan_problem, per_state_timeout)
        last_status = planner_status
        if plan is None:
            continue
        actions = flatten_plan_actions(plan)
        signature = tuple(map(str, actions))
        if signature in seen_action_signatures:
            continue
        seen_action_signatures.add(signature)
        if _actions_solve_all_explicit_belief_states(
            base_problem, feature_names, unique_states, actions
        ):
            return planner_status, actions
    return last_status, None


def all_belief_states_satisfy_initial_goals(
    base_problem,
    feature_names: Sequence[str],
    belief_states: Sequence[State],
    unknown_feature_names: Sequence[str] = (),
) -> bool:
    """Return True when the empty plan is guaranteed for the explicit belief.

    CPOR currently reports no plan for some contingent problems whose initial
    belief already entails the goals. This pre-check recognizes that case before
    invoking CPOR. It is intentionally conservative with hidden unknown fluents:
    if a goal references an unknown fluent, we let the planner handle it instead
    of assuming the base problem's ground-truth value.
    """
    if not belief_states:
        return False
    unknown_fluents = atoms_to_fnodes(base_problem, unknown_feature_names)
    if _goals_reference_any(base_problem, unknown_fluents):
        return False
    for belief_state in belief_states:
        problem = _belief_state_initial_problem(
            base_problem, feature_names, belief_state
        )
        with SequentialSimulator(problem=problem) as simulator:
            if not bool(simulator.is_goal(simulator.get_initial_state())):
                return False
    return True


def run_oracle_planning(
    domain_path: Path,
    problem_path: Path,
    timeout_seconds: float,
) -> PlanningResult:
    start = time.perf_counter()
    try:
        problem = parse_classical_problem(domain_path, problem_path)
        planner_status, plan, runtime = solve_classical(problem, timeout_seconds)
        plan_found = plan is not None
        actions = flatten_plan_actions(plan)
        success, unsafe_first, pre_fail = (
            execute_plan(problem, actions) if plan_found else (False, False, False)
        )
        status = "ok" if plan_found else "no_plan"
        return PlanningResult(
            status,
            planner_status,
            runtime,
            len(actions) if plan_found else None,
            tuple(map(str, actions)),
            success,
            unsafe_first,
            pre_fail,
            False,
        )
    except Exception as exc:
        return PlanningResult(
            "error",
            None,
            time.perf_counter() - start,
            None,
            (),
            False,
            False,
            False,
            False,
            f"{type(exc).__name__}: {exc}",
        )


def run_singleton_classical_planning(
    domain_path: Path,
    problem_path: Path,
    feature_names: Sequence[str],
    belief_state: State,
    timeout_seconds: float,
    *,
    oracle_success: bool,
) -> PlanningResult:
    start = time.perf_counter()
    try:
        gt_problem = parse_classical_problem(domain_path, problem_path)
        plan_problem = parse_classical_problem(domain_path, problem_path)
        em = plan_problem.environment.expression_manager
        for atom, value in zip(feature_names, belief_state):
            fluent_exp = atom_to_fnode(plan_problem, atom)
            plan_problem.set_initial_value(fluent_exp, em.Bool(bool(value)))
        planner_status, plan, runtime = solve_classical(plan_problem, timeout_seconds)
        plan_found = plan is not None
        actions = flatten_plan_actions(plan)
        success, unsafe_first, pre_fail = (
            execute_plan(gt_problem, actions) if plan_found else (False, False, False)
        )
        no_plan_fn = bool(oracle_success and not plan_found)
        status = "ok" if plan_found else "no_plan"
        return PlanningResult(
            status,
            planner_status,
            runtime,
            len(actions) if plan_found else None,
            tuple(map(str, actions)),
            success,
            unsafe_first,
            pre_fail,
            no_plan_fn,
        )
    except Exception as exc:
        return PlanningResult(
            "error",
            None,
            time.perf_counter() - start,
            None,
            (),
            False,
            False,
            False,
            False,
            f"{type(exc).__name__}: {exc}",
        )


def _apply_planner_memory_cap() -> None:
    """Optionally cap planner-subprocess address space via RLIMIT_AS.

    Disabled by default. The worker subprocess re-imports the project's stack
    (UP + numpy + pythonnet/mono), which reserves several GiB of VAS at startup,
    so a too-low cap fails the worker before any planning happens. Use your job
    scheduler's memory limit as the primary node-safety budget; opt into this
    RLIMIT_AS ceiling via ``MBS_PLANNER_MAX_GIB`` for a per-task cap on top of
    it. 0 / unset disables.
    """
    try:
        gib = float(os.environ.get("MBS_PLANNER_MAX_GIB", "0"))
    except ValueError:
        return
    if gib <= 0:
        return
    limit = int(gib * (1 << 30))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError):
        # Lower than existing hard cap or otherwise refused — leave uncapped
        # rather than fail the planning task on a config issue.
        pass


def _belief_planning_worker(
    domain_path_str: str,
    problem_path_str: str,
    feature_names: Sequence[str],
    belief_states: Sequence[State],
    unknown_feature_names: Sequence[str],
    initial_constraint_groups: Sequence[Sequence[str]],
    initial_constraints_exactly_one: bool,
    exactly_one_initial_constraint_groups: Sequence[Sequence[str]] | float = (),
    at_least_one_initial_constraint_groups: Sequence[Sequence[str]] | Any = (),
    timeout_seconds: float | None = None,
    out_queue=None,
) -> None:
    # Backward-compatible positional form used before the two mixed-semantics
    # constraint arguments were added:
    # (..., initial_constraints_exactly_one, timeout_seconds, out_queue).
    if (
        timeout_seconds is None
        and out_queue is None
        and isinstance(exactly_one_initial_constraint_groups, (int, float))
    ):
        timeout_seconds = float(exactly_one_initial_constraint_groups)
        out_queue = at_least_one_initial_constraint_groups
        exactly_one_initial_constraint_groups = ()
        at_least_one_initial_constraint_groups = ()
    if timeout_seconds is None or out_queue is None:
        raise TypeError("timeout_seconds and out_queue are required")
    try:
        _apply_planner_memory_cap()
        domain_path = Path(domain_path_str)
        problem_path = Path(problem_path_str)
        configure_pythonnet_runtime()
        base_problem = parse_classical_problem(domain_path, problem_path)
        if all_belief_states_satisfy_initial_goals(
            base_problem, feature_names, belief_states, unknown_feature_names
        ):
            success, unsafe_first, pre_fail = execute_plan(base_problem, ())
            out_queue.put(
                (
                    "ok",
                    "ALREADY_GOAL",
                    True,
                    (),
                    bool(success),
                    bool(unsafe_first),
                    bool(pre_fail),
                )
            )
            return

        unique_belief_states = tuple(dict.fromkeys(tuple(state) for state in belief_states))
        if len(unique_belief_states) == 1 and not unknown_feature_names:
            # A singleton, fully specified belief is an ordinary classical
            # planning problem. Solving it directly is equivalent to conformant
            # planning over one state and avoids CPOR false negatives.
            plan_problem = _belief_state_initial_problem(
                base_problem, feature_names, unique_belief_states[0]
            )
            planner_status, plan, _ = solve_classical(plan_problem, timeout_seconds)
            plan_found = plan is not None
            actions = flatten_plan_actions(plan)
            success, unsafe_first, pre_fail = (
                execute_plan(base_problem, actions)
                if plan_found
                else (False, False, False)
            )
            out_queue.put(
                (
                    "ok",
                    planner_status,
                    bool(plan_found),
                    tuple(map(str, actions)),
                    bool(success),
                    bool(unsafe_first),
                    bool(pre_fail),
                )
            )
            return

        if len(unique_belief_states) <= 20 and not unknown_feature_names:
            planner_status, candidate_actions = find_classical_candidate_conformant_plan(
                base_problem,
                feature_names,
                unique_belief_states,
                min(timeout_seconds, 10.0),
            )
            if candidate_actions is not None:
                success, unsafe_first, pre_fail = execute_plan(
                    base_problem, candidate_actions
                )
                out_queue.put(
                    (
                        "ok",
                        planner_status or "CLASSICAL_CANDIDATE",
                        True,
                        tuple(map(str, candidate_actions)),
                        bool(success),
                        bool(unsafe_first),
                        bool(pre_fail),
                    )
                )
                return

        register_cpor_meta_engine()
        contingent = clone_as_contingent_with_belief(
            base_problem,
            feature_names,
            belief_states,
            unknown_feature_names,
            initial_constraint_groups,
            initial_constraints_exactly_one,
            exactly_one_initial_constraint_groups,
            at_least_one_initial_constraint_groups,
        )
        planner_status, plan, _ = solve_conformant(contingent, timeout_seconds)
        plan_found = plan is not None
        actions = flatten_plan_actions(plan)
        success, unsafe_first, pre_fail = (
            execute_plan(base_problem, actions) if plan_found else (False, False, False)
        )
        out_queue.put(
            (
                "ok",
                planner_status,
                bool(plan_found),
                tuple(map(str, actions)),
                bool(success),
                bool(unsafe_first),
                bool(pre_fail),
            )
        )
    except Exception as exc:
        out_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def run_belief_planning(
    domain_path: Path,
    problem_path: Path,
    feature_names: Sequence[str],
    belief_states: Sequence[State],
    timeout_seconds: float,
    *,
    oracle_success: bool,
    unknown_feature_names: Sequence[str] = (),
    initial_constraint_groups: Sequence[Sequence[str]] = (),
    initial_constraints_exactly_one: bool = False,
    exactly_one_initial_constraint_groups: Sequence[Sequence[str]] = (),
    at_least_one_initial_constraint_groups: Sequence[Sequence[str]] = (),
) -> PlanningResult:
    if not belief_states:
        return PlanningResult(
            "no_plan",
            None,
            0.0,
            None,
            (),
            False,
            False,
            False,
            bool(oracle_success),
            "empty_belief_set",
        )

    start = time.perf_counter()
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    proc = ctx.Process(
        target=_belief_planning_worker,
        args=(
            str(domain_path),
            str(problem_path),
            tuple(feature_names),
            tuple(tuple(state) for state in belief_states),
            tuple(unknown_feature_names),
            tuple(tuple(group) for group in initial_constraint_groups),
            bool(initial_constraints_exactly_one),
            tuple(tuple(group) for group in exactly_one_initial_constraint_groups),
            tuple(tuple(group) for group in at_least_one_initial_constraint_groups),
            float(timeout_seconds),
            out_queue,
        ),
        daemon=True,
    )
    proc.start()
    grace = max(5.0, 0.25 * timeout_seconds)
    proc.join(timeout=timeout_seconds + grace)
    if proc.is_alive():
        proc.terminate()
        proc.join(2.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return PlanningResult(
            "timeout",
            None,
            time.perf_counter() - start,
            None,
            (),
            False,
            False,
            False,
            bool(oracle_success),
            "subprocess_timeout",
        )

    try:
        payload = out_queue.get_nowait()
    except queue_mod.Empty:
        return PlanningResult(
            "error",
            None,
            time.perf_counter() - start,
            None,
            (),
            False,
            False,
            False,
            False,
            f"subprocess_exit_{proc.exitcode}",
        )

    runtime = time.perf_counter() - start
    if payload[0] == "error":
        return PlanningResult(
            "error", None, runtime, None, (), False, False, False, False, payload[1]
        )

    _, planner_status, plan_found, action_strs, success, unsafe_first, pre_fail = payload
    no_plan_fn = bool(oracle_success and not plan_found)
    status = "ok" if plan_found else "no_plan"
    return PlanningResult(
        status,
        planner_status,
        runtime,
        len(action_strs) if plan_found else None,
        action_strs,
        success,
        unsafe_first,
        pre_fail,
        no_plan_fn,
    )
