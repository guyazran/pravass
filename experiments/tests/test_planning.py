from __future__ import annotations

import unittest
from unittest.mock import patch

from unified_planning.shortcuts import BoolType, Fluent, InstantaneousAction, Problem


class _EmptyPlan:
    actions = ()


class _CapturingQueue:
    def put(self, payload):
        self.payload = payload


def _goal_problem(*, initially_at_goal: bool) -> Problem:
    problem = Problem("empty-plan-test")
    at_goal = Fluent("at_goal")
    problem.add_fluent(at_goal, default_initial_value=False)
    problem.set_initial_value(at_goal, initially_at_goal)
    problem.add_goal(at_goal)
    return problem


def _finish_unless_blocked_problem() -> Problem:
    problem = Problem("finish-unless-blocked-test")
    blocker = Fluent("blocker", BoolType())
    done = Fluent("done", BoolType())
    problem.add_fluent(blocker, default_initial_value=False)
    problem.add_fluent(done, default_initial_value=False)
    problem.set_initial_value(blocker(), False)
    problem.set_initial_value(done(), False)
    finish = InstantaneousAction("finish")
    finish.add_precondition(~blocker())
    finish.add_effect(done(), True)
    problem.add_action(finish)
    problem.add_goal(done())
    return problem


def _finish_if_enabled_problem() -> Problem:
    problem = Problem("finish-if-enabled-test")
    enabled = Fluent("enabled", BoolType())
    done = Fluent("done", BoolType())
    problem.add_fluent(enabled, default_initial_value=False)
    problem.add_fluent(done, default_initial_value=False)
    problem.set_initial_value(enabled(), False)
    problem.set_initial_value(done(), False)
    finish = InstantaneousAction("finish")
    finish.add_precondition(enabled())
    finish.add_effect(done(), True)
    problem.add_action(finish)
    problem.add_goal(done())
    return problem


def _two_unknowns_problem() -> Problem:
    problem = Problem("two-unknowns-test")
    left = Fluent("left", BoolType())
    right = Fluent("right", BoolType())
    problem.add_fluent(left, default_initial_value=False)
    problem.add_fluent(right, default_initial_value=False)
    problem.set_initial_value(left(), False)
    problem.set_initial_value(right(), False)
    return problem


class PlanningEmptyPlanTests(unittest.TestCase):
    def test_execute_empty_plan_succeeds_when_initial_state_is_goal(self) -> None:
        from mbs_experiments.planning import execute_plan

        success, unsafe_first, pre_fail = execute_plan(
            _goal_problem(initially_at_goal=True), ()
        )

        self.assertTrue(success)
        self.assertFalse(unsafe_first)
        self.assertFalse(pre_fail)

    def test_execute_empty_plan_fails_when_initial_state_is_not_goal(self) -> None:
        from mbs_experiments.planning import execute_plan

        success, unsafe_first, pre_fail = execute_plan(
            _goal_problem(initially_at_goal=False), ()
        )

        self.assertFalse(success)
        self.assertFalse(unsafe_first)
        self.assertFalse(pre_fail)

    def test_oracle_empty_plan_is_success_when_initial_state_is_goal(self) -> None:
        from mbs_experiments.planning import run_oracle_planning

        problem = _goal_problem(initially_at_goal=True)
        with patch("mbs_experiments.planning.parse_classical_problem", return_value=problem), patch(
            "mbs_experiments.planning.solve_classical",
            return_value=("PlanGenerationResultStatus.SOLVED_SATISFICING", _EmptyPlan(), 0.1),
        ):
            result = run_oracle_planning("domain.pddl", "problem.pddl", 30.0)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.plan_length, 0)
        self.assertEqual(result.actions, ())
        self.assertTrue(result.success)
        self.assertFalse(result.no_plan_fn)

    def test_singleton_empty_plan_is_not_no_plan_when_initial_state_is_goal(self) -> None:
        from mbs_experiments.planning import run_singleton_classical_planning

        problem = _goal_problem(initially_at_goal=True)
        with patch("mbs_experiments.planning.parse_classical_problem", return_value=problem), patch(
            "mbs_experiments.planning.solve_classical",
            return_value=("PlanGenerationResultStatus.SOLVED_SATISFICING", _EmptyPlan(), 0.1),
        ):
            result = run_singleton_classical_planning(
                "domain.pddl",
                "problem.pddl",
                (),
                (),
                30.0,
                oracle_success=True,
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.plan_length, 0)
        self.assertEqual(result.actions, ())
        self.assertTrue(result.success)
        self.assertFalse(result.no_plan_fn)

    def test_belief_worker_skips_cpor_when_initial_belief_entails_goal(self) -> None:
        from mbs_experiments.planning import _belief_planning_worker

        problem = _goal_problem(initially_at_goal=True)
        out_queue = _CapturingQueue()
        with patch("mbs_experiments.planning.parse_classical_problem", return_value=problem), patch(
            "mbs_experiments.planning.register_cpor_meta_engine",
            side_effect=AssertionError("CPOR should not be registered"),
        ), patch(
            "mbs_experiments.planning.solve_conformant",
            side_effect=AssertionError("CPOR should not be called"),
        ):
            _belief_planning_worker(
                "domain.pddl",
                "problem.pddl",
                (),
                ((),),
                (),
                (),
                False,
                30.0,
                out_queue,
            )

        self.assertEqual(
            out_queue.payload,
            ("ok", "ALREADY_GOAL", True, (), True, False, False),
        )

    def test_initial_belief_goal_check_is_conservative_for_unknown_goal_fluents(self) -> None:
        from mbs_experiments.planning import all_belief_states_satisfy_initial_goals

        problem = _goal_problem(initially_at_goal=True)

        self.assertFalse(
            all_belief_states_satisfy_initial_goals(
                problem,
                feature_names=(),
                belief_states=((),),
                unknown_feature_names=("at_goal()",),
            )
        )


class ContingentBeliefConstructionTests(unittest.TestCase):
    def test_unknown_fluent_is_hidden_and_not_left_as_known_initial_value(self) -> None:
        from mbs_experiments.planning import atom_to_fnode, clone_as_contingent_with_belief

        problem = _finish_unless_blocked_problem()
        contingent = clone_as_contingent_with_belief(
            problem,
            feature_names=(),
            belief_states=((),),
            unknown_feature_names=("blocker()",),
        )
        blocker = atom_to_fnode(contingent, "blocker()")
        not_blocker = contingent.environment.expression_manager.Not(blocker)

        self.assertNotIn(blocker, contingent.explicit_initial_values)
        self.assertIn(blocker, contingent.hidden_fluents)
        self.assertIn(not_blocker, contingent.hidden_fluents)
        self.assertIn([not_blocker, blocker], contingent.or_constraints)

    def test_at_most_one_initial_constraints_are_added_to_contingent_problem(self) -> None:
        from mbs_experiments.planning import atom_to_fnode, clone_as_contingent_with_belief

        problem = _two_unknowns_problem()
        contingent = clone_as_contingent_with_belief(
            problem,
            feature_names=(),
            belief_states=((),),
            unknown_feature_names=("left()", "right()"),
            initial_constraint_groups=(("left()", "right()"),),
        )
        left = atom_to_fnode(contingent, "left()")
        right = atom_to_fnode(contingent, "right()")
        em = contingent.environment.expression_manager

        self.assertIn([em.Not(left), em.Not(right)], contingent.or_constraints)

    def test_exactly_one_initial_constraints_are_added_to_contingent_problem(self) -> None:
        from mbs_experiments.planning import atom_to_fnode, clone_as_contingent_with_belief

        problem = _two_unknowns_problem()
        contingent = clone_as_contingent_with_belief(
            problem,
            feature_names=(),
            belief_states=((),),
            unknown_feature_names=("left()", "right()"),
            initial_constraint_groups=(("left()", "right()"),),
            initial_constraints_exactly_one=True,
        )
        left = atom_to_fnode(contingent, "left()")
        right = atom_to_fnode(contingent, "right()")

        self.assertIn([left, right], contingent.oneof_constraints)

    def test_planner_does_not_assume_unknown_fluent_is_false(self) -> None:
        from mbs_experiments.planning import clone_as_contingent_with_belief, solve_conformant

        problem = _finish_unless_blocked_problem()
        contingent = clone_as_contingent_with_belief(
            problem,
            feature_names=(),
            belief_states=((),),
            unknown_feature_names=("blocker()",),
        )

        status, plan, _runtime = solve_conformant(contingent, 5.0)

        self.assertIsNone(plan, status)

    def test_planner_uses_known_chosen_fluent_value(self) -> None:
        from mbs_experiments.planning import atom_to_fnode, clone_as_contingent_with_belief, flatten_plan_actions, solve_conformant

        problem = _finish_if_enabled_problem()
        contingent = clone_as_contingent_with_belief(
            problem,
            feature_names=("enabled()",),
            belief_states=((True,),),
            unknown_feature_names=(),
        )
        enabled = atom_to_fnode(contingent, "enabled()")

        self.assertTrue(contingent.initial_value(enabled).is_true())
        status, plan, _runtime = solve_conformant(contingent, 5.0)

        self.assertIsNotNone(plan, status)
        self.assertEqual(tuple(map(str, flatten_plan_actions(plan))), ("finish",))


if __name__ == "__main__":
    unittest.main()
