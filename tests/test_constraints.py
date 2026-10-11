"""Constraint and FunctionConstraint."""

import pytest

from pravass import Constraint, FunctionConstraint


def test_constraint_is_abstract():
    with pytest.raises(TypeError):
        Constraint()


def test_default_support_is_unknown():
    class AlwaysValid(Constraint):
        def __call__(self, state):
            return True

    assert AlwaysValid().support is None


def test_function_constraint_delegates():
    constraint = FunctionConstraint(lambda s: "a" not in s)
    assert constraint(frozenset({"b"})) is True
    assert constraint(frozenset({"a"})) is False


def test_function_constraint_coerces_to_bool():
    assert FunctionConstraint(lambda s: len(s))(frozenset()) is False


def test_function_constraint_support():
    constraint = FunctionConstraint(lambda s: True, support=["a", "b"])
    assert constraint.support == frozenset({"a", "b"})
    assert FunctionConstraint(lambda s: True).support is None


def test_function_constraint_rejects_non_callable():
    with pytest.raises(TypeError):
        FunctionConstraint("a")


def test_function_constraint_rejects_string_support():
    with pytest.raises(TypeError):
        FunctionConstraint(lambda s: True, support="ab")
