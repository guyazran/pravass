"""Shared fixtures: the paper's App. A worked example."""

import pytest

from pravass import MutexGroups, UnconstrainedBelief

# Paper App. A: factored belief [0.4, 0.7, 0.9], the first two features mutually
# exclusive; the normalizer is Z = 0.72.
APP_A_FACTORED = {"f1": 0.4, "f2": 0.7, "f3": 0.9}
APP_A_GROUPS = [["f1", "f2"]]


@pytest.fixture
def app_a_unconstrained():
    return UnconstrainedBelief(APP_A_FACTORED)


@pytest.fixture
def app_a_constraint():
    return MutexGroups(APP_A_GROUPS)
