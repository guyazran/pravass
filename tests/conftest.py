"""Shared fixtures: the paper's App. A worked example."""

import pytest

from pravass import UnconstrainedBelief

# Paper App. A: factored belief [0.4, 0.7, 0.9], the first two features mutually
# exclusive; the normalizer is Z = 0.72.
APP_A_FACTORED = {"f1": 0.4, "f2": 0.7, "f3": 0.9}


@pytest.fixture
def app_a_unconstrained():
    return UnconstrainedBelief(APP_A_FACTORED)
