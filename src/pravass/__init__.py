"""PRaVaSS: task-aware minimal belief sets from pretrained perception.

Turn per-feature probabilities from a perception model (the factored belief)
into a small set of likely, task-valid symbolic states (a θ-confidence minimal
belief set) for a downstream planner.
"""

from importlib.metadata import PackageNotFoundError, version

from ._validation import DEFAULT_MAX_STATES
from .base import Belief, BeliefSet, InfeasibleConstraintError, State, StateLimitError, TimeLimitError
from .constraints import Constraint, FunctionConstraint
from .unconstrained import FactoredBelief, UnconstrainedBelief

try:
    __version__ = version("pravass")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "State",
    "BeliefSet",
    "FactoredBelief",
    "Belief",
    "UnconstrainedBelief",
    "Constraint",
    "FunctionConstraint",
    "DEFAULT_MAX_STATES",
    "InfeasibleConstraintError",
    "StateLimitError",
    "TimeLimitError",
]
