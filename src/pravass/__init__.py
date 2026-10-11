"""PRaVaSS: task-aware minimal belief sets from pretrained perception.

Turn per-feature probabilities from a perception model (the factored belief)
into a small set of likely, task-valid symbolic states (a θ-confidence minimal
belief set) for a downstream planner.

Quick start::

    from pravass import MutexGroups, TaskBelief, UnconstrainedBelief, pravass

    unconstrained = UnconstrainedBelief({"in(milk,fridge)": 0.6, "on(milk,counter)": 0.5})
    belief = TaskBelief(unconstrained, MutexGroups([["in(milk,fridge)", "on(milk,counter)"]]))
    belief_set = pravass(belief, theta=0.9)
"""

from importlib.metadata import PackageNotFoundError, version

from ._validation import DEFAULT_MAX_STATES
from .base import Belief, BeliefSet, InfeasibleConstraintError, State, StateLimitError, TimeLimitError
from .constraints import Constraint, FunctionConstraint
from .mutex import MutexGroups, MutexNormalizer
from .normalizers import EnumerationNormalizer, Normalization, Normalizer, UnitNormalizer
from .search import TOLERANCE, PravassIterator, iter_pravass, pravass
from .task import TaskBelief
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
    "TaskBelief",
    "Constraint",
    "FunctionConstraint",
    "MutexGroups",
    "Normalization",
    "Normalizer",
    "EnumerationNormalizer",
    "UnitNormalizer",
    "MutexNormalizer",
    "pravass",
    "iter_pravass",
    "PravassIterator",
    "TOLERANCE",
    "DEFAULT_MAX_STATES",
    "InfeasibleConstraintError",
    "StateLimitError",
    "TimeLimitError",
]
