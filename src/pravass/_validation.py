"""Input validation shared across the package (private)."""

import math
import numbers
from collections.abc import Iterable

DEFAULT_MAX_STATES = 2**20


def check_feature(feature: object) -> str:
    """Return ``feature`` if it is a non-empty string, else raise."""
    if not isinstance(feature, str):
        raise TypeError(f"feature names must be strings, got {feature!r}")
    if not feature:
        raise ValueError("feature names must be non-empty strings")
    return feature


def check_features(features: Iterable[str], *, what: str) -> frozenset[str]:
    """Return a frozenset of feature names, rejecting a bare string."""
    if isinstance(features, str):
        raise TypeError(
            f"{what} must be an iterable of feature names, not a single string "
            f"{features!r}"
        )
    return frozenset(check_feature(f) for f in features)


def check_probability(value: object, *, what: str) -> float:
    """Return ``value`` as a float in [0, 1]; reject bools and non-reals."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        hint = "; did you pass a boolean state instead of probabilities?" if isinstance(value, bool) else ""
        raise TypeError(f"{what} must be a real number, got {value!r}{hint}")
    value = float(value)
    if not 0.0 <= value <= 1.0:  # also rejects NaN
        raise ValueError(f"{what} must be in [0, 1], got {value!r}")
    return value


def check_theta(theta: object) -> float:
    """Return the mass threshold θ as a float in [0, 1]."""
    return check_probability(theta, what="theta")


def check_max_states(max_states: object) -> float:
    """Return a state limit as a number (``math.inf`` when unlimited).

    Accepts an int >= 1, ``None``, or a float equal to infinity.
    """
    if max_states is None:
        return math.inf
    if isinstance(max_states, bool):
        raise TypeError(f"max_states must be an int, None, or math.inf, got {max_states!r}")
    if isinstance(max_states, numbers.Integral):
        if max_states < 1:
            raise ValueError(f"max_states must be at least 1, got {max_states!r}")
        return int(max_states)
    if isinstance(max_states, numbers.Real) and max_states == math.inf:
        return math.inf
    raise TypeError(
        f"max_states must be an int, None, or math.inf, got {max_states!r} "
        "(pass an int, e.g. 1_000_000 rather than 1e6)"
    )


def check_time_limit(time_limit: object) -> "float | None":
    """Return a time limit in seconds as a positive float, or ``None`` for no limit."""
    if time_limit is None:
        return None
    if isinstance(time_limit, bool) or not isinstance(time_limit, numbers.Real):
        raise TypeError(f"time_limit must be a number of seconds or None, got {time_limit!r}")
    time_limit = float(time_limit)
    if not time_limit > 0.0:  # also rejects NaN
        raise ValueError(f"time_limit must be positive, got {time_limit!r}")
    return time_limit


def check_state(state: object, features: frozenset[str]) -> frozenset[str]:
    """Return ``state`` as a frozenset of features drawn from ``features``."""
    if not isinstance(state, (set, frozenset)):
        hint = (
            " (to convert a dict of truth values, use "
            "frozenset(f for f, v in d.items() if v))"
            if isinstance(state, dict)
            else ""
        )
        raise TypeError(
            "a state must be a set or frozenset of the features that are true, "
            f"got {type(state).__name__}{hint}"
        )
    state = frozenset(state)
    unknown = state - features
    if unknown:
        raise ValueError(f"state contains unknown features: {sorted(map(repr, unknown))}")
    return state
