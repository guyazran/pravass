"""Brute-force reference implementation in exact rational arithmetic (tests only)."""

from fractions import Fraction
from itertools import product


def all_states(features):
    features = sorted(features)
    for values in product((False, True), repeat=len(features)):
        yield frozenset(f for f, v in zip(features, values) if v)


def unconstrained(probabilities, state):
    result = Fraction(1)
    for feature, p in probabilities.items():
        p = Fraction(p)
        result *= p if feature in state else 1 - p
    return result


def exact_z(probabilities, constraint):
    return sum(
        (unconstrained(probabilities, s) for s in all_states(probabilities) if constraint(s)),
        Fraction(0),
    )


def task_probability(probabilities, constraint, state, z=None):
    if not constraint(state):
        return Fraction(0)
    if z is None:
        z = exact_z(probabilities, constraint)
    return unconstrained(probabilities, state) / z


def ranked_valid(probabilities, constraint):
    """Valid states with nonzero probability, most probable first."""
    z = exact_z(probabilities, constraint)
    if z == 0:
        return []
    scored = [
        (task_probability(probabilities, constraint, s, z), s)
        for s in all_states(probabilities)
        if constraint(s)
    ]
    scored = [(p, s) for p, s in scored if p > 0]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def minimal_size(probabilities, constraint, theta):
    """Size of a minimal belief set reaching mass theta (exact arithmetic)."""
    theta = Fraction(theta)
    total = Fraction(0)
    for count, (p, _) in enumerate(ranked_valid(probabilities, constraint), start=1):
        total += p
        if total >= theta:
            return count
    return len(ranked_valid(probabilities, constraint))
