"""Experiment 7: correlated perception errors versus the conditional-independence lift.

The belief converter lifts per-feature marginals to a state distribution by assuming
conditional independence.  This module builds a *correlated* ground-truth joint whose
per-feature marginals are exactly the ones the estimator consumes, so the only
discrepancy between the two distributions is the independence assumption itself.

The correlated joint is a two-component mixture driven by a shared latent nuisance
event ``E`` (the "one lighting condition corrupts many features" failure mode)::

    P_true(s) = (1 - pi) * prod_i p0_i(s_i) + pi * prod_i p1_i(s_i)

``p0`` are the calibrated marginals as measured; ``p1`` equals ``p0`` outside the
corrupted block and is pushed toward the opposite value inside it.  A mixture of two
products is genuinely correlated, yet every quantity below stays closed-form: masses,
normalizers, and total variation are exact, and the top-prefix search over the mixture
carries a certificate rather than a truncation.

Because the estimator is handed the exact mixture marginals
``p_i = (1 - pi) p0_i + pi p1_i``, it is *perfectly feature-calibrated by construction*.
Any resulting error is attributable to conditional independence alone.
"""
from __future__ import annotations

import itertools
import math
import random
import time
from collections.abc import Mapping, Sequence
from statistics import mean
from dataclasses import dataclass
from typing import Any

from mbs_experiments.search import iter_ranked_states
from mbs_experiments.states import (
    State,
    clamp_probability,
    is_valid_state,
    selected_groups_normalizer,
    unconstrained_probability,
)

PROTOCOL_VERSION = "exp7-correlated-noise-v1"

# How the corrupted feature block is chosen.  ``constraint`` is the default because a
# mutex group names features about the same object, which is the block a shared visual
# nuisance would corrupt together -- and it is the adversarial choice for us, since the
# constraint layer is exactly what is supposed to re-couple those features.
GROUP_MODES = ("constraint", "uncertain", "random")

# Probabilities are clamped away from {0, 1} so that a corrupted marginal never makes a
# state impossible under one mixture component and thereby silently drops it.
PROB_EPS = 1e-6

# Guard for the exact total-variation block enumeration.  A block larger than this is
# reported as ``None`` rather than silently truncated.
MAX_TV_BLOCK_FEATURES = 16

MASS_EPSILON = 1e-12


@dataclass(frozen=True)
class CorruptionModel:
    """A two-component correlated observation model over one sample's features.

    ``marginals`` is what the estimator sees; ``clean``/``corrupt`` define the joint it
    does not see.  ``rate`` is the nuisance probability pi, ``strength`` the pull rho.
    """

    corrupted_indices: tuple[int, ...]
    rate: float
    strength: float
    clean: tuple[float, ...]
    corrupt: tuple[float, ...]
    marginals: tuple[float, ...]

    @property
    def weights(self) -> tuple[float, float]:
        return 1.0 - self.rate, self.rate


@dataclass(frozen=True)
class JointPrefix:
    """Minimal prefix of the true mixture reaching ``theta``, with a search certificate.

    ``certified`` is True only when every state left unexamined provably has a lower
    mixture density than the least likely selected state, i.e. the prefix is exactly the
    highest-density set and not an artifact of stopping early.
    """

    states: tuple[State, ...]
    mass: float
    certified: bool
    candidates_examined: int
    runtime_seconds: float


def _clamp(q: float, *, eps: float = PROB_EPS) -> float:
    return min(1.0 - eps, max(eps, clamp_probability(float(q))))


def uncertainty(q: float) -> float:
    """Bernoulli uncertainty min(q, 1-q); higher means the feature is less decided."""
    return min(float(q), 1.0 - float(q))


def select_corrupted_indices(
    probabilities: Sequence[float],
    selected_groups: Sequence[Sequence[int]],
    *,
    mode: str,
    size: int,
    seed: int,
) -> tuple[int, ...]:
    """Choose the feature block a shared nuisance event corrupts together.

    ``constraint`` walks mutex groups from largest to smallest, so the block is made of
    features describing the same objects.  ``uncertain`` takes the least decided
    features, modelling a nuisance that strikes wherever perception is already weak.
    ``random`` is the seeded control.
    """
    if mode not in GROUP_MODES:
        raise ValueError(f"Unknown group mode {mode!r}; expected one of {GROUP_MODES}")
    n = len(probabilities)
    if n == 0 or size <= 0:
        return ()
    budget = min(size, n)

    if mode == "constraint":
        chosen: list[int] = []
        for group in sorted(selected_groups, key=lambda g: (-len(g), tuple(g))):
            for idx in group:
                if idx not in chosen:
                    chosen.append(idx)
                if len(chosen) >= budget:
                    return tuple(sorted(chosen))
        # Fewer grouped features than requested: top up with the least decided ones so
        # the block size stays comparable across samples and modes.
        remaining = sorted(
            (i for i in range(n) if i not in chosen),
            key=lambda i: (-uncertainty(probabilities[i]), i),
        )
        chosen.extend(remaining[: budget - len(chosen)])
        return tuple(sorted(chosen))

    if mode == "uncertain":
        order = sorted(range(n), key=lambda i: (-uncertainty(probabilities[i]), i))
        return tuple(sorted(order[:budget]))

    rng = random.Random(seed)
    return tuple(sorted(rng.sample(range(n), budget)))


def build_corruption_model(
    probabilities: Sequence[float],
    corrupted_indices: Sequence[int],
    *,
    rate: float,
    strength: float,
) -> CorruptionModel:
    """Build the mixture whose marginals are exactly ``probabilities``.

    Inside the corrupted block the nuisance pulls each marginal toward its opposite:
    ``p1_i = q_i + rho * (1 - 2 q_i)``, so ``rho = 1`` mirrors the marginal and
    ``rho = 0`` is a no-op.  The pull is defined against the model's own estimate, never
    against the label, so no ground truth leaks into the belief.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate outside [0, 1]: {rate}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength outside [0, 1]: {strength}")
    clean = tuple(_clamp(q) for q in probabilities)
    block = set(int(i) for i in corrupted_indices)
    corrupt = tuple(
        _clamp(q + strength * (1.0 - 2.0 * q)) if i in block else q
        for i, q in enumerate(clean)
    )
    marginals = tuple(
        (1.0 - rate) * c + rate * d for c, d in zip(clean, corrupt)
    )
    return CorruptionModel(
        corrupted_indices=tuple(sorted(block)),
        rate=float(rate),
        strength=float(strength),
        clean=clean,
        corrupt=corrupt,
        marginals=marginals,
    )


def component_normalizers(
    model: CorruptionModel, selected_groups: Sequence[Sequence[int]]
) -> tuple[float, float]:
    """Validity mass of each mixture component, in closed form."""
    return (
        selected_groups_normalizer(model.clean, selected_groups),
        selected_groups_normalizer(model.corrupt, selected_groups),
    )


def true_normalizer(
    model: CorruptionModel, selected_groups: Sequence[Sequence[int]]
) -> float:
    """Validity mass of the mixture: a mixture of normalizers, exactly."""
    w0, w1 = model.weights
    z0, z1 = component_normalizers(model, selected_groups)
    return w0 * z0 + w1 * z1


def true_state_mass(
    states: Sequence[State],
    model: CorruptionModel,
    selected_groups: Sequence[Sequence[int]],
) -> float:
    """Exact mass the correlated joint assigns to ``states``, conditioned on validity.

    Because the true state is drawn from this joint, the returned value *is* the exact
    probability that the set contains the true state -- expected coverage without any
    Monte-Carlo error.
    """
    w0, w1 = model.weights
    denominator = true_normalizer(model, selected_groups)
    if denominator <= 0.0:
        return float("nan")
    numerator = 0.0
    for state in states:
        numerator += w0 * unconstrained_probability(state, model.clean)
        numerator += w1 * unconstrained_probability(state, model.corrupt)
    return numerator / denominator


def affected_block(
    corrupted_indices: Sequence[int],
    selected_groups: Sequence[Sequence[int]],
    feature_count: int,
) -> tuple[int, ...]:
    """Smallest union of whole mutex groups (plus loose features) covering the block.

    Outside this block the two distributions are the same product, and with disjoint
    at-most-one groups both factorize across it.  So restricting to the block loses
    nothing: the total variation over the block equals the total variation overall.
    """
    block = set(int(i) for i in corrupted_indices)
    changed = True
    while changed:
        changed = False
        for group in selected_groups:
            members = set(int(i) for i in group)
            if members & block and not members <= block:
                block |= members
                changed = True
    return tuple(sorted(i for i in block if 0 <= i < feature_count))


def total_variation(
    model: CorruptionModel,
    selected_groups: Sequence[Sequence[int]],
    *,
    max_block_features: int = MAX_TV_BLOCK_FEATURES,
) -> float | None:
    """Exact total variation between the product belief and the correlated joint.

    Returns ``None`` when the affected block exceeds ``max_block_features``, so a skipped
    sample is visible in the aggregate rather than silently counted as zero distortion.
    """
    block = affected_block(model.corrupted_indices, selected_groups, len(model.clean))
    if not block:
        return 0.0
    if len(block) > max_block_features:
        return None

    position = {idx: pos for pos, idx in enumerate(block)}
    local_groups = [
        tuple(position[i] for i in group)
        for group in selected_groups
        if all(i in position for i in group)
    ]
    w0, w1 = model.weights
    product_total = 0.0
    true_total = 0.0
    product_terms: list[float] = []
    true_terms: list[float] = []
    for assignment in itertools.product((False, True), repeat=len(block)):
        if not is_valid_state(assignment, local_groups):
            continue
        product_value = 1.0
        clean_value = 1.0
        corrupt_value = 1.0
        for pos, value in enumerate(assignment):
            idx = block[pos]
            product_value *= (
                model.marginals[idx] if value else 1.0 - model.marginals[idx]
            )
            clean_value *= model.clean[idx] if value else 1.0 - model.clean[idx]
            corrupt_value *= model.corrupt[idx] if value else 1.0 - model.corrupt[idx]
        true_value = w0 * clean_value + w1 * corrupt_value
        product_terms.append(product_value)
        true_terms.append(true_value)
        product_total += product_value
        true_total += true_value
    if product_total <= 0.0 or true_total <= 0.0:
        return None
    return 0.5 * sum(
        abs(p / product_total - t / true_total)
        for p, t in zip(product_terms, true_terms)
    )


def _mixture_density(state: State, model: CorruptionModel) -> float:
    w0, w1 = model.weights
    return w0 * unconstrained_probability(state, model.clean) + w1 * (
        unconstrained_probability(state, model.corrupt)
    )


def oracle_joint_prefix(
    model: CorruptionModel,
    selected_groups: Sequence[Sequence[int]],
    theta: float,
    *,
    max_candidates: int = 20000,
    timeout_seconds: float | None = 30.0,
    check_every: int = 32,
) -> JointPrefix:
    """Minimum-cardinality set reaching ``theta`` under the *true* correlated joint.

    This is the tractable-joint comparator: what an estimator that knew the correlated
    joint would return.  Neither mixture component alone orders states by mixture
    density, so the search advances both component enumerations and stops on a
    certificate: with ``tail_c`` the density of the last valid state that PRaVaSS (the
    pravass package) yielded for component ``c``, every unexamined valid state has
    mixture density at most ``w0 * tail_0 + w1 * tail_1``.  Once the least likely selected state clears that
    bound, no unseen state can belong in the prefix and the answer is exact.
    """
    start = time.perf_counter()
    w0, w1 = model.weights
    denominator = true_normalizer(model, selected_groups)
    if denominator <= 0.0:
        return JointPrefix((), float("nan"), False, 0, time.perf_counter() - start)

    iterators = (
        iter_ranked_states(model.clean, selected_groups),
        iter_ranked_states(model.corrupt, selected_groups),
    )
    weights = (w0, w1)
    tails = [1.0, 1.0]
    exhausted = [w0 <= 0.0, w1 <= 0.0]
    for component, is_done in enumerate(exhausted):
        if is_done:
            tails[component] = 0.0

    seen: dict[State, float] = {}
    steps = 0

    def certificate() -> JointPrefix | None:
        bound = sum(weights[c] * tails[c] for c in (0, 1))
        ranked = sorted(seen.items(), key=lambda item: -item[1])
        accumulated = 0.0
        selected: list[State] = []
        for state, value in ranked:
            selected.append(state)
            accumulated += value
            if accumulated / denominator + MASS_EPSILON >= theta:
                # Exact only if nothing unseen could outrank the weakest pick.
                if value + MASS_EPSILON >= bound:
                    return JointPrefix(
                        states=tuple(selected),
                        mass=accumulated / denominator,
                        certified=True,
                        candidates_examined=len(seen),
                        runtime_seconds=time.perf_counter() - start,
                    )
                return None
        return None

    while not all(exhausted):
        if timeout_seconds is not None and time.perf_counter() - start >= timeout_seconds:
            break
        if len(seen) >= max_candidates:
            break
        # Advance whichever component still hides the most mixture mass.
        component = max(
            (c for c in (0, 1) if not exhausted[c]),
            key=lambda c: weights[c] * tails[c],
        )
        try:
            state, probability = next(iterators[component])
        except StopIteration:
            exhausted[component] = True
            tails[component] = 0.0
            continue
        # PRaVaSS yields a component's valid states in non-increasing density,
        # so no unseen valid state of that component is denser than this one.
        tails[component] = probability
        if state not in seen:
            seen[state] = _mixture_density(state, model)
        steps += 1
        if steps % check_every == 0:
            result = certificate()
            if result is not None:
                return result

    result = certificate()
    if result is not None:
        return result

    # Uncertified fallback: report the best prefix found so the row stays analysable,
    # flagged so the aggregate can exclude it from the exact-comparison denominator.
    ranked = sorted(seen.items(), key=lambda item: -item[1])
    accumulated = 0.0
    selected: list[State] = []
    for state, value in ranked:
        selected.append(state)
        accumulated += value
        if accumulated / denominator + MASS_EPSILON >= theta:
            break
    return JointPrefix(
        states=tuple(selected),
        mass=accumulated / denominator if denominator > 0 else float("nan"),
        certified=False,
        candidates_examined=len(seen),
        runtime_seconds=time.perf_counter() - start,
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_mean_ci(
    values_by_sample: Mapping[str, float],
    *,
    seed: int = 0,
    repetitions: int = 1000,
) -> dict[str, float | int | None]:
    """Bootstrap sample IDs, preserving all within-sample pairing.

    Deliberately duplicated from ``closed_loop`` rather than imported: that module
    pulls in the contingent planner stack, and Experiment 7 does no planning at all.
    Keeping it here lets the experiment run wherever the belief code runs.  The
    algorithm is identical, so intervals stay comparable across experiments.
    """
    sample_ids = tuple(sorted(values_by_sample))
    if not sample_ids:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    values = tuple(float(values_by_sample[sample_id]) for sample_id in sample_ids)
    rng = random.Random(seed)
    draws = [
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(max(1, repetitions))
    ]
    return {
        "n": len(values),
        "mean": mean(values),
        "ci_low": _percentile(draws, 0.025),
        "ci_high": _percentile(draws, 0.975),
    }


def summarize_gap(
    attributed_mass: float, true_mass: float, theta: float
) -> dict[str, Any]:
    """Directional summary of one operating point.

    ``gap`` is what the belief claims minus what the correlated joint delivers, so a
    positive gap is overconfidence and a negative gap is conservatism.  ``shortfall``
    isolates the only direction that can mislead a downstream planner.
    """
    gap = float(attributed_mass) - float(true_mass)
    return {
        "attributed_mass": float(attributed_mass),
        "true_mass": float(true_mass),
        "mass_coverage_gap": gap,
        "coverage_shortfall": max(0.0, float(theta) - float(true_mass)),
        "meets_theta": bool(float(true_mass) + MASS_EPSILON >= float(theta)),
    }


def sweep_points(
    rates: Sequence[float], strengths: Sequence[float], thetas: Sequence[float]
) -> tuple[tuple[float, float, float], ...]:
    """Deduplicated (rate, strength, theta) grid; rate 0 collapses over strength.

    At ``rate == 0`` the nuisance never fires, so every strength names the same
    uncorrelated model.  Keeping one representative avoids inflating the denominator of
    the aggregate with identical rows.
    """
    points: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for rate in rates:
        for strength in strengths:
            effective = 0.0 if float(rate) == 0.0 else float(strength)
            for theta in thetas:
                key = (float(rate), effective, float(theta))
                if key in seen:
                    continue
                seen.add(key)
                points.append(key)
    return tuple(points)
