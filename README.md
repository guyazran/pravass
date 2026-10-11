# PRaVaSS: Task-Aware Minimal Belief Sets from Pretrained Perception

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A robot that asks a vision-language model (VLM) "Is the top shelf clear?" gets a
probability, not a fact. Most pipelines round every answer to yes or no and
plan as if the result were the true state. When one answer is wrong, the plan
fails or is unsafe. Tracking a full distribution over symbolic states is the
safe alternative, but it is intractable.

**PRaVaSS** finds the middle ground. From per-feature probabilities and task
constraints, it computes a **θ-confidence minimal belief set (MBS_θ)**: the
smallest set of task-valid states whose total probability reaches θ. A planner
can then plan for every state in the set. The search explores states lazily,
most probable first, instead of enumerating the state space, and it provably
returns the most probable valid states.

This package implements the method from *Robust Task-Aware State Estimation
from Pretrained Perception* (Azran and Keren, NeurIPS 2026). The paper's
experiments can be reproduced with [`experiments/`](experiments/).

## Installation

```bash
pip install git+https://github.com/guyazran/pravass.git
```

PRaVaSS is pure Python (3.10+) with no dependencies. To estimate states from
images, also install [s3e](https://github.com/CLAIR-LAB-TECHNION/s3e):

```bash
pip install "s3e[pddl,hf,calibration]"   # add "openai" or "vllm" for those backends
```

## From images to belief sets with s3e

[s3e](https://github.com/CLAIR-LAB-TECHNION/s3e) is a library for semantic
symbolic state estimation. It asks a VLM one question per grounded PDDL
predicate and reads the probability of the answer, which is exactly the
*factored belief* PRaVaSS starts from. Together they turn camera images into a
belief set for a planner.

```python
from PIL import Image
from s3e import SemanticStateEstimator, TemplateTranslator
from pravass import MutexGroups, TaskBelief, UnconstrainedBelief, pravass

domain = """
(define (domain blocksworld)
  (:requirements :typing)
  (:types block)
  (:predicates (on ?x - block ?y - block) (clear ?x - block)))
"""
problem = """
(define (problem three-blocks)
  (:domain blocksworld)
  (:objects a b c - block)
  (:init) (:goal (and)))
"""

# 1. Perception: the probability that each state feature is true (s3e).
estimator = SemanticStateEstimator.from_pddl(
    domain, problem,
    vlm="HuggingFaceTB/SmolVLM-256M-Instruct",  # tiny, for trying it out; the paper used GPT-4.1 and Qwen3-VL
    translator=TemplateTranslator({"on": "Is block {0} on block {1}?", "clear": "Is block {0} clear?"}),
)
results = estimator.estimate([Image.open("scene.png")])  # optionally calibrator=... (s3e Platt scaling)

# 2. Task constraints: a block sits on at most one block, and holds up at most one
#    block, in which case it is not clear.
blocks = ["a", "b", "c"]
constraint = MutexGroups(
    [[f"on({x},{y})" for y in blocks] for x in blocks]
    + [[f"on({x},{y})" for x in blocks] + [f"clear({y})"] for y in blocks]
)

# 3. The task belief and its minimal belief set.
belief = TaskBelief(UnconstrainedBelief(results.probabilities()), constraint)
belief_set = pravass(belief, theta=0.9)

# 4. Hand every state in the set to the planner.
up_states = [estimator.to_up_state({f: f in s for f in belief.features}) for s in belief_set]
```

The paper calibrates the VLM's probabilities with s3e's Platt scaling and
extracts the mutex groups from the PDDL domain with Fast Downward; see
[`experiments/`](experiments/) for both.

If you estimate only some of a problem's features (`estimator.estimate(images,
predicates=subset)`), the features outside the subset are *unknown*, yet
Unified Planning treats any feature missing from a state as false. Add their
known values to the dictionary before calling `to_up_state`.

## Without images

The factored belief can come from any source:

```python
from pravass import MutexGroups, TaskBelief, UnconstrainedBelief, pravass

# Paper, App. A: three features, the first two mutually exclusive.
unconstrained = UnconstrainedBelief({"f1": 0.4, "f2": 0.7, "f3": 0.9})
belief = TaskBelief(unconstrained, MutexGroups([["f1", "f2"]]))

belief.normalization                    # Normalization(value=0.72, exact=True)
belief(frozenset({"f2", "f3"}))         # 0.525: the task belief of one state
sorted(sorted(s) for s in pravass(belief, theta=0.8))
# [['f1', 'f3'], ['f2', 'f3'], ['f3']]
```

A state is the `frozenset` of features that are true in it. A belief set is a
`frozenset` of states. Its probability is
`sum(belief(s) for s in belief_set)`.

`iter_pravass` runs the same search lazily and yields the valid states one at
a time, most probable first. Use it to stop early (for example at the largest
prefix a planner can handle, as the paper's AMS does), to bound the search
time, or to see how much search a result took:

```python
from pravass import TimeLimitError, iter_pravass

search = iter_pravass(belief, theta=0.8, time_limit=1.0)
ranked = []
try:
    for state in search:
        ranked.append(state)  # most probable first
except TimeLimitError:
    pass  # the states found so far are still the most probable ones
search.mass, search.explored  # certified probability of `ranked`, states inserted into the search
```

## Concepts

The names follow the paper.

| Paper | Code | Meaning |
|---|---|---|
| State features `F`, state `s ∈ 2^F` | `State` (`frozenset[str]`) | The features that are true in a state |
| Factored belief `β_F` (Def. 1) | `FactoredBelief` (`Mapping[str, float]`) | P(feature is true), per feature. With s3e: `results.probabilities()` |
| Unconstrained belief `β̃` (Eq. 1) | `UnconstrainedBelief` | Joint belief assuming the features are independent |
| Constraints `C : 2^F → {0,1}` | `Constraint`, `FunctionConstraint`, `MutexGroups` | Which states are valid for the task |
| Task belief `β_C` (Eq. 2) | `TaskBelief` | A belief conditioned on the constraints |
| Normalizer `Z` | `Normalizer`, `Normalization` | `Z = P(state satisfies C)`, the denominator of Eq. 2 |
| Relaxed normalizer (Sec. 2.3) | `UnitNormalizer`, inexact `MutexNormalizer` | An upper bound on `Z` |
| `MBS_θ` (Def. 2) | `BeliefSet` returned by `pravass` | Smallest set of most probable valid states with mass ≥ θ |
| PRaVaSS (Alg. 1) | `pravass()` | Best-first search that computes `MBS_θ` |
| | `iter_pravass()` | The same search, yielding valid states most probable first |
| | `iter_pravass()` | The same search, yielding valid states most probable first |

Any function of the state can be a constraint:

```python
from pravass import FunctionConstraint

one_hand = FunctionConstraint(
    lambda s: not {"holding(cup)", "holding(plate)"} <= s,
    support={"holding(cup)", "holding(plate)"},  # optional: the features it reads
)
```

To combine several constraints, write one function that checks all of them,
or list all the groups in a single `MutexGroups`.

## Choosing a normalizer

The task belief divides by the normalizer `Z`. This is the normalizing constant
of constraint conditioning, not the evidence term of a Bayesian filter update.
The beliefs here describe a single observation and are never updated.

| Normalizer | Cost | Exact? | Use when |
|---|---|---|---|
| `EnumerationNormalizer` (default) | `2^|support|` states, or `2^|F|` without a declared support | Yes | The constraint reads few features, or `F` is small (limit: `max_states=2**20`) |
| `MutexNormalizer` | `O(|F|)` | When the groups are disjoint | `MutexGroups` constraints, e.g. from a planner's invariants |
| `UnitNormalizer` | `O(1)` | No (`Z = 1`) | Nothing else applies |

```python
from pravass import MutexNormalizer

belief = TaskBelief(unconstrained, MutexGroups([["f1", "f2"]]), normalizer=MutexNormalizer())
belief.normalization.exact  # True here; if False, probabilities are lower bounds
```

An inexact normalizer is an upper bound on `Z`, so it is always safe. The
result still holds at least θ of the probability, with the most probable
states first, but it can be **much larger than minimal**. Prefer an exact
normalizer when one is affordable. For overlapping mutex groups from a
planner, pass the planner's disjoint cover
(`MutexNormalizer(selected_groups=...)`) to tighten the bound.

## Guarantees and limits

- **Exactness.** With an exact normalizer, `pravass` returns a minimal belief
  set: no smaller set of valid states reaches θ (paper Theorem 1 and
  Corollary 1). If states tie in probability, the minimal set is not unique,
  and ties are broken deterministically. Mass comparisons use an absolute
  tolerance of `TOLERANCE = 1e-9`.
- **Mass, not coverage.** θ bounds the probability that the belief attributes
  to the set. It equals the chance that the set contains the true state only
  insofar as the perception model is well calibrated, which is why the paper
  calibrates its VLMs first.
- **Independence.** The unconstrained belief treats features as independent
  given the observation. The constraints restore logical dependencies, but
  perceptual correlations stay unmodeled. PRaVaSS's ordering argument relies
  on independence, so it does not extend to beliefs with dependencies.
- **Cost.** With no constraints, PRaVaSS runs in time polynomial in the
  number of features and the size of the result (Prop. 1). With constraints,
  the worst case is exponential.
  For example, a VLM that is confidently wrong about a mutex group makes the
  most likely state invalid, and the search must pass many invalid states.
  `pravass` stops with `StateLimitError` once it has inserted `max_states`
  states into its search (default `2**20`). Each explored state inserts up to
  one neighbour per feature, so with hundreds of features this bounds the
  search to a few thousand explored states. Pass a larger value, or
  `max_states=None` for no limit.
- **Unanswered features.** s3e assigns probability 0.5 to queries the model
  did not answer. *k* such features make 2^k states equally likely, and the
  belief set grows accordingly.
- **Hard evidence.** Probabilities of exactly 0 or 1 rule states out. If they
  contradict the constraints, `InfeasibleConstraintError` is raised. To soften
  them, clamp first, e.g. `{f: min(max(p, 1e-6), 1 - 1e-6) for f, p in probs.items()}`.
  Probabilities far below `1e-12` gain nothing over clamping, and products
  beyond the floating-point range raise `ValueError`.

## Reproducing the paper

[`experiments/`](experiments/) contains the code, environment, and step-by-step
instructions to reproduce every result in the paper, including those added
during the review. Every belief set there is computed by this package. The
directory is not part of the installed package.

## Citation

If you use PRaVaSS, please cite:

```bibtex
@inproceedings{azran2026robust,
  title     = {Robust Task-Aware State Estimation from Pretrained Perception},
  author    = {Azran, Guy and Keren, Sarah},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {To appear}
}
```

If you use s3e for perception, please also cite s3e (see its
[README](https://github.com/CLAIR-LAB-TECHNION/s3e#citation)).

## License

[MIT](LICENSE)
