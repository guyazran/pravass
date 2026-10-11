# Reproducing the paper's experiments

This directory reproduces the experimental results of *Robust Task-Aware State
Estimation from Pretrained Perception* (Azran and Keren, NeurIPS 2026),
including the results added during the review. It is a self-contained project
with its own pinned environment. It is not part of the installed `pravass`
package.

Every belief set in the experiments is computed by the `pravass` package in
the parent directory, which `uv sync` installs from there.
`mbs_experiments/search.py` translates the experiments' inputs (states as
tuples of booleans, constraints as groups of feature indices) to the package's
types. The constraints, normalizers and baselines are in `mbs_experiments/`.

## Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Step 1: Data](#step-1-data)
5. [Step 2: VLM predictions](#step-2-vlm-predictions) (download, or run the VLMs)
6. [Step 3: Calibration](#step-3-calibration)
7. [Step 4: Experiments](#step-4-experiments)
8. [Step 5: Figures and tables](#step-5-figures-and-tables)
9. [Where each result comes from](#where-each-result-comes-from)
10. [Notes on the protocol](#notes-on-the-protocol)
11. [Troubleshooting](#troubleshooting)

## Overview

```
raw_data.zip (OSF)
  └─ split_data.py ─ relevant_atoms.py ─ extract_constraints.py      splits, ground truth, task constraints
        └─ make_predictions.py      [download, or GPU / OpenAI API]  VLM probability per state feature
              └─ calibrate_vlm.py ─ calibrate_tau.py                 Platt scaling, abstention thresholds
                    └─ scripts/run_experiment_{1,2,3,6,7}_*.py       experiments
                          └─ results.ipynb                           figures/ and tables
```

| Experiment | Results | Script | Needs |
|---|---|---|---|
| 1. Belief-set quality: coverage vs. size | Fig. 5, Fig. 10, Table 3; six backbones, reliability view | `run_experiment_1_static_quality.py` | CPU |
| 2. PRaVaSS scaling vs. exhaustive search | Table 4, Figs. 6 and 11; explored-state counts | `run_experiment_2_prevas_scaling.py` | CPU |
| 3. Planning with belief sets (AMS) | Tables 1, 5 and 6, Figs. 7, 12 and 13; six backbones, constraint ablation | `run_experiment_3_planning_utility.py` | CPU, CPOR planner |
| 6. Closed-loop replanning (ALFWorld) | closed-loop study (review addition) | `run_experiment_6_controlled_prior.py` | CPU, CPOR planner |
| 7. Correlated perceptual noise | correlated-noise study (review addition) | `run_experiment_7_correlated_noise.py` | CPU |

Experiments 4 and 5 were pilots that the paper does not use, so they are not
included. The other experiments keep their original numbering.

Experiments 3 and 6 plan with the [CPOR planner for Unified Planning](https://github.com/aiplan4eu/up-cpor)
(Apache-2.0), in the version used for the paper: the branch
[`pravass-nips-reproduce`](https://github.com/guyazran/up-cpor/tree/pravass-nips-reproduce)
of the fork [guyazran/up-cpor](https://github.com/guyazran/up-cpor) (commit
`6215474`), which includes the prebuilt planner. Setup (below) clones it into
`up-cpor/`, which git ignores, and `uv sync` installs it from there.

The directory holds the pipeline scripts (Steps 1–3), `scripts/` (experiments,
notebook builder, checks), `mbs_experiments/` (the experiment code),
`prompts/` and `nl_templates/` (the VLM prompts and query templates),
`conditional_domains/` (PDDL domains for planning), and `tests/`.

## Requirements

- **Linux x86-64.** The CPOR contingent planner runs on .NET/Mono.
- **[conda](https://github.com/conda-forge/miniforge)** for Mono and .NET, and
  **[uv](https://docs.astral.sh/uv/)** for the pinned Python 3.12 environment.
- **Disk:** about 5 GB for data and predictions, and about 6 GB for the Python
  environment. The experiment outputs are larger: Experiment 1 writes 2–8 GB
  per model.
- **VLM inference** (Step 2), only if you do not download the predictions:
  - GPT-4.1 through the OpenAI API;
  - GPUs for the five open models. The 31B and 32B models need about 80 GB of
    GPU memory.
  - Each model answers about 110,000 single-token queries over 5,240 images.
- **CPU for everything else.** The experiments parallelize over `--jobs` cores.
  The planning experiments budget about 15 GB of memory per worker. The paper's
  runs took up to 10 hours each on a 1 TB-RAM cluster node.

## Setup

```bash
git clone https://github.com/guyazran/pravass.git
cd pravass/experiments

conda env create -f environment.yml      # Mono, .NET and uv
conda activate pravass-experiments
git clone --branch pravass-nips-reproduce https://github.com/guyazran/up-cpor.git up-cpor   # the CPOR planner
uv sync --frozen                         # the exact Python environment from uv.lock (~10 min)
uv run pytest -q                         # optional: the experiment code's tests (~1 min)
```

Run every command below from this directory with the conda environment
active. `uv run` uses the pinned environment. This includes s3e 0.2.0: the
pipeline uses its 0.2 API, which later s3e releases changed.

## Step 1: Data

Download the dataset (about 1 GB) from [OSF](https://osf.io/64huf/overview?view_only=22fbd6c6ad144b4b8d9c499e3f3876ab)
and check it:

```bash
curl -fL --retry 8 --retry-delay 60 --retry-all-errors -o raw_data.zip \
  "https://osf.io/download/vadyc/?view_only=22fbd6c6ad144b4b8d9c499e3f3876ab"
sha256sum raw_data.zip
# 6b2ba02e7a9e02afde8f824278f24032f1bbd3a1861a96c95951f4309fc42a29  raw_data.zip
```

It holds one directory per sample, with the PDDL domain and problem, the true
state, and the rendered image(s). There are three domains:
- Photorealistic Blocksworld (`prb`);
- Groceries World (`gw`);
- ALFWorld (`alf`).

Then build the splits, the task-relevant ground truth, and the task constraints:

```bash
uv run split_data.py                                  # extracts raw_data.zip; deterministic splits (seed 0)
uv run relevant_atoms.py --domains prb gw             # task-relevant state features
uv run relevant_atoms.py --domains alf --visible-only # ALFWorld: only features of visible objects
uv run extract_constraints.py                         # mutex groups via Fast Downward; lifted rules for ALF (App. G)
```

`split_data.py` writes two splits, as in App. F of the paper:
- `test`: PRB 1200, GW 600, ALF 2000 instances; every experiment runs on these;
- `validation`: PRB 240, GW 200, ALF 1000 instances; Step 3 fits the Platt
  scaling of the VLM probabilities and the Feature Abstention threshold on it.

## Step 2: VLM predictions

Choose one of two paths.

### Option A: download the predictions (CPU only)

Download `predictions.zip` (about 210 MB) from the files of the same
[OSF project](https://osf.io/64huf/overview?view_only=22fbd6c6ad144b4b8d9c499e3f3876ab)
into this directory, then:

```bash
sha256sum predictions.zip
# fb208795a72b11d1da24bb7f65f6fb694b835608929704f81e9acf1365dbdd06  predictions.zip
unzip -q predictions.zip                 # creates prediction_data/
uv run scripts/verify_predictions.py     # checks every file against the archive's manifest
```

It holds the six models' answers on both splits.

### Option B: run the VLMs

Put your OpenAI key in a file named `.env` (`OPENAI_API_KEY=...`). The open
models download from Hugging Face on first use. The Gemma models are gated:
accept their license on Hugging Face and log in (`uv run hf auth login`, or
set `HF_TOKEN`) before running them.

```bash
# GPT-4.1 (OpenAI API)
uv run --env-file .env make_predictions.py --model OpenAI/gpt-4.1-2025-04-14 --domains prb gw alf --batch-size 100

# Open models (one GPU job each)
uv run make_predictions.py --model Qwen/Qwen3-VL-8B-Instruct  --domains prb gw alf --batch-size 1
uv run make_predictions.py --model Qwen/Qwen3-VL-32B-Instruct --domains prb gw alf --batch-size 1
uv run make_predictions.py --model google/gemma-4-E2B-it      --domains prb gw alf --batch-size 16
uv run make_predictions.py --model google/gemma-4-E4B-it      --domains prb gw alf --batch-size 16
uv run make_predictions.py --model google/gemma-4-31B-it      --domains prb gw alf --batch-size 4
```

Run the Qwen models with batch size 1. With batching, s3e 0.2.0 reads the
answer probabilities of Qwen3-VL from the wrong position for some queries:
those answers put almost no probability on any answer token ("yes", "no" or
"null"), so their normalized probabilities are noise. The published Qwen
predictions were made with batching and have this defect. On the test split,
the share of answers with less than 1% probability on every answer token is
67% for Qwen3-VL-8B on PRB, and 54% (PRB), 43% (GW) and 11% (ALF) for
Qwen3-VL-32B. Gemma-4-E2B has the same symptom on 15% of its ALF answers; every
other model and domain is below 1%. The Qwen results therefore understate what
these models can do, and Option A is the only way to reproduce them.
`uv run scripts/check_answer_mass.py` prints these shares for any predictions.

Predictions go to `prediction_data/<split>/<model>/`. The script resumes where
it stopped and skips existing files. Decoding is deterministic (temperature 0,
greedy), but the open models run in bfloat16. Their probabilities shift with
the GPU type and the batch size, so individual answers can change. Option A
gives exactly the inputs used for the paper.

## Step 3: Calibration

Both calibration steps use only the validation split:

- `calibrate_vlm.py` fits Platt scaling (paper, Sec. 4): a logistic
  recalibration of each model's per-feature log-odds, one per domain, written
  to `calib_data/<model>/<domain>/`. Every method in every experiment reads
  these calibrated probabilities.
- `calibrate_tau.py` picks the threshold τ of the Feature Abstention
  baseline (written to `abstention_data/<model>/`): for each θ, the smallest τ
  whose belief sets have a mean probability mass of at least θ (paper,
  Experiment 1). It reads the calibrated probabilities, so run it second.

```bash
MODELS="OpenAI/gpt-4.1-2025-04-14 Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-32B-Instruct google/gemma-4-E2B-it google/gemma-4-E4B-it google/gemma-4-31B-it"

for MODEL in $MODELS; do
  uv run calibrate_vlm.py --model "$MODEL" --domains prb gw alf --calibration-mode global
  uv run calibrate_tau.py --model "$MODEL" --domains prb gw alf --calibrations global --jobs 8
done
```

## Step 4: Experiments

The experiments read the cached predictions; none calls a VLM. Set `--jobs` to
at most your number of physical cores, and lower it if memory runs short
(Experiments 3 and 6 budget up to 15 GB per worker). Every run is resumable:
records that already exist are skipped.

```bash
export MBS_PLANNER_MAX_GIB=15   # memory cap per CPOR planning call (Experiments 3 and 6)
MODELS="OpenAI/gpt-4.1-2025-04-14 Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-32B-Instruct google/gemma-4-E2B-it google/gemma-4-E4B-it google/gemma-4-31B-it"

# Experiment 1: belief-set quality (all six backbones)
for MODEL in $MODELS; do
  uv run scripts/run_experiment_1_static_quality.py --model "$MODEL" --calibrations global --domains prb gw alf --jobs 8
done

# Experiment 2: PRaVaSS scaling (GPT-4.1; 10 s per instance, as in the paper)
uv run scripts/run_experiment_2_prevas_scaling.py --model OpenAI/gpt-4.1-2025-04-14 --calibrations global --domains prb gw alf --jobs 8

# Experiment 3: planning with belief sets (all six backbones)
for MODEL in $MODELS; do
  uv run scripts/run_experiment_3_planning_utility.py --model "$MODEL" --calibrations global --domains prb gw --jobs 8
done

# Experiment 6: closed-loop replanning on ALFWorld (GPT-4.1)
#   6a. the 10 tasks of an earlier pilot, which the main run excludes (only the task list is needed)
uv run scripts/run_experiment_6_controlled_prior.py --prior-mode uniform_hidden --corruption-counts 0 \
  --run-name uniform_hidden_pilot_10 --limit 10 --sample-seed 0 --query-seed 0 --prepare-only
#   6b. main run: the next 120 tasks
uv run scripts/run_experiment_6_controlled_prior.py --prior-mode uniform_hidden --corruption-counts 0 \
  --run-name uniform_hidden_main_n120_adaptive_v3 --limit 120 --sample-seed 0 --query-seed 0 \
  --exclude-samples-from experiments_data/experiment_6_controlled_prior/OpenAI__gpt-4.1-2025-04-14/uniform_hidden_pilot_10/selected_samples.txt \
  --abstention-calibration-id adaptive-count-map-fallback-v1 --planning-timeout 30 --method-timeout 30 --jobs 6 --freeze-protocol

# Experiment 7: correlated perceptual noise (GPT-4.1)
uv run scripts/run_experiment_7_correlated_noise.py --limit 200 --run-name main --jobs 8 \
  --rates 0.0 0.1 0.2 0.3 0.5 --strengths 0.5 1.0 --thetas 0.5 0.9 0.99 --group-modes constraint uncertain
```

Outputs go to `experiments_data/experiment_<n>_*/<model>/` (for Experiments 6
and 7, one more level: `<run-name>/`). Experiment 6 runs only in the
`uniform_hidden` mode shown; its other modes and options are not used in the
paper.

## Step 5: Figures and tables

```bash
uv run jupyter nbconvert --to notebook --execute --inplace results.ipynb
```

The notebook renders every figure and table, prints the tables, and saves them to
`figures/` (PDF and CSV). Its first part reproduces the submitted paper, with the
functions and settings that produced it; its second part holds the results added
during the review. The first run streams the experiment outputs once and caches
small summaries next to them. The notebook is generated by
`scripts/build_results_notebook.py`: to change it, edit the builder, run
`uv run scripts/build_results_notebook.py`, then execute the notebook again.

## Where each result comes from

Sections refer to `results.ipynb`; files are written to `figures/`.

**Paper** (GPT-4.1)

| Result | Experiment | Notebook section | Output |
|---|---|---|---|
| Table 3: coverage, set size and mass per θ | 1 | Experiment 1 › Table 3 | `paper-table3-exp1.csv` |
| Fig. 5: coverage vs. set size | 1 | Experiment 1 › Fig. 5 | `paper-fig5-coverage-size.pdf` |
| Fig. 10: set size vs. mass and coverage | 1 | Experiment 1 › Fig. 10 | `paper-fig10-size-mass-coverage.pdf` |
| Random θ-Mass on GW; set-size ratios at θ = 0.9 | 1 | Experiment 1 › Numbers quoted in the text | printed |
| Table 4: scaling of PRaVaSS vs. exhaustive search | 2 | Experiment 2 › Table 4 | `paper-table4-exp2.csv` |
| 154% / 42% more features than exhaustive search | 2 | Experiment 2 › How many more features | printed |
| Fig. 6: instances solved within a time budget | 2 | Experiment 2 › Fig. 6 | `paper-fig6-cactus.pdf` |
| Fig. 11: runtime and explored states by number of features | 2 | Experiment 2 › Fig. 11 | `paper-fig11a-runtime-by-features.pdf`, `paper-fig11b-states-by-features.pdf` |
| Tables 1 and 5: planning success; relative improvements | 3 | Experiment 3 › Tables 1 and 5 | `paper-table5-exp3-success.csv` |
| Fig. 7: AMS confidence vs. success | 3 | Experiment 3 › Fig. 7 | `paper-fig7-confidence-vs-success.pdf` |
| Fig. 12: success by goal horizon | 3 | Experiment 3 › Fig. 12 | `paper-fig12-success-by-horizon.pdf` |
| Table 6: retention and recovery | 3 | Experiment 3 › Table 6 | `paper-table6-exp3-retention-recovery.csv` |
| Fig. 13: recovery rate vs. θ | 3 | Experiment 3 › Fig. 13 | `paper-fig13-recovery-vs-theta.pdf` |
| Fig. 9: worked example (App. A) | — | `pravass` package tests (`tests/test_search.py::TestPaperExample`) | — |

Fig. 8 combines Fig. 6, Fig. 7 and Table 1. Figs. 1–4 are illustrations and
example images from the dataset, and Table 2 describes the constraint sources.
The Table 5 CSV also lists Random θ-Mass and the shares of plans that violate a
precondition or are not found.

**Review additions**

| Result | Experiment | Notebook section | Output |
|---|---|---|---|
| Six backbones: belief-set quality at θ = 0.9 | 1 | Six VLM backbones: belief-set quality | `review-exp1-backbones.csv` |
| Coverage vs. set size per backbone | 1 | … › Coverage vs. set size per backbone | `review-exp1-backbones-mbs.pdf`, `review-exp1-backbones-all-methods.pdf` |
| Reliability: belief mass vs. coverage, and the largest gaps | 1 | Reliability | `review-reliability.pdf`, `review-reliability-backbones.pdf`, `review-reliability-gaps.csv` |
| Explored states on constrained instances | 2 | Explored states on constrained instances | `review-exp2-explored-states.csv` |
| Six backbones: planning success | 3 | Six VLM backbones: planning success | `review-exp3-backbones.csv` |
| Constraint ablation; exact vs. relaxed normalizer | 3 | Constraint ablation | `review-exp3-constraint-ablation.csv` |
| Paired contrasts (e.g. AMS vs. Adaptive Abstention) | 3 | Constraint ablation › Paired contrasts | `review-exp3-constraint-ablation-paired.csv` |
| Closed-loop replanning on ALFWorld | 6 | Closed-loop replanning | `review-exp6-{success,effects,efficiency}.csv`, `review-exp6-closed-loop.pdf` |
| Correlated perceptual noise | 7 | Correlated perceptual noise | `review-exp7-{distortion,robustness,block-modes}.csv`, `review-exp7-correlated-noise.pdf` |
| Correlated noise: ranges over the whole sweep | 7 | … › Ranges over the whole sweep | `review-exp7-sweep-ranges.csv` |

## Notes on the protocol

- **Calibration data.** Platt scaling and the Feature Abstention threshold
  are both fit on the whole validation split. `predictions.zip` holds the VLM
  answers on that split, so Step 3 is deterministic. The submitted version of
  the paper differs in two ways, both for GPT-4.1: its Platt scaling was fit
  on validation answers that were queried during calibration and not saved,
  and its Feature Abstention thresholds were picked on test instances (the
  thresholds use no labels, only the belief mass). The answers in
  `predictions.zip` come from later API calls on the same images, and
  GPT-4.1's log-probabilities vary between calls, so the GPT-4.1 numbers of
  the submitted version differ slightly from those this pipeline produces.
  The Feature Abstention baseline of Experiment 1 changes most: its set
  doubles with every abstained feature, so a small shift in the calibrated
  probabilities or in τ changes its size by orders of magnitude at θ ≥ 0.7.
- **Normalizer in Experiment 3.** AMS and MBSθ use all extracted mutex
  groups with the exact normalizer, the primary normalizer of App. G.5 (its
  variable elimination stays within the scope bound on every PRB and GW test
  instance), as in the submitted version. The constraint ablation adds the
  closed-form relaxed normalizer over the selected groups ("relaxed Z"). The
  runs during the review labeled the relaxed variant "AMS", and in GW they
  treated every selected group as exactly-one. Those groups are an object's
  table atoms without its gripper atom, so they are only at-most-one, and the
  normalizer fell below the exact one on 236 of 600 GPT-4.1 test instances.
  The code now applies exactly-one only to selected groups that are whole
  constraint groups, which keeps the relaxed normalizer an upper bound. For
  GPT-4.1, AMS success is then the same with either normalizer, so Tables 1
  and 5 do not depend on the choice; Table 6 and Fig. 7 use the belief mass,
  and do.
- **Time limits and hardware.** Each belief-set construction and each
  planner call has a wall-clock limit: 30 s in Experiments 1, 3 and 6, 10 s in
  Experiment 2, and 20 s in Experiment 7. A call that hits its limit counts as
  a failure, so results near the limit depend on the machine. The paper's runs
  used Intel Xeon 6710E cores, one process per core. On slower cores, or with
  more processes than physical cores, more calls time out, which lowers
  coverage, the number of instances solved and planning success. To
  compensate, scale the limits (`--method-timeout`, `--planning-timeout`) by
  the ratio of single-core speeds, and keep `--jobs` at most the number of
  physical cores. Runtimes themselves (Experiment 2) are always
  hardware-specific.
- **Splitting a run across machines.** The runners of Experiments 1–3 accept
  `--domains` and `--configs` and append to one `sample_results.jsonl` per
  model. To split a run, run each part in a separate clone of the repository,
  concatenate the `sample_results.jsonl` files, and run the same command once
  more to rebuild `aggregate_metrics.json` (it skips the records that exist).
- **Confidence intervals.** Intervals are 95% percentile-bootstrap intervals
  over test instances, with 10,000 resamples as the paper's captions state
  (the submitted Table 4 used 500, so its intervals differ in the last digit).
  Two exceptions: the six-backbone and constraint-ablation planning tables
  pool Wilson score intervals, and Experiments 6 and 7 compute their paired
  intervals in the runner with 1,000 resamples.

## Troubleshooting

- **OSF returns HTTP 429 (rate limit).** Wait a few minutes; the `curl`
  commands above retry automatically. `-f` makes `curl` fail instead of saving
  an error page.
- **The planner fails to start (`pythonnet`/Mono errors).** Activate the conda
  environment before `uv run`, so Mono is on the path.
- **`uv sync` fails with `Distribution not found at: .../experiments/up-cpor`.**
  Clone the planner into `up-cpor/` first (see Setup).
- **Out of memory in Experiments 3 or 6.** Lower `--jobs`. Each planning worker
  can use up to `MBS_PLANNER_MAX_GIB` GB.
