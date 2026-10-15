# Experiments: research questions over one store of runs

The paper asks two kinds of question, and the code has one harness for each, sharing the runs.

- **RQ1, comparison** (`src/experiments/comparison.py`, `configs/experiments/rq1_comparison.yaml`): our best configuration per benchmark against each baseline's best. Its cells are the **core** runs (`out/results/core/`).
- **RQ2, ablations** (`src/experiments/ablation.py`, `configs/experiments/rq2_*.yaml`): one factor at a time against a reference cell (the core configuration). Runs go to `out/results/ablations/`.

RQ1 covers transfer with and without gridworld's shortest-path feature `closer` (the `grids_noclose` benchmark). Two further questions come from the same runs: cost (tokens and dollars per training set, in every report) and larger instances (grids up to 20x20 and Pac-Man horizon 25, below). Readability (rules against tree nodes and network weights) has no metric in the harness.

```bash
python src/run_experiment.py configs/experiments/rq2_feedback.yaml                 # plan + report from what exists
python src/run_experiment.py configs/experiments/rq2_*.yaml --run --check          # launch missing repeats, certify
python src/run_experiment.py configs/experiments/rq1_comparison.yaml --check --baselines
```

Reports: `out/results/experiments/<experiment>/REPORT.md`, a figure (ablations) and `runs.csv` (one row per run).

## Datasets and splits

- **Random grids** (`grid_20_large.csv`: four grids of each size 9x9-13x13). Training sets of a 9x9, a 10x10 and an 11x11 grid: ids `[0,4,8]`, `[1,5,9]`, `[2,6,10]`, `[3,7,11]`. Validation: grids 12-19 (12x12, 13x13), the metric for iterating.
- **Gridworld families** (`families_v0.csv`; generator `domains/gridworld/data/families.py`): parameterized layouts in the sense of 1-2-3-Go!'s parameterized MDPs, with the size `n` and a seed as parameters. Three families (rooms, crossing, adjacent), sizes 6-20, seeds 0-3: 60 members each. Training: three 6x6-8x8 members of one family. Validation: the 10x10-14x14 members (20 per family). Test: the 15x15-20x20 members. Positions scale with `n` (goals in corners, doors in the middle), which trees over raw coordinates cannot follow. A family is kept only if, at every size checked:
  - every member is jointly solvable (the joint query);
  - the template or generic greedy rules fail some members, so the family is not trivial;
  - some single rule list certifies every member checked, so transfer is possible.
- **Pac-Man**: the loop trains on `pacman.csv` (horizons 8-16); frozen rules are certified on `pacman_transfer.csv` (horizons 8-25).
- **Sealed test sets** (`sealed` in `configs/transfer/default.yaml`; `transfer.py` and `run_baseline.py` refuse their targets without `--unseal`):
  - `grid_test_large.csv`: the 30 grids of sizes 9-13 in `grid_100_balanced.csv` that are not in `grid_20_large.csv` (6 per size), plus 4 each of 15x15, 17x17 and 20x20 from `datagen.py` (seeds 515, 517, 520). No LLM run trains on them.
  - The 15x15-20x20 members of `families_v0.csv` (ids 36-59, 96-119, 156-179).

  They give only the final numbers: the core configuration (5 repeats) and every baseline, once the configuration is fixed.

## Methods

The cells and `baselines` of `rq1_comparison.yaml`: ours with Haiku 5.5 and with Sonnet 5.5; 1-2-3-Go! reimplemented (trees over the raw state variables, `src/baselines/decision_tree.py`) and the same trees over our features; the search over the rule language (`src/baselines/feature_search.py`); PPO over our features (`ppo.py`; a MiniGrid-style local view is the `ppo_view_*` configs); the hand-written template (`domains/gridworld/data/template_policy.py`); hand-written maze rules for Pac-Man. The per-instance optimum (`src/ceilings.py`) is the ceiling, not a transfer method. Every method's output is certified by PRISM in `transfer.py`'s CSV format.

- **Trees** follow the paper: one reachability objective (every goal in order, nothing entered early, no collision), CART on the training instances' optimal actions, frozen, then certified.
- **Baselines** (`src/baselines/`): one module per method, each a `Baseline` (`base.py`), run by `src/run_baseline.py configs/baselines/<name>.yaml` (keys in `configs/baselines/README.md`). Results: `out/results/baselines/<config stem>/transfer_<dataset stem>.csv`, the transfer CSV plus `method`. The search and the trees rank candidates with a numpy model of the gridworld (`gridworld_mdp.py`); PRISM certifies what they keep.
- **Third-party code** lives in `third_party/<name>/` (not committed; `third_party/README.md` records source, version and license). Nothing in `core/` imports it.

## Running conventions

- **Parallel runs.** `src/run_search.py` and `run_experiment.py --run` run several jobs at once (local qwen runs go one at a time), so wall-clock times are not comparable and tokens are the cost measure. Each job keeps up to `run.workers` PRISM processes busy (`prism.java_max_mem: 4g` each), so `--max-parallel` also caps memory (the machine has 31 GB).
- **Haiku settings:** thinking on, 16k output tokens, 5 rounds, unseeded repeats (Claude endpoints take no seed). Sonnet runs need Asher's go-ahead.
- **Each idea is a config flag**, off by default, with a condition file of its own, documented in `docs/config.md` (and `docs/semantics.md` when it changes the loop). The run store depends on this (caveat below).

## The run store (`src/experiments/store.py`)

A run is identified by **what it did**, not by its condition's name. Its identity is its resolved config (`config.json`, with any key it lacks taken from `configs/run/default.yaml`) without the settings that cannot change its results:
- the condition name;
- `run.workers` and `run.scheduler`;
- timeouts and client retries;
- the seed, which only labels a repeat.

An experiment cell is a condition plus overrides. Its runs are all finished runs with the same identity, under any name. So a configuration two experiments need runs once. For example, the "worst case over the choices" ablation cell reuses `haiku_ts_norestart`, which is the core configuration with `rules.any_order: none`. The plan printed by `run_experiment.py` names the runs each cell reuses.

Missing repeats are launched next to the cell's existing runs. If there are none, they go under `<root>/<condition>+<cell name slug>/seed_none_<k>/`.

Started but unfinished runs are listed as pending and never launched over. Delete failed ones.

**Caveat.** The identity is the config, so a code change that alters what a run does must come with a config key, as every behaviour change here does. Otherwise the store would mix runs made before and after it.

**Corrections.** A run's optional `identity.json` corrects what its `config.json` could not record: domain features its prompts never showed although its config does not hide them (`patrol`, `early`, `obstacle`, `risky`, Pac-Man's `closer_g*`), recorded as hidden. Runs whose config does not load (keys the schema has dropped, e.g. the qwen runs) are never matched.

## Experiment files (`configs/experiments/`)

| Key | What it is |
|---|---|
| `kind` | `comparison` (RQ1) or `ablation` (RQ2) |
| `title`, `question` | for the report |
| `benchmarks` | name -> {`base` condition, `targets`, optional `dataset` (targets from another dataset), `groups` and `source_groups` when each rule list belongs to one group (families), `label`}; or the name of a set in `benchmarks.yaml` |
| `repeats` | runs per cell (a cell may set its own) |
| `cells` | {`name` (descriptive, used in tables), `code` (short, figures), `set` (overrides: a list for every benchmark, or benchmark -> list), `condition` (benchmark -> a condition other than the base, e.g. the Sonnet one), `only` (benchmarks), `repeats`, `launch` (false: only existing runs; Sonnet needs Asher's go-ahead)} |
| `reference` | ablations: the reference cell's name (default: the first) |
| `baselines` | comparisons: {`name`, `config` (benchmark -> `configs/baselines/*.yaml`), `sources` (benchmark -> list, or group -> list), `ties` (false: one action per state, so nothing to leave open)} |
| `prices` | model -> [$ per M prompt tokens, $ per M output tokens] |

## Metrics (`src/experiments/metrics.py`)

All metrics are computed on held-out targets only; a list's own training instances are left out.

| Metric | Definition |
|---|---|
| **certified** | targets certified per rule list, on its own group's targets where there are groups; mean over a run's lists, then over runs |
| **every target** | the same, on all targets, whatever the group (e.g. a rooms-trained list on all three families) |
| **best list** | the best single list's count |
| **library** | targets certified by at least one of a run's lists (certification at run time picks one that passes) |
| **ties open** | certified with the lists' action orders left open (`core.rules.open_ties`, `transfer.py --rules open`): the worst case over every way of breaking ties. A list that passes this way needs no solver choice at all |

Ablation reports add the difference to the reference, with a 90% bootstrap interval over runs. With 3–5 runs per cell, read the interval as a rough guide.

`src/tie_analysis.py` (gridworld, numpy model) reports how much the orders decide:
- how many actions the deciding rule allows per reachable state;
- whether the list passes with ties open, at the worst and at the best choice per state;
- how many targets it passes under each of the 24 fixed orders.
