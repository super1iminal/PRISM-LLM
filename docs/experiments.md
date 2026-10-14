# Experiments: research questions over one store of runs

The paper asks two kinds of question, and the code has one harness for each, sharing the runs.

- **RQ1, comparison** (`src/experiments/comparison.py`, `configs/experiments/rq1_comparison.yaml`): our best configuration per benchmark against each baseline's best. Its cells are the **core** runs (`out/results/core/`).
- **RQ2, ablations** (`src/experiments/ablation.py`, `configs/experiments/rq2_*.yaml`): one factor at a time against a reference cell (the core configuration). Runs go to `out/results/ablations/`.

```bash
python src/run_experiment.py configs/experiments/rq2_feedback.yaml                 # plan + report from what exists
python src/run_experiment.py configs/experiments/rq2_*.yaml --run --check          # launch missing repeats, certify
python src/run_experiment.py configs/experiments/rq1_comparison.yaml --check --baselines
```

Reports: `out/results/experiments/<experiment>/REPORT.md`, a figure (ablations) and `runs.csv` (one row per run).

## The run store (`src/experiments/store.py`)

A run is identified by **what it did**, not by its condition's name. Its identity is its resolved config (`config.json` over today's defaults) without the settings that cannot change its results:
- the condition name;
- `run.workers` and `run.scheduler`;
- timeouts and client retries;
- the seed, which only labels a repeat.

An experiment cell is a condition plus overrides. Its runs are all finished runs with the same identity, under any name. So a configuration two experiments need runs once. For example, the "worst case over the choices" ablation cell reuses `haiku_ts_norestart`, which is the core configuration with `rules.any_order: none`. The plan printed by `run_experiment.py` names the runs each cell reuses.

Missing repeats are launched next to the cell's existing runs. If there are none, they go under `<root>/<condition>+<cell name slug>/seed_none_<k>/`.

Started but unfinished runs are listed as pending and never launched over. Delete failed ones.

**Caveat.** The identity is the config, so a code change that alters what a run does must come with a config key, as every behaviour change here does. Otherwise the store would mix runs made before and after it.

**Corrections.** A run's optional `identity.json` corrects what its `config.json` could not record. The case so far: features added to the domain after the run (`patrol`, `early`, `obstacle`, `risky`, Pac-Man's `closer_g*`). Its prompts never showed them, as if they were hidden. Runs whose recorded `rules.hidden_features` predates their condition file carry one. Runs whose config no longer loads (removed keys, e.g. the qwen-era runs) are never matched.

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
