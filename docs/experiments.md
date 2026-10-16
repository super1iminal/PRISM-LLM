# Experiments: research questions over one store of runs

The paper asks two kinds of question, and the code has one harness for each, sharing the runs.

- **RQ1, comparison** (`src/experiments/comparison.py`, `configs/experiments/rq1_comparison.yaml`): our best configuration per benchmark against each baseline's best. Its cells are the **core** runs (`out/results/core/`).
- **RQ2, ablations** (`src/experiments/ablation.py`, `configs/experiments/rq2_*.yaml`): one factor at a time against a reference cell (the core configuration), on random grids, the gridworld families and Pac-Man (the `rq2` set in `configs/experiments/benchmarks.yaml`; cells that only make sense on gridworld say `only: [grids, families]`). Runs go to `out/results/ablations/`.

RQ1 covers transfer with and without gridworld's shortest-path feature `closer` (the `grids_noclose` benchmark). Two further questions come from the same runs: cost (tokens and dollars per training set, in every report) and larger instances (grids up to 20x20 and Pac-Man horizon 30, below). Readability (rules against tree nodes and network weights) has no metric in the harness.

```bash
python src/run_experiment.py configs/experiments/rq2_feedback.yaml                 # plan + report from what exists
python src/run_experiment.py configs/experiments/rq2_*.yaml --run --check          # launch missing repeats, certify
python src/run_experiment.py configs/experiments/rq1_comparison.yaml --check --baselines
python src/run_experiment.py --index                                                # which cells miss runs
```

Reports: `out/results/experiments/<experiment>/REPORT.md`, figures and `runs.csv` (one row per run). The index, `out/results/experiments/README.md` (`--index`), lists every experiment's cells per benchmark with their runs, pending and missing repeats, and each baseline's certification; it also draws `rq2_overview.png`, every ablation cell against the core configuration.

**Best configuration per method (RQ1).** Each configuration (cell or baseline) has a `method` (e.g. trees over raw variables and over our features are both 1-2-3-Go!). Per benchmark, a method's best configuration is the one with the highest mean certified per list on the validation targets, then the larger library, then the lower cost; only fully certified configurations compete. The tables show every configuration, the figures (`rq1_<benchmark>.png`, and Pac-Man's `rq1_pacman_horizon.png`) only the best of each method. The choice is written to `chosen.csv`; the test pass uses it and never re-chooses on test results. A baseline is counted on the same held-out targets as our lists (on Pac-Man, without our training horizons).

## Datasets and splits

- **Random grids** (`grid_20_large.csv`: four grids of each size 9x9-13x13). Training sets of a 9x9, a 10x10 and an 11x11 grid: ids `[0,4,8]`, `[1,5,9]`, `[2,6,10]`, `[3,7,11]`. Validation: grids 12-19 (12x12, 13x13), the metric for iterating.
- **Gridworld families** (`families_v0.csv`; generator `domains/gridworld/data/families.py`): parameterized layouts in the sense of 1-2-3-Go!'s parameterized MDPs, with the size `n` and a seed as parameters. Three families (rooms, crossing, adjacent), sizes 6-20, seeds 0-3: 60 members each. Training: three 6x6-8x8 members of one family. Validation: the 10x10-14x14 members (20 per family). Test: the 15x15-20x20 members. Positions scale with `n` (goals in corners, doors in the middle), which trees over raw coordinates cannot follow. A family is kept only if, at every size checked:
  - every member is jointly solvable (the joint query);
  - the template or generic greedy rules fail some members, so the family is not trivial;
  - some single rule list certifies every member checked, so transfer is possible.
- **Pac-Man**: the core configuration (`haiku_mi_pacman_long`) trains one rule list on horizons 14, 16 and 18 (`pacman_transfer.csv` ids 3-5); frozen rules are certified on the validation horizons of `pacman_transfer.csv` (8-25 and 30, ids 0-7 and 9), so 6 are held out (8, 10, 12, 20, 25, 30; id 9 was appended later, so older reports stop at 25). Horizon 5 (id 8) is 1-2-3-Go!'s training instance and no validation target: every policy gets 0.5511 there, since the bare model's minimum and maximum are equal. Older conditions train on `pacman.csv` (horizons 8-16). The bare model grows fast with the horizon: 498 states at 5, 0.41M at 18, 3.7M at 25 (317k decision states), 8.3M at 30; PRISM's explicit engine builds 30 in about 2 minutes with a 4 GB heap and runs out of memory on 50 (past 13.8M states). The optimum is 0.5511 at every horizon up to 30.
- **Mars Exploration Rovers trained on one instance** (`mer_one`, `mer_n_one`): as 1-2-3-Go!'s paper, every method trains on one base instance, n = 1 (x = 0.01 on `mer_transfer.csv`, id 7; x = 0.89 on `mer_n_transfer.csv`, id 0), and is certified on the other validation ids (0-6 and 8; 1-8). With three training instances both MER benchmarks end in a tie at 6 of 6. Ours: the port of the clean-slate method (`port_mer_one`, `port_mer_n_one`).
- **Sealed test sets** (`sealed` in `configs/transfer/default.yaml`; `transfer.py` and `run_baseline.py` refuse their targets without `--unseal`):
  - `grid_test_large.csv`: the 30 grids of sizes 9-13 in `grid_100_balanced.csv` that are not in `grid_20_large.csv` (6 per size), plus 4 each of 15x15, 17x17 and 20x20 from `datagen.py` (seeds 515, 517, 520). No LLM run trains on them.
  - The 15x15-20x20 members of `families_v0.csv` (ids 36-59, 96-119, 156-179).
  - Pac-Man horizons 50 and 100 (`pacman_transfer.csv` ids 10, 11; Asher). 200 was dropped: PRISM cannot build the model (above). Even 50 is beyond the bare model; whether policies certify there in time is only known at the test pass. The RQ1 horizon figure stops at 30.

  They give only the final numbers: the core configuration (5 repeats) and every baseline, once the configuration is fixed.

## Methods

The cells and `baselines` of `rq1_comparison.yaml`: ours with Haiku 5.5 and with Sonnet 5.5; 1-2-3-Go! reimplemented (trees over the raw state variables, `src/baselines/decision_tree.py`) and the same trees over our features; the search over the rule language (`src/baselines/feature_search.py`); PPO over our features and over a MiniGrid-style local view (`ppo.py`, the `ppo_features_*` and `ppo_view_*` configs; one method, two configurations); the hand-written template (`domains/gridworld/data/template_policy.py`); hand-written maze rules for Pac-Man; and on Pac-Man, the uniform random policy, the best of 1000 random deterministic policies and the optimal strategy of one instance applied to the others (`simple.py`). The per-instance optimum (`src/ceilings.py`) is the ceiling, not a transfer method. Every method's output is certified by PRISM in `transfer.py`'s CSV format.

- **Trees** follow the paper: one reachability objective (every goal in order, nothing entered early, no collision; Pac-Man: the crash requirement), CART on the training instances' optimal actions, frozen, then certified. On Pac-Man they learn from horizon 5 (the paper's training instance) and from our training horizons 14-18, with every optimal action of every decision state (weight 1/k, states where every action is optimal left out).
- **Any domain** (`explicit_mdp.py`): trees (`decision_tree.py`) and PPO (`ppo.py`) take `params.model: explicit`, and the uniform, random and one-instance baselines (`simple.py`) are built on it too. PRISM exports the instance's model, numpy computes optimal actions, policy values and simulation on it, and nothing in it is domain-specific, so a new domain (plan 08) gets these baselines from a config. A tree compiles exactly into rules, certified like any rule list. Policies with no compact rules (a network, a table, the uniform policy) are certified on the target's model restricted to the policy, written as PRISM explicit files and checked with `-importmodel`; this needs the bare model of the target, so on Pac-Man it stops at horizon 30.
- **Baselines** (`src/baselines/`): one module per method, each a `Baseline` (`base.py`), run by `src/run_baseline.py configs/baselines/<name>.yaml` (keys in `configs/baselines/README.md`). Results: `out/results/baselines/<config stem>/transfer_<dataset stem>.csv`, the transfer CSV plus `method`. The search and the gridworld trees rank candidates with a numpy model of the gridworld (`gridworld_mdp.py`); PRISM certifies what they keep. `fit.json` next to the CSV records what a fit produced (a tree's size and its rules, one per leaf).
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
| `cells` | {`name` (descriptive, used in tables), `code` (short, figures), `set` (overrides: a list for every benchmark, or benchmark -> list), `condition` (benchmark -> a condition other than the base, e.g. the Sonnet one), `only` (benchmarks), `repeats`, `launch` (false: only existing runs; Sonnet needs Asher's go-ahead)}. A cell whose `set` or `condition` is given per benchmark runs only on the benchmarks it names, never as the base configuration on the others |
| `reference` | ablations: the reference cell's name (default: the first) |
| `baselines` | comparisons: {`name`, `code`, `method`, `config` (benchmark -> `configs/baselines/*.yaml`), `sources` (benchmark -> list, or group -> list), `ties` (false: one action per state, so nothing to leave open)} |
| `method` (cells and baselines) | comparisons: the method a configuration belongs to (default: its name); figures show each method's best configuration |
| `horizon_figure` | comparisons: a value against the horizon, one line per method: {`benchmark`, `methods` (optional: only these lines), `value` (the transfer CSVs' `worst_<value>`), `horizon` (the target dataset's column), `optimum`, `threshold`, `max_horizon` (and `why_max`), `published` ({`label`, `points`: horizon -> value})}. Sealed targets are always left out |
| `prices` | model -> [$ per M prompt tokens, $ per M output tokens] |
| `budget` | ablations: comparisons at equal cost, fixed before the runs (`rq2_feedback.yaml`): {`plot` (a `configs/plot/` file for `viz/plot_budget_transfer.py`), `reference`, `cost` (dollars), `level`, `note`, `planned`: list of {`name`, `metric` (certified or solved), `cell`, `at` (fraction of the reference's mean spend)}}. The report then opens with the planned comparisons and the budget figure |

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

**Budget curves** (`src/budget_transfer.py`, PRISM only): per training set and round, the list keep-best holds after that round (the last improved iteration's rules, `any` already resolved), the tokens and dollars spent so far, whether the set is solved, and the list's certification on the benchmark's targets, cached by a hash of the list (`<run>/budget_transfer.csv`). Final lists reuse the transfer CSV; other lists get the exact re-check only when the values near a threshold decide the verdict. It also records the kept list's training score (requirements met in the worst case, summed over the training members), so a training score that rises while held-out certification stays flat shows the loop patching the training instances. Round 1 is the LLM's first answer, before any feedback. Read at a budget per training set, a run's value is that of the list held after its last round within the budget. Keep-best makes every curve rise, so the curves only compare cells at equal spend (`budget` in an experiment file).

`src/tie_analysis.py` (gridworld, numpy model) reports how much the orders decide:
- how many actions the deciding rule allows per reachable state;
- whether the list passes with ties open, at the worst and at the best choice per state;
- how many targets it passes under each of the 24 fixed orders.
