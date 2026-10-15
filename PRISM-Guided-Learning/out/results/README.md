# Results

## Current results (validation; the test set is sealed)

- **Which cells have runs, which are missing:** [`experiments/README.md`](experiments/README.md) (`python src/run_experiment.py --index`).
- **RQ1, our best configuration against the baselines:** [`experiments/rq1_comparison/REPORT.md`](experiments/rq1_comparison/REPORT.md). Every baseline number is PRISM-certified. Figures, best configuration of each method: `rq1_grids.png`, `rq1_grids_noclose.png`, `rq1_families.png`, `rq1_pacman.png`; Pac-Man's headline, worst-case crash probability against the horizon: [`rq1_pacman_horizon.png`](experiments/rq1_comparison/rq1_pacman_horizon.png). `chosen.csv`: the configuration chosen per method, which the test pass uses.
- **RQ2, ablations against the core configuration:** at a glance in [`experiments/rq2_overview.png`](experiments/rq2_overview.png); one report each:
  - [feedback, restarts, or both](experiments/rq2_feedback/REPORT.md) (`rq2_feedback`)
  - [what the verifier tells the LLM](experiments/rq2_verifier_output/REPORT.md) (`rq2_verifier_output`)
  - [prompting](experiments/rq2_prompting/REPORT.md) (`rq2_prompting`)
  - [the domain's features](experiments/rq2_features/REPORT.md) (`rq2_features`)
  - [training sets and rounds](experiments/rq2_training/REPORT.md) (`rq2_training`)
  - [who resolves the open choices](experiments/rq2_choices/REPORT.md) (`rq2_choices`)

Rebuild a report: `python src/run_experiment.py configs/experiments/<experiment>.yaml` (from `PRISM-Guided-Learning/`; `docs/experiments.md`).

## Layout

- `core/<run name>/seed_*/`: the method's main runs, the ones the paper's main results come from (the cells of the
  RQ1 comparison, `configs/experiments/rq1_comparison.yaml`).
- `ablations/<run name>/seed_*/`: every other run: RQ2 ablations (`configs/experiments/rq2_*.yaml`; a cell with
  overrides runs under `<condition>+<cell name>`), and the experiments outside the two harnesses (retry, blame and
  catch-all ablations, rule-language runs). `ablations/summary/`: the qwen ablation summary page.
- `baselines/<config name>/`: baselines (`src/run_baseline.py configs/baselines/<name>.yaml`), in the transfer CSV
  format (`_open.csv`: with ties left open).
- `experiments/<experiment>/`: the reports of `src/run_experiment.py` (`REPORT.md`, figures, `runs.csv`), and the
  index `experiments/README.md`.
- `ceilings/`: the per-instance optimum (`src/ceilings.py`).

Experiments find their runs by identity, not by name (`src/experiments/store.py`, `docs/experiments.md`). A run's
`identity.json`, if present, corrects what its `config.json` could not record (features its prompts never showed).
`src/run_ablation.py` and `src/run_search.py` choose `core/` or `ablations/` from the RQ1 cells, and
`viz/transfer_summary.py` reads each condition from wherever it is.

## Archive

`archive/`: the pages and runs from before the RQ harnesses (qwen-era runs and comparisons, `haiku_large`,
`sonnet_vs_qwen`, the rule-language pages, the transfer search, the qwen UUV run). [`archive/README.md`](archive/README.md)
says what each was and which decision in `DECISIONS.md` it supports.
