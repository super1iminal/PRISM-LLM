# Results layout

- `core/<run name>/seed_*/`: the method's main runs, the ones the paper's main results come from (the cells of the
  RQ1 comparison, `configs/experiments/rq1_comparison.yaml`).
- `ablations/<run name>/seed_*/`: every other run: RQ2 ablations (`configs/experiments/rq2_*.yaml`; a cell with
  overrides runs under `<condition>+<cell name>`), and the earlier experiments (retry, blame and catch-all
  ablations, rule-language runs).
- `baselines/<config name>/`: baselines (`src/run_baseline.py configs/baselines/<name>.yaml`), in the transfer CSV
  format (`_open.csv`: with ties left open).
- `experiments/<experiment>/`: the reports of `src/run_experiment.py` (`REPORT.md`, figure, `runs.csv`) and batch
  logs.
- `transfer_search/`: earlier summary pages (`SUMMARY.md`, `FAMILIES.md`), diagnosis files and launcher logs.
- `ceilings/`, `regression/`, and the older top-level run directories: as described in `DECISIONS.md`.

Experiments find their runs by identity, not by name (`src/experiments/store.py`, `docs/experiments.md`). A run's
`identity.json`, if present, corrects what its `config.json` could not record (features added after the run).
`src/run_ablation.py` and `src/run_search.py` choose `core/` or `ablations/` from the RQ1 cells, and
`viz/transfer_summary.py` reads each condition from wherever it is.
