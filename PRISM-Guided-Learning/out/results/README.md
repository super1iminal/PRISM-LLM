# Results layout

- `core/<condition>/seed_*/`: the method's main runs, the ones the paper's main results come from (the conditions
  in `configs/core.yaml`).
- `baselines/<config name>/`: baselines (`src/run_baseline.py configs/baselines/<name>.yaml`), in the transfer CSV
  format.
- `ablations/<condition>/seed_*/`: every other condition: ablations of the method, and the earlier experiments
  (retry, blame and catch-all ablations, rule-language runs).
- `transfer_search/`: summary pages (`SUMMARY.md`, `FAMILIES.md`), diagnosis files and launcher logs.
- `ceilings/`, `regression/`, and the older top-level run directories: as described in `DECISIONS.md`.

`src/run_ablation.py` and `src/run_search.py` choose `core/` or `ablations/` from `configs/core.yaml`, and
`viz/transfer_summary.py` reads each condition from wherever it is.
