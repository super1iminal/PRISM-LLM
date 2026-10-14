# Baseline configs

One file per baseline run of `src/run_baseline.py` (methods in `src/baselines/`). Results go to `out/results/baselines/<file stem>/transfer_<dataset stem>.csv`, in `src/transfer.py`'s format plus `method`, so `viz/transfer_summary.py` lists baselines next to LLM conditions.

| Key | What it does |
|---|---|
| `method` | the baseline (`src/baselines/__init__.py`, `METHODS`) |
| `condition` | run condition whose rule language (`rules.*`) and PRISM settings apply; `null` = the default run config |
| `dataset` | instances, as in a run config |
| `train` | ids of the training instances (`"0-11"`, `"0,4,8"`); `""` for methods that are not fitted |
| `targets` | ids of the instances to certify on |
| `workers` | targets checked at once |
| `exact_near` | re-check with the exact solvers when a worst case is within this of its threshold (0: never) |
| `params` | the method's own settings (its module docstring) |
