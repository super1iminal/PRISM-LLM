# Baseline configs

One file per baseline run of `src/run_baseline.py` (methods in `src/baselines/`). Results go to `out/results/baselines/<file stem>/transfer_<dataset stem>.csv`, in `src/transfer.py`'s format plus `method`, so `viz/transfer_summary.py` lists baselines next to LLM conditions, and `fit.json` for methods that report what their fit produced (e.g. a tree's size and rules).

| Key | What it does |
|---|---|
| `method` | the baseline (`src/baselines/__init__.py`, `METHODS`) |
| `condition` | run condition whose rule language (`rules.*`) and PRISM settings apply; `null` = the default run config |
| `dataset` | instances, as in a run config |
| `target_dataset` | the targets' dataset, when it differs from `dataset` (e.g. a sealed test set); `null` = the same. Then no target counts as trained |
| `train` | ids of the training instances (`"0-11"`, `"0,4,8"`); `""` for methods that are not fitted |
| `targets` | ids of the instances to certify on |
| `workers` | targets checked at once |
| `exact_near` | re-check with the exact solvers when a worst case is within this of its threshold (0: never) |
| `params` | the method's own settings (its module docstring) |

`params.model: explicit` (`decision_tree`, `ppo`) runs a method on any domain, from the explicit models PRISM exports (`src/baselines/explicit_mdp.py`); without it they are the gridworld versions. The methods `uniform`, `random_policies` and `one_instance` (`src/baselines/simple.py`) work on any domain too. Per-state policies (PPO, `simple.py`) are certified on each target's explicit model, one target at a time (a model of millions of states takes gigabytes), so `workers` does not apply to them.

Pac-Man (`pacman_*.yaml`, condition `pacman_baseline`: extended vocabulary, numbers allowed): trees over raw variables and over our features, each trained on horizon 5 and on horizons 14-18 (`pacman_tree_raw`, `pacman_tree_features`); PPO over raw variables and over our features, three seeds (`pacman_ppo_raw`, `pacman_ppo_features`); `pacman_uniform`, `pacman_random_policies`, `pacman_one_instance` (horizon 18's strategy, aligned by moves made or by moves left); the hand-written maze rules. All are certified on the validation horizons, ids 0-9; ids 10 and 11 (horizons 50, 100) are sealed.
