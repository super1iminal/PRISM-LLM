# Results

The current method's results (the rebuilt loop of Oct 16, the `port_*` conditions; validation only, the test set is
sealed). Results of older versions of the approach (the previous loop, the legacy planner, the qwen-era runs, the
transfer search) were removed from the tree on Oct 19, 2026 and are in the git history; the clean-slate
implementation the loop was ported from is on branch `fresh-transfer` (tag `fresh-transfer-oct16`).

- **Which cells have runs, which are missing:** [`experiments/README.md`](experiments/README.md) (`python src/run_experiment.py --index`).
- **RQ1, our method against the baselines:** [`experiments/rq1_comparison/REPORT.md`](experiments/rq1_comparison/REPORT.md). Figures, best configuration of each method: `rq1_grids.png`, `rq1_grids_noclose.png`, `rq1_families.png`, `rq1_pacman.png`; Pac-Man's worst-case crash probability against the horizon: `rq1_pacman_horizon.png`. `chosen.csv`: the configuration chosen per method, which the test pass uses.
- **RQ2, ablations of the current loop:** [the gridworld feedback fix](experiments/rq2_feedback_fix/REPORT.md) (`rq2_feedback_fix`), [feedback against pure restarts](experiments/rq2_restarts/REPORT.md) (`rq2_restarts`, before the fix).
- **The paper's tables and figures:** `../paper/` (`viz/paper_results.py`; `../paper/RESULTS-NOTES.md` maps each to its runs), including the sequential study (`sequential/`).

Rebuild a report: `python src/run_experiment.py configs/experiments/<experiment>.yaml` (from `PRISM-Guided-Learning/`; `docs/experiments.md`).

## Layout

- `core/<run name>/seed_*/`: the RQ1 cells (`configs/experiments/rq1_comparison.yaml`).
- `ablations/<run name>/seed_*/`: every other run of the current loop (a cell with overrides runs under
  `<condition>+<cell name>`), and `D7`, `B2`, `R1`, `S1`: the four saved runs `tests/test_scheduler.py` replays.
- `baselines/<config name>/`: baselines (`src/run_baseline.py configs/baselines/<name>.yaml`), in the transfer CSV
  format (`_open.csv`: with ties left open).
- `experiments/<experiment>/`: the reports of `src/run_experiment.py` (`REPORT.md`, figures, `runs.csv`), and the
  index `experiments/README.md`.
- `sequential/<study>/`: the sequential adaptation study (`src/sequential.py`).
- `equal_spend/`: feedback runs re-picked at the restarts' spend (`src/equal_spend.py`).
- `ceilings/`: the per-instance optimum (`src/ceilings.py`).

Held-out checks per run: `transfer_<dataset>[_<rules>].csv` (the loop's solver, `src/transfer.py`) and
`transfer_exact_*.csv` (the exact solvers, `src/recertify.py`: interval iteration, then linear programming where it
stalls; `transfer_ii_*` / `transfer_lp_*` are the one-method checks they were combined from).

Experiments find their runs by identity, not by name (`src/experiments/store.py`, `docs/experiments.md`). A run's
`identity.json`, if present, corrects what its `config.json` could not record. `src/run_ablation.py` and
`src/run_search.py` choose `core/` or `ablations/` from the RQ1 cells.
