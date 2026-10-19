# Data visualization

- **Experiment reports** (RQ1 per benchmark, the RQ2 ablations, the index) come from the experiment harness:
  `python src/run_experiment.py configs/experiments/<experiment>.yaml` (and `--index`) writes them next to the reports
  in `out/results/experiments/` (`docs/experiments.md`).
- **The paper's tables and figures**: `python viz/paper_results.py` (from `PRISM-Guided-Learning/`) reads existing runs
  only and writes `out/paper/` (`out/paper/RESULTS-NOTES.md` maps every table and figure to its runs).
- `plot_budget_transfer.py`: budget curves (training score and held-out certification against spend) of an ablation
  experiment with a `budget:` block, drawn by its report.
- `theme.py`: shared colours and axes style.

Facts about a run (dataset, model, round budget) are read from its `config.json` (`src/results_io.py`), never
repeated in a figure's settings.
