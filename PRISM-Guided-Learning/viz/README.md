# Data visualization

Every figure (and the comparison report) is made from a config in `configs/plot/`, which names its
script, the runs, labels, title and output. Run from `PRISM-Guided-Learning/`:

```bash
python viz/plot_runs.py configs/plot/catchall_ablation.yaml
```

Override any key for a one-off (`--set out=/tmp/test.png`), or copy a config to make a new figure.
Facts about a run (dataset, model, round budget) are read from its `config.json`
(`src/results_io.py`), never repeated in a plot config.

| script | what it draws | config(s) |
|---|---|---|
| `plot_comparison.py` | grid-by-grid legacy vs one symbolic run, plus a CSV | `grid20.yaml`, `grid20_default.yaml` |
| `plot_runs.py` | legacy vs up to two symbolic runs (e.g. prompt ablations) by grid size, loop-mode usage and coverage; also a markdown table and a CSV | `catchall_ablation.yaml`, `catchall_ablation_full.yaml` |
| `plot_budget.py` | success / requirements met / shortfall vs rounds budget k (ablation F1), replaying keep-best on the first k rounds; pools `seed_*` dirs. With `cost`, x = mean spend through round k (symbolic runs only) | `budget_grid20.yaml`, `budget_grid20_default.yaml`, `budget_tokens_qwen.yaml` |
| `plot_domain.py` | any domain: final policies and the domain's reference policies against the range any controller achieves, per instance and requirement | `uuv.yaml`, `uuv_sonnet.yaml` |
| `plot_uuv_summary.py` | UUV: our policy vs the paper's controller (probabilities, policy size, the paper's Table 2 costs) | `uuv_summary.yaml`, `uuv_summary_sonnet.yaml` |
| `ablation_summary.py` | a set of conditions under `out/results/ablations/` on one page (`SUMMARY.md` + figures, paired tests vs each reference, or `planned` comparisons with Holm correction; a spend row in the budget figure with `cost`): every ablation, the model comparison, or the Haiku ablation. Rerun after each run | `ablation_summary.yaml`, `sonnet_vs_qwen.yaml`, `haiku_large.yaml` |
| `plot_transfer.py` | transfer heatmaps: each frozen rule set (row: an instance's, or a training set's) certified on every instance (column), from `src/transfer.py`'s CSVs; certified cells in blue, the instances a rule set was written for outlined | `transfer_grid.yaml`, `transfer_mi_grid.yaml` |
| `plot_ablation_grid.py` | the ablation design grids in `docs/` (`ablation_run.png`, `ablation_not_run.png`). Edit its row tables when the plan changes | `ablation_grid.yaml` |
| `../src/compare.py` | end-to-end legacy vs symbolic report (`report.md`, `per_sample.csv`) | `compare_grid20.yaml`, `compare_grid20_default.yaml` |

- `loaders.py`: reads legacy and symbolic runs into one per-sample table (final probabilities, requirements met
  and shortfall against each instance's own requirements, time, tokens). Symbolic runs without a parquet (aborted
  or in progress) are rebuilt from their worker logs; token counts are unavailable in that case.
- `theme.py`: shared colours and axes style.
- `figures/`: generated output.
