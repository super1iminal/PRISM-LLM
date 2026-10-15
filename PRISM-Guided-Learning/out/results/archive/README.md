# Archive: result pages from before the RQ harnesses

Pages and runs made before `src/run_experiment.py` (RQ1, RQ2). They are kept because a decision in `DECISIONS.md` rests on them, or because a test, the regression or a baseline reads them. Current results: `out/results/README.md`. Moved here with `git mv`, so `git log --follow` shows each one's history; the plot configs that rebuild them (`configs/plot/*.yaml`) point here.

| Directory | What it was | Supports (`DECISIONS.md`) |
|---|---|---|
| `legacy_grid20/` | The predecessor's per-state planner with qwen3:14b on `grid_20_balanced` (one seed). Read by the regression (`configs/regression/default.yaml`) and `tests/test_compare.py` | Running and comparing: "Legacy baseline"; Solvers and numerics: "Regression" |
| `legacy_qwen/` | Summary page: legacy with the obstacle visible against the symbolic loop, qwen3:14b (`configs/plot/legacy_qwen.yaml`) | "Legacy baseline" (changed only for fairness) |
| `symbolic_grid20_capped/`, `symbolic_grid20_default/`, `symbolic_grid20_nocatchall/`, `symbolic_grid20_noexample/` | Single-seed qwen3:14b symbolic runs on `grid_20_balanced`: schema caps, the default config, and without the catch-all instruction or example | Method: "Output and parsing" (caps); "The default prompt keeps the catch-all instruction and example" |
| `comparison_grid20/`, `comparison_grid20_default/` | End-to-end legacy against symbolic reports of those runs (`src/compare.py`) | Same as above |
| `symbolic_uuv/` | A qwen3:14b UUV run, read by `configs/plot/uuv.yaml` and `uuv_summary.yaml` | Case studies: "UUV"; Open questions: UUV |
| `haiku_large/` | Haiku 5.5 on `grid_20_large`: feedback against resampling at equal cost (`configs/plot/haiku_large.yaml`) | Running and comparing: "Feedback must beat resampling at equal cost" |
| `sonnet_vs_qwen/` | Claude Sonnet 5 / 5.5 and Haiku 5.5 against qwen3:14b (`configs/plot/sonnet_vs_qwen.yaml`) | Running and comparing: "Ablation model and dataset: Haiku 5.5 on `grid_20_large`" |
| `rule_language_grid/`, `rule_language_pacman/` | The base rule vocabulary against the general one (constants, features, `any`), grids and Pac-Man | Method: "Rule language", "Gridworld features" |
| `transfer_search/` | The transfer search's pages (`SUMMARY.md`, `FAMILIES.md`) and `diagnosis/`: the searched and order-diagnosis rule lists that `configs/baselines/search_grid_large.yaml`, `any_orders.yaml` and `pacman_orders.yaml` certify | Method: "Multi-instance training", "`any` resolved in the loop", "Gridworld features"; the bar (rule-list search) |

`out/results/ablations/summary/` (the qwen ablation summary, `configs/plot/ablation_summary.yaml`) predates the harnesses too, but stays next to the runs it summarizes, as `ablations/` holds runs the run store and the replays read.
