# 01 · Results organization: one place, PRISM-certified, best configuration per method

Read `plans/00-overview.md` first (standing rules, coordination, file owners).

## Goal

Anyone, Asher, Rajan or a reviewer, opens one page and sees the current results: RQ1 against the baselines and each RQ2 ablation, with what is still missing. Everything older is archived and labelled. Every baseline number is PRISM-certified. The RQ1 figures show only the best configuration of each method.

## Why

Results are spread over about fifteen top-level pages under `out/results/` (qwen-era comparisons, `haiku_large`, `sonnet_vs_qwen`, `rule_language_*`, `transfer_search`, `ablations/summary`, UUV figures). RQ1 shows several baselines as "unchecked" on random grids. The paper's figures (`viz/plot_transfer_results.py`, `configs/plot/transfer_results.yaml`) draw numpy-model values for baselines, and Pac-Man is compared only with hand-written rules.

## Step 1: Pac-Man in RQ2 (do first, merge within hours: plans 03, 04 and 06 wait on it)

1. In `configs/experiments/benchmarks.yaml`, add `pacman` to the `rq2` set, as in `rq1_comparison.yaml`: base `haiku_mi_pacman_long`, `dataset: pacman_transfer.csv`, `targets: "0-7"`, a label. Its training members (ids 3–5) are left out by the metrics, so 5 targets are held out; say so in the label.
2. Go through every `configs/experiments/rq2_*.yaml` and scope the cells that do not apply to Pac-Man with `only: [grids, families]`:
   - `rq2_features` (gridworld feature names; plan 06 adds the Pac-Man feature cells);
   - `rq2_training` (gridworld ids in `train_sets`, `hidden_sets`, `pool_sets`);
   - `rq2_choices`: Pac-Man's reference already uses `rotations`, so the "Best of 4 rotations" cell would just repeat it. Scope that cell to grids and families, and add a Pac-Man-only cell "Best of 24 orders each round" (`set: {pacman: ["rules.any_order=all"]}`).
   - Check how `src/experiments/base.py` treats a cell whose `set` is a benchmark map without the new benchmark. If it silently becomes the reference configuration, fix that, so such a cell is skipped on that benchmark, and add a test in `tests/test_experiments.py`.
3. Run each `rq2_*.yaml` without `--run` (plan only). Check that the Pac-Man reference resolves to the three runs in `out/results/core/haiku_mi_pacman_long/` and that no Pac-Man cell would launch a gridworld override.
4. Merge into `rule-language` and push. Do **not** launch Pac-Man runs here: plans 03, 04 and 06 launch their cells from the main checkout, and the first of them also launches the two missing reference repeats.

## Step 2: certify every baseline (PRISM only, no LLM)

- `src/run_experiment.py configs/experiments/rq1_comparison.yaml --check --baselines` fills the grid CSVs that are missing for trees (raw and features), PPO and the search without `closer`. Then `--rules open` where `ties` allows it.
- Add the PPO variant with the MiniGrid-style local view (`configs/baselines/ppo_view_*.yaml`) to RQ1 as a second PPO configuration and certify it.
- Rough cost: trees and PPO on 8 grids each, minutes. Within the machine budget in the overview.

## Step 3: best configuration per method

- In `rq1_comparison.yaml`, give every baseline and cell a `method:` key that groups configurations of one method: "1-2-3-Go! (reimplemented)" for trees over raw variables and over our features; "PPO" for features and view; "Rule-list search"; "Hand-written template"; "Hand-written maze rules"; "Ours (Haiku 5.5)"; "Ours (Sonnet 5.5)". Plan 02 adds the Pac-Man entries, plan 12 the LLM-program baseline.
- Selection rule, written in the report: per benchmark, a method's best configuration is the one with the highest mean certified per list on the **validation** targets (ties: library, then cost). On the test pass (plan 09) the configuration chosen on validation is used and never re-chosen on test results.
- The tables keep every configuration. The figures show only the chosen one, with its name in small type under the bar.

## Step 4: figures from the harness

- `src/experiments/comparison.py` writes an RQ1 figure per benchmark (random grids, without `closer`, families, Pac-Man): one bar per method, certified per list (mean over runs, range as whiskers), with the library as a marker. Families also get per-group panels. Style from `viz/theme.py`. The data must be the PRISM-certified CSVs the report uses, never the numpy model.
- **Pac-Man figure:** the worst-case crash probability of each method's frozen lists against the horizon (`worst_crash` in the transfer CSVs), the optimum (0.5511), the threshold (0.56), and 1-2-3-Go!'s published points (horizon 5: 0.5511, 25: 0.869, 200: 0.99998), marked "published, different setup". One line per method, best configuration only. Horizons 5–30 only: 50, 100 and 200 are sealed (overview, decision 1) and stay out until the test pass.
- Retire `viz/plot_transfer_results.py` and `configs/plot/transfer_results.yaml` once the harness figures replace what the paper uses. Tell plan 10 the new paths.

## Step 5: one entry page, archive the rest

- `src/run_experiment.py --index` writes `out/results/experiments/README.md`: per experiment, per cell and benchmark, runs present / repeats wanted / pending, and a link to the report. This replaces asking "which cells are missing?".
- Move the pages that predate the RQ harnesses into `out/results/archive/` with `git mv`: the qwen-era directories (`comparison_grid20*`, `legacy_grid20`, `symbolic_grid20_*`, `opt`, `smoke`), `haiku_large`, `sonnet_vs_qwen`, `rule_language_grid`, `rule_language_pacman`, `transfer_search`, `symbolic_uuv`, `template_policy.log`. First grep every reference (`configs/plot/*.yaml`, `tests/`, `docs/`, `DECISIONS.md`, `src/`) and update the paths, so that `tests/test_plot_configs.py` and the rest still pass. Leave `core/`, `ablations/`, `baselines/`, `experiments/`, `ceilings/` and `regression/` where they are. The run store and the replays read from them.
- `out/results/archive/README.md`: one line per directory, saying what it was and which decision in `DECISIONS.md` it supports.
- `out/results/README.md`: "Current results" first (RQ1, each RQ2, the Pac-Man figure, the experiments index), then the layout, then the archive.
- Update the results pointers at the top of `DECISIONS.md` and in `docs/experiments.md`.
- Record Asher's Oct 15 decisions (overview, "Asher's decisions") in `DECISIONS.md`, one line each, in the right sections:
  - Pac-Man's sealed horizons and its 0.56 threshold with the horizon figure;
  - SEAMS reports validation only;
  - new benchmark thresholds at the optimum ± 5%;
  - Haiku only until an improvement or the final method;
  - `closer` stays for now.

  Delete or shrink the **[REVIEW]** items they settle: `closer`, and Pac-Man thresholds beyond horizon 18. Mention the sealed Pac-Man horizons in `docs/experiments.md` ("Sealed test sets").

## Done when

- The experiments index lists every RQ1 and RQ2 cell with its status.
- RQ1 has no "unchecked" baseline on any benchmark it applies to.
- RQ1 has a figure per benchmark (best configuration per method) and the Pac-Man horizon figure.
- The old pages are in `archive/` with a map; tests pass (`-k "not replay"`, plus `tests/test_plot_configs.py`).
- Committed and pushed on `rule-language`.

## Files

Owns `configs/experiments/benchmarks.yaml`, the structure of `rq1_comparison.yaml`, `src/experiments/comparison.py`, `src/run_experiment.py`, `viz/`, `out/results/README.md` and `out/results/archive/`. Step 1 touches every `rq2_*.yaml` (only `only:` lines and the one Pac-Man cell).

## Report back

The experiments index (which cells are missing), the newly certified baseline numbers, the figure paths, and anything moved that another plan referred to.
