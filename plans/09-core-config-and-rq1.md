# 09 · The core configuration, RQ1, the test pass, and the hand-off to Rajan

Read `plans/00-overview.md` first. Starts once plans 03, 04 and 06 have reported (around Oct 19). Plans 05 and 12 are folded in if ready.

## Goal

RQ1 compares *our best mechanisms and features* with the baselines:
1. pick that configuration from the RQ2 results by a rule fixed in advance;
2. freeze it as the core;
3. run RQ1 on validation;
4. when Asher unseals, run the single test pass;
5. hand experimentation to Rajan.

## Step 1: write the selection rule now, before the new RQ2 results come in

Suggested; commit it to the top of `rq1_comparison.yaml` as a comment, and to `docs/experiments.md`.
- A mechanism (one RQ2 cell's change) **enters the core** only if both hold:
  - its mean certified per list, on validation, summed over the benchmarks with each benchmark normalized by its number of held-out targets, beats the current core;
  - on no benchmark is the lower end of its 90% interval below −1 target.
- If it costs more than twice the core's dollars per set, it must also win at equal cost, read off plan 03's budget curves.
- **Solver choices** (any `rules.any_order` value, hidden members, polishing) may enter while Asher allows PRISM to choose. Each keeps its feedback-only counterpart reported next to it.
- **Feedback stays in coordinates** (`feedback.space: states`). Feature space was tested (plan 04), lost, and is out of the core (Asher, Oct 15).
- **Memory** (plan 07) never enters this core. If Asher wants it, it becomes a separate `-memory` RQ1 cell.
- **Features:** `closer` stays in the core for now (Asher, Oct 15). `grids_noclose` stays as a second random-grid benchmark, ready for a later attempt without it.

## Step 2: combine and confirm

- If two or more mechanisms qualify, one combined cell with 5 repeats on validation on every benchmark. Adopt it only if it beats the current core by the same rule.
- If none qualifies, the current core stays. That is a fine outcome: say so.

## Step 3: freeze

- The frozen configuration's condition files are the RQ1 cells: one per benchmark (grids, grids without `closer`, families, Pac-Man, and plan 08's domains once they exist).
- Runs go to `out/results/core/`, 5 repeats. A configuration that did not change keeps its existing core runs.
- Write the configuration in words at the top of the RQ1 report: rule language, features, `any` resolution, training sets, rounds, prompt, model and provider.
- **Sonnet** (Asher's decision 5): none until we truly think the configuration is an improvement or the final method. Only then ask Asher which cells to run. Until then they stay `launch: false`, and the existing Sonnet runs of the old configuration stay in RQ1 labelled with their configuration.

## Step 4: RQ1 on validation

- `src/run_experiment.py configs/experiments/rq1_comparison.yaml --run --check --baselines`, then the figures from plan 01.
- Every method's configuration is picked on validation (plan 01's rule) and recorded in `rq1_comparison.yaml` as `chosen:`. That makes the test pass mechanical.

## Step 5: the test pass (only when Asher unseals; not for SEAMS, which reports validation only)

- A separate experiment file `configs/experiments/rq1_test.yaml` with the sealed targets:
  - `grid_test_large.csv`;
  - the 15x15–20x20 family members;
  - Pac-Man's sealed horizons (overview, decision 1);
  - plan 08's sealed sizes.
- Only the `chosen:` configurations.
- Our frozen lists come from the existing core runs; no new LLM runs. Certify with `--unseal`, once. No configuration changes after looking.
- Report: `out/results/experiments/rq1_test/REPORT.md` and its figures. Plan 10 or the ICAPS draft takes them from there.

## Step 6: hand-off to Rajan

- A "Running the experiments" section in `docs/experiments.md`, stateless:
  - setup (venv, PRISM, the OpenRouter key variable);
  - launching from the main checkout;
  - the machine budget;
  - the sealed and Sonnet rules;
  - how to add a cell, a condition, a baseline or a domain;
  - how to regenerate every report;
  - where results go;
  - the `-memory` rule.
- Check it from a fresh clone: venv, then regenerate one RQ2 report and the RQ1 report from committed runs, with no LLM.
- Remove the **[REVIEW]** items Asher has settled from `DECISIONS.md`. Add the core configuration as one decision line pointing to the RQ1 report.

## Done when

The selection rule is committed, the core is frozen, RQ1 is regenerated on validation with `chosen:` recorded, and the hand-off section is checked from a clean clone. The test pass is done only when Asher says.

## Files

Owns the `cells` of `rq1_comparison.yaml`, `rq1_test.yaml`, the core condition files, and the hand-off section of `docs/experiments.md`.

## Report back

Which mechanisms qualified and why, the frozen configuration, the RQ1 validation table, and the hand-off check.
