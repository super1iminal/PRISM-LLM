# 04 · What the verifier tells the LLM

Read `plans/00-overview.md` first. Needs plan 01 step 1 before the Pac-Man runs. Plan 05 (failing trajectories) appends its cells to the same experiment file.

## Goal

Complete RQ2b (`configs/experiments/rq2_verifier_output.yaml`) on random grids, families and Pac-Man, with enough repeats to read the cells that matter for the story. The question: which parts of PRISM's output help the LLM write rules that transfer?

## What exists (validation, certified per list)

| Cell | Grids | Families | Runs |
|---|---|---|---|
| Full feedback (core) | 6.25 | 9.67 | 5 |
| Blame by one-step regret | 5.83 | 7.78 | 3 |
| Random blame (placebo) | 6.42 | 7.33 | 3 |
| No blame | 6.75 | unchecked | 1 |
| Results table only | – | – | 0 |
| States named by their features | – | – | 0 |
| Plus a failing trace (plan 05) | 5.92 | 7.33 | 3 |

The placebo matching mass blame on grids is the result to test properly.

## Steps

1. **Find out why two cells have no runs.** Look in `out/results/ablations/` for started or failed directories of `planner.feedback=table` and `feedback.space=features`, and in `out/results/experiments/rq2_verifier_output/*.log`. Then smoke-test each with one training set on grids (`src/run_symbolic.py --condition haiku_ts_order --set planner.feedback=table --set run.limit=1 --out <scratch dir>`), and the same for `feedback.space=features`, on families and Pac-Man too. Fix what breaks: these modes predate training sets and `any_order`. Any fix that changes what a run does needs a config key. If a mode cannot apply to Pac-Man, scope the cell with `only:`.
2. **Certify the unchecked run** ("No blame" on families): `--check`.
3. **Launch** (Haiku, from the main checkout, `src/run_experiment.py configs/experiments/rq2_verifier_output.yaml --run --check`):
   - Results table only, feature-space feedback, no blame: 3 repeats on each benchmark.
   - **5 repeats** for the three cells the story turns on: full feedback (core), results table only, random blame.
   - Pac-Man: every cell that applies. The reference repeats may already have been launched by plan 03; the store will not run them twice if both launch from the main checkout.
4. **A mechanism measure (no LLM).** From each run's iterations: how often the next answer changes or removes the rules the feedback blamed, against the rules it did not blame. If the LLM ignores the blame, the placebo result is explained. Put it in the report as one table.
5. **Not now (overview, item 7, recommended against for SEAMS): simulation instead of model checking.** Skip this step unless Asher asks for it. If he does, write a design note first, before any code: `feedback.source: simulation`.
   - The loop estimates each requirement from N simulated episodes of the rules (uncovered states take a uniform random action) on plan 02's explicit model.
   - The feedback reports the estimates and one failed episode.
   - The final list is still certified by PRISM.
   - Open points for the note: what keep-best, the branch and `any_order` use (estimates), and the token parity.

   This is the most direct test of "the *verifier* helps".
6. **Report.** Regenerate RQ2b. A short reading: which outputs matter, whether the placebo still matches at 5 repeats, and what that means for the story. Pass it to plans 09 and 10.

## Done when

- Every cell has its repeats on every benchmark it applies to, and nothing is unchecked.
- The mechanism table is in the report; committed.

## Files

Owns `configs/experiments/rq2_verifier_output.yaml` (plan 05 appends cells). Fixes for the table and feature-space modes live in `src/core/prompts.py` and templates: keep them minimal, behind config keys, and run the replays (`tests/test_scheduler.py`) before merging.

## Report back

The table above, filled in; the mechanism numbers; any mode that had to be fixed, and why.
