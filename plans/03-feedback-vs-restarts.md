# 03 · Feedback against pure restarts, at equal cost

Read `plans/00-overview.md` first. Needs plan 01 step 1 (Pac-Man in the `rq2` benchmark set) before the Pac-Man runs.

## Goal

Settle, on the same plot and at equal cost, whether refining with the verifier's feedback beats restarting from the initial prompt with no feedback. Use every benchmark: random grids, families and Pac-Man. This is the core of the story ("the verifier helps refine it"), so the comparison is fixed before the new runs.

## What exists

`configs/experiments/rq2_feedback.yaml` (RQ2a) has four cells:
- feedback with no restarts (core, 5 runs);
- feedback that restarts when a round gains less than 0.05;
- feedback that restarts after 2 rounds without gain;
- restarts only (3 runs).

At 5 rounds, feedback is ahead of restarts only on grids (6.25 against 5.42) and level on families (9.67 against 9.78). Restarts only spend about a third of the output tokens, so "equal rounds" flatters feedback. `DECISIONS.md` requires equal cost on the same plot (Marsha). Keep-best makes any budget curve rise, so a rising curve is no evidence on its own.

## Steps

1. **Check what "restarts only" sends** (`planner.retry=always`). Read a few prompts in a restarts-only run's `llm_tasks.jsonl`: every round must be the initial prompt, with nothing from earlier rounds. The verifier still resolves `any` into the best order and keep-best still applies; write that down in the report, because it is the same in both arms.
2. **Budget curves from existing runs (no LLM, PRISM only).** New tool `src/budget_transfer.py <run dirs>`:
   - Per training set and round k, the list keep-best holds after round k: the last improved iteration ≤ k in `outputs/sample_*.json` `iterations` (its `rules` are already resolved).
   - Its cumulative cost: prompt and output tokens, and dollars at the prices in the experiment file.
   - Its certified count on the benchmark's validation targets, through `src/transfer.py`'s certification, cached by a hash of the rule list (most rounds keep the same list).
   - Writes `<run>/budget_transfer.csv`. Round 1 is the LLM's first answer with no feedback: label it, because it doubles as an "LLM alone" point.
   - Test on a tiny saved run (`tests/fakes.py`).
3. **Figure** (`viz/plot_budget_transfer.py` + `configs/plot/budget_transfer.yaml`, wired into the `rq2_feedback` report): certified per list against dollars per training set. One line per cell, mean over runs with a band, all cells on the same axes, one panel per benchmark. Also a second row for training sets solved against dollars.
4. **Fix the comparisons before launching.** Write them into the experiment file's `question` and the report. Suggested:
   - (a) certified per list at feedback's mean spend after 5 rounds;
   - (b) the same at half that spend;
   - (c) training sets solved at feedback's spend.

   Each is computed on grids, families and Pac-Man. Report the difference with a bootstrap interval over runs and Holm correction, as `out/results/haiku_large` did. No new comparisons are added after seeing the results; extra looks are labelled exploratory.
5. **New cells and repeats** (Haiku; launch from the main checkout with `src/run_experiment.py configs/experiments/rq2_feedback.yaml --run --check`):
   - "Restarts only, equal cost": `planner.retry=always`, `planner.max_rounds=16`. About 3.4 times as many rounds matches feedback's tokens on both gridworld benchmarks: work out the factor from the existing runs and round up. The curve is read at equal cost, so overshooting is harmless.
   - Restarts only and the core reference up to **5 repeats** each. The hybrids stay at 3.
   - Pac-Man: the reference (2 more repeats of `haiku_mi_pacman_long`), restarts only, and restarts only at equal cost, 5 repeats each. Pac-Man rounds are cheap.
   - Cost: a few dollars in Haiku tokens in total. PRISM time dominates, since the verifier checks 24 action orders a round on every member. Keep to the machine budget and spread the launches.
   - Reuse "more rounds" (`rq2_training`, 8 and 10 rounds) for a longer feedback curve: the store finds those runs.
6. **Report.** Regenerate `rq2_feedback`, with the figure and the planned comparisons first. If feedback does not beat restarts at equal cost somewhere, say so plainly and tell plan 09 and plan 10. The story then leans on the verifier's other roles (resolving the order, keep-best, certifying), and the paper must not claim feedback helps there.

## Done when

- Every rq2_feedback cell has its repeats on all three benchmarks.
- The budget figure is in the report; the planned comparisons are computed and stated.
- `budget_transfer.csv` exists for every run in the experiment.
- Tests pass; committed.

## Files

Owns `configs/experiments/rq2_feedback.yaml`, `src/budget_transfer.py`, `viz/plot_budget_transfer.py`, `configs/plot/budget_transfer.yaml`. Coordinate with plan 01 if the report code in `src/experiments/ablation.py` needs a hook for the figure; keep it to one call.

## Report back

The three planned comparisons per benchmark (difference, interval), the figure path, and one sentence on what it means for the story.
