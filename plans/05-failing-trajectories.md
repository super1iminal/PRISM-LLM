# 05 · Failing trajectories in the feedback

Read `plans/00-overview.md` first. Appends cells to plan 04's experiment file (`rq2_verifier_output.yaml`).

## Goal

Find out whether showing the LLM how its rules fail helps the rules transfer. Then make the trace as useful as possible, behind config keys, and test the variants on random grids, families and Pac-Man.

## What exists

`feedback.trajectory: true` (`src/core/analysis.py`, `failing_trace`): one likely failure.
- Its start is the initial state. It follows the scheduler the verdict is about and takes the successor most likely to lead to failure. It stops once failure is certain, at a repeated state, or after 20 steps.
- Each step shows the state, the action, the deciding rule and the failure probability.
- It is shown for the focus member's worst requirement.
- With 3 runs, it scored 5.92 against 6.25 on grids and 7.33 against 9.67 on families (crossing 0.0).

## Steps

1. **Diagnose before building.** Read the trace sections of the prompts in the existing trace runs (find them with the store: `rq2_verifier_output.yaml`'s "Plus a failing trace" cell), and check the next answers against them.
   - Is the trace informative, or mostly forced steps?
   - Does it show raw coordinates? In general mode these invite instance-specific exceptions, which cannot transfer.
   - Does the next answer add rules that single out the trace's cells?
   - Why did crossing drop to 0?

   Write the answers as a short note; it decides which variants are worth building.
2. **Variants**, each a value of one new key. Keep `feedback.trajectory: true` meaning exactly what it means now, so recorded runs keep their identity. For example, add `feedback.trace: likely | features | critical | per_requirement`, used only when `trajectory` is true, with `likely` as the default. Build only those the diagnosis supports:
   - `features`: steps named as `feedback.space=features` names states (the features that hold per action, the goal flags, no positions), plus the deciding rule's index and text.
   - `critical`: also mark the step where the choice mattered most: the largest gap between the action taken and the best action at that state, from the Q-values `failing_trace` already computes. The LLM then sees which rule to change, not only a path.
   - `per_requirement`: one short trace for each failing requirement (at most 3), instead of only the worst.
   - Pac-Man: check that the trace reads naturally there (Pac-Man's and the ghosts' positions, step by step). Forced corridor moves should be collapsed into one line.
3. **Tests.**
   - Render each variant on the tiny grid (`tests/fakes.py`) and on a Pac-Man horizon-8 instance.
   - Check that with the defaults every prompt is byte-identical (`tests/test_scheduler.py` replays).
   - Update the "Failing trace" bullet in `docs/semantics.md`, and add the row to `docs/config.md`.
4. **Runs** (Haiku, from the main checkout, after merging). Append the variant cells to `rq2_verifier_output.yaml` with descriptive names ("Plus a failing trace in features", …). Launch 3 repeats on grids, families and Pac-Man; then 5 for the best variant and the reference.
5. **Report.** Regenerate RQ2b. A short reading: whether any trace beats full feedback, and on which benchmark. If none helps, say so; the cells stay as evidence.

## Done when

- The diagnosis note is written (in the report or as the docstring of the new key).
- The variants are merged with tests and docs; runs done; RQ2b regenerated; committed.

## Files

Owns `src/core/analysis.py` (`failing_trace` and its rendering) and the trace template in `src/core/templates/`. Shares `src/config.py`, `configs/run/default.yaml`, `docs/config.md` and `docs/semantics.md` with plan 07: one block each, rebase before merging.

## Report back

The diagnosis in three lines, the variant numbers, and the recommendation for plan 09 (in the core or not).
