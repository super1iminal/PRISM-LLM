# 04 · What the verifier tells the LLM: feedback in feature space first

Read `plans/00-overview.md` first. Plan 05 (failing trajectories) appends its cells to the same experiment file and builds on this plan's feature-space rendering.

## Goal

**Asher (Oct 15): feedback should be in feature space, not coordinates.** States should be named by the features that hold, the goal flags and other instance-independent quantities, never by positions.

The working hypothesis (from RQ2a): feedback helps the LLM solve the training instances (instance-only runs; training sets solved), but not the held-out ones. Blame and hotspots name states by coordinates, which invites patches that fix one training instance and cannot transfer.

So this plan:
1. makes feature-space feedback complete: every place the feedback names a state;
2. runs it with the core settings on random grids, families and Pac-Man;
3. measures whether it reduces instance-specific patches;
4. then completes the rest of RQ2b.

## What exists

- `feedback.space: features` (`src/core/prompts.py`, `PromptBuilder._state`): blamed and uncovered states are named by their boolean variables, the state features that hold, and per action the action features that hold. States that read the same are merged.
- Older runs with the choices left open (`out/results/ablations/haiku_ts_features/`, 3 runs) certified 6.25, 4.25 and 6.25 on random grids, about level with coordinates in that setting (5.83). It has never run with the core settings (`rules.any_order: all`): RQ2b's "States named by their features" cell has 0 runs.

RQ2b (`configs/experiments/rq2_verifier_output.yaml`), validation, certified per list:

| Cell | Grids | Families | Runs |
|---|---|---|---|
| Full feedback (core) | 6.25 | 9.67 | 5 |
| Blame by one-step regret | 5.83 | 7.78 | 3 |
| Random blame (placebo) | 6.42 | 7.33 | 3 |
| No blame | 6.75 | unchecked | 1 |
| Results table only | – | – | 0 |
| States named by their features | – | – | 0 |
| Plus a failing trace (plan 05) | 5.92 | 7.33 | 3 |

## Steps

1. **Audit: where does the feedback name a state or a position?** Read a REFINE and an EXTEND prompt with `feedback.space=features` on each benchmark: render them with a scripted backend, or take the existing `haiku_ts_features` prompts. List every place a position can appear:
   - the blame per rule and its hotspots;
   - the uncovered hotspots;
   - the per-member summary;
   - the failing trace (`feedback.trajectory`, which renders `policy_valuation`, i.e. coordinates);
   - anything else.

   The problem description and the training instances' maps stay: that is the task, not feedback.
2. **Make it complete.** In feature mode, nothing in the feedback names a position:
   - Integer variables that are positions (`x`, `y`, `xP`, `yP`, ghost positions, `obs_idx`) are left out. Instance-independent quantities may stay, such as Pac-Man's `moves_left` or a heading. State features with integer values are shown as `name=value`; check that `_state` handles them, since today it assumes boolean features.
   - The failing trace in feature mode names each step the same way. Do a minimal version here: plan 05 builds its variants on top, and owns `failing_trace` itself.
   - **Run identity:** if any of this changes what `feedback.space: features` did for the old `haiku_ts_features` runs, add a new value (for example `feedback.space: abstract`) rather than changing `features`, so the store never mixes them. Say which in `docs/config.md`.
   - Tests:
     - feedback prompts in feature mode contain no position variable, on the tiny grid and on a Pac-Man horizon-8 instance;
     - defaults are byte-identical (replays in `tests/test_scheduler.py`).

   Merge into `rule-language`.
3. **Smoke** (Haiku, one training set per benchmark, `--set run.limit=1`, into a scratch directory): read the prompts of rounds 2–3 and check that the feedback reads sensibly without positions.
4. **Runs** (Haiku, from the main checkout, `src/run_experiment.py configs/experiments/rq2_verifier_output.yaml --run --check`):
   - **First:** the feature-space cell, 5 repeats on grids, families and Pac-Man, plus the core reference up to 5 on Pac-Man. Plan 03 may already have launched those; the store will not repeat them.
   - **Then:** results table only and no blame, 3 repeats on each benchmark; random blame up to 5.
   - Keep to the machine budget (plans 01–03 also use PRISM): `--max-parallel 2`, and check `Get-Process python` before each launch.
5. **The hypothesis, measured (no LLM).** From the final and per-round rule lists, count instance-specific rules per list under coordinate feedback and under feature feedback (and under restarts only, from plan 03's runs):
   - conditions comparing a position variable with a constant or with another position;
   - offsets from constants (`G2C - 1`);
   - relations between constants.

   Use the rule parser, not regexes over text. Also: how often the next answer changes the rules the feedback blamed, against those it did not. One table in the report.
6. **What goes into the core.** Asher's preference is feature space. It enters the core configuration (plan 09) unless it is clearly worse: the lower end of its 90% interval below −1 target on any benchmark. In that case report to Asher before plan 09 decides. Add a line to `DECISIONS.md`.
7. **Report.** Regenerate RQ2b. Three lines: feature space against coordinates per benchmark, the instance-specific-rule counts, and the other cells.

## Not in this plan

Simulation feedback instead of PRISM (overview, item 8): not now.

## Done when

- Feature-space feedback names no position anywhere and is merged with tests.
- Its cell has 5 repeats on all three benchmarks; the other cells have theirs; nothing is unchecked.
- The hypothesis table is in the report; committed after every step ("[plan 04] step K: ...").

## Files

Owns `configs/experiments/rq2_verifier_output.yaml` (plan 05 appends cells) and the feature-space rendering in `src/core/prompts.py` and its templates. The trace's feature rendering is minimal here; plan 05 owns `failing_trace` and its variants: keep the change to one function, and rebase before merging.

## Report back

Feature space against coordinates per benchmark (difference and interval), the instance-specific-rule counts per feedback kind, and anything the audit found.
