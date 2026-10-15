# 06 · Feature engineering in the domain

Read `plans/00-overview.md` first. Needs plan 01 step 1 before the Pac-Man runs.

## Goal

Measure how much transfer depends on the features each domain provides, for our method and for the baselines, on random grids, families and Pac-Man. Separate "the features carry the information" from "the LLM uses it".

`closer` stays in the configuration we claim for now (Asher, Oct 15), but removing it may be tried again later. So the no-`closer` numbers here are the baseline for that later attempt, not a vote on the core.

## What exists

RQ2d (`configs/experiments/rq2_features.yaml`):
- Only two cells have runs: all features (core) and no `closer` on grids (3.50 against 6.25).
- The other cells have no runs: no `closer` on families, local features only, no `risky`, no `obstacle`, `hazard` split and unsplit, and no features.
- Pac-Man's core hides the maze-distance features `closer_g0` and `closer_g1`.
- Baselines show the features carry the information: on crossing, trees over our features certify 19 of 20 and PPO with features gets 19 of rooms, where our lists get 4.6 and 8.4.

## Steps

1. **Plan check.** Run `src/run_experiment.py configs/experiments/rq2_features.yaml` (plan only). Confirm each cell's `rules.hidden_features` list is right for each benchmark: the grid conditions hide `patrol` and `early` already, and the family conditions hide `hazard`. A cell that hides a feature its base already hides is the reference again; fix such lists.
2. **Pac-Man cells.** Add them to `rq2_features.yaml` with `only: [pacman]`. Pac-Man's base hides `closer_g0` and `closer_g1`, so overrides restate the whole list:
   - "Plus maze-distance features": `rules.hidden_features=[]`;
   - "Maze distance instead of rows and columns": `[nearer_g0, nearer_g1]`;
   - "No `moves_left`": `[closer_g0, closer_g1, moves_left]`;
   - "No features (positions, headings, steps only)": every feature. Check that `rules.extended` still makes sense with none.
3. **Launch** (Haiku, from the main checkout): every missing cell, 3 repeats, on every benchmark it applies to.
   - That is about 40 gridworld runs; they are cheap in tokens and heavy in PRISM time, since 24 orders are verified per round, so spread them over the machine budget.
   - Then 5 repeats for "no `closer`", because `closer` is the open question.
4. **The same ablation for the baselines (no LLM).** Trees over our features and the rule-list search, each with the same features hidden: configs `configs/baselines/*_hide_<feature>.yaml`, PRISM-certified.
   - This is the "is the information there" column: if a baseline keeps its score without a feature and ours drops, the LLM, not the vocabulary, is the limit.
   - Put a table in the report with one row per feature and four columns: ours, trees, search, and PPO where cheap.
5. **New features: only with Asher's OK, and only by this protocol.** Candidates such as a doorway feature for rooms, or a dead-end feature, are not built in this plan unless Asher says so (the same open question as `closer`). If he does:
   - the feature is computed from the instance's layout, never from PRISM;
   - it means the same thing on every instance;
   - it is hidden for every older condition (`rules.hidden_features`, or `identity.json` corrections for finished runs) so their prompts and identities do not change;
   - it is first tried with the trees and the search (no LLM).
6. **Report.** Regenerate RQ2d with the baseline table, plus three lines: which features matter, whether the vocabulary or the LLM is the bottleneck per family, and the recommendation on `closer` for plan 09 and for Asher.

## Done when

- Every RQ2d cell has its repeats on every benchmark it applies to.
- The baseline feature table is in the report; committed.

## Files

Owns `configs/experiments/rq2_features.yaml` and `configs/baselines/*_hide_*.yaml`. Pac-Man feature code in `domains/pacman/domain.py` only if a cell needs a change (none should); plan 02 only appends dataset rows there.

## Report back

The feature table (ours against the baselines), the `closer` numbers at 5 repeats, and the bottleneck reading per family.
