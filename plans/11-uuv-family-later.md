# 11 · A UUV family (later: not before SEAMS)

Read `plans/00-overview.md` first.

## Goal

A family of UUV (pipeline-inspection AUV, Päßler et al., iFM 2023) instances where no simple controller passes every member and members differ in ways finite-family tools cannot handle. That would make UUV a real third case study for SEAMS-style adaptation, or for ICAPS.

## Why it waits

The failure-rate family is trivial: the paper's own `stay` controller meets every requirement across a 0.5×–3× sweep in both scenarios (`docs/positioning.md`, Check 2). UUV has been set aside since. The bare MDP must keep reproducing the paper's numbers (`tests/test_uuv.py`).

## Steps (PRISM only until step 4)

1. **Candidate families**, built with a generator like `domains/uuv/data/failure_sweep.py`:
   - visibility ranges (the altitude bands move);
   - deadlines;
   - inspection lengths;
   - and combinations.

   Requirements stay the paper's; thresholds come from each member's best value over all controllers plus the paper's slack (`src/calibrate.py`).
2. **Non-triviality check per family.**
   - The paper's controllers and `domains/uuv/data/reference_policies.json` must each fail some members.
   - Some single rule list must certify every member checked: hand-written, or found by a small search, so transfer is possible.
   - Every member must be jointly solvable. The joint query is undecided for step-bounded requirements, so use per-requirement optima and say so.
3. **Report the check to Asher, and decide together.** Settle the open UUV items in `DECISIONS.md` at the same time: the energy thresholds, the description of energy costs, and the two ideas from the `optimization` branch.
4. **Only then:** datasets (train, validation, sealed test), a core-configuration condition for UUV, Haiku runs, the generic baselines (plan 02), and an RQ1 benchmark.

## Files

`domains/uuv/data/` (generators and datasets) only, until step 4.
