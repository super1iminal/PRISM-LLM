# Transfer search: finding a foundation for rules that generalize

Brief for the next working session (written Oct 13). Read `DECISIONS.md` (top: story and paper target; "Rule language", "Gridworld transfer", "Multi-instance training") and `docs/semantics.md` ("Training sets") first.

## Goal

Find a combination of changes under which **Haiku 5.5 solves multi-grid training sets and its frozen rules certify on held-out grids**. Our "why LLMs" claim is generalized rules over many examples. First find any combination that works, even if it leans on hard-coding or on similar examples; then trim it back and build up from there.

Working style (Asher): many cheap Haiku runs **in parallel**, iterate quickly ("gradient descent" over ideas), no limit on the number of Haiku runs. Sonnet only once Haiku shows something works. The LLM-written vocabulary (Nicholas's `rule-language-generation`) is **parked**: the domain provides its vocabulary, and two moving parts at once would be too confusing.

## Where things stand

- Branch `rule-language` (checked out in the main checkout; no worktrees): rule language (constants, features, `any`, general mode), `src/transfer.py`, `viz/plot_transfer.py`, multi-instance training (`run.train_sets`), and Nicholas's scheduler work (merged Oct 13: the planner yields PRISM calls; `event` is the default scheduler for symbolic runs).
- Branch `rule-language-generation` (on `origin` as a backup; from Nicholas's fork `script-jpg/PRISM-LLM`): the LLM-written vocabulary, plus two Windows test fixes and level-0 conditions. Not merged, not run.
- Results so far (`DECISIONS.md` has the numbers):
  - Single-grid training: rules overfit to their grid (Sonnet: 9 of 380 other pairs certify). Even with only the literals 0 and 1, rules identify instances through offsets (`x = G2R & y = G2C - 1`) and relations between constants (`G2R = G3R + 1`).
  - With the shortest-path feature `closer`, Haiku writes nearly the same generic greedy rule set every time; it certifies the grids where greedy-and-safe works (9–14 of 20) and none of the others. Without `closer`, nothing transfers.
  - Multi-instance training (4 sets of a 9x9, a 10x10 and an 11x11 grid from grids 0–11; test grids 12–19 never trained on): Haiku solves 0 of 12 sets over three runs, with or without `closer`. On the test grids per rule set: 0 of 8 without `closer`, 5.3 of 8 with it. Sonnet (one run) solves 1 of 4 sets, 1.75 of 8 test grids (its single-grid comparison used vocabulary v1: confounded).
  - Grids certified by generic `closer` rule sets: 8, 10, 11, 13–18 (one rule set also 4–7 and 19). The 9x9 grids (0–3) are never certified by a generic rule set: worth understanding first.
  - A hand-written template certifies 10 of 20 (`domains/gridworld/data/template_policy.py`).

## Protocol for every idea

- Dataset `grid_20_large.csv`; training sets from grids 0–11; **test set grids 12–19** (`src/transfer.py --targets 12-19`, or filter the full transfer CSV on `target >= 12`).
- Haiku 5.5 on OpenRouter (Anthropic endpoint), thinking on, 16k output cap, 5 rounds unless the idea is about rounds; three unseeded repeats per condition (`src/run_ablation.py <conditions> --unseeded 3`).
- Report per condition: training sets solved; test grids certified per rule set (of 8); requirements met on test grids (partial credit); tokens per set.
- Baselines (already run, three repeats): `haiku_mi_grid_large` (v2, no `closer`) and `haiku_mi_grid_large_closer`.
- Each idea is a config flag with a condition file; the default loop must stay byte-identical (`tests/test_scheduler.py` replays). Run conditions in parallel (one `run_ablation.py` process per condition); a Haiku multi-instance run costs about $0.03 and takes 5–10 minutes.
- Useful infrastructure to build first: a launcher that runs a list of conditions in parallel and then their transfer checks, and one summary table (or `ablation_summary`-style page) over all conditions on the test-set metrics.

## Ideas (from the Oct 13 discussion)

0. **Diagnosis, no LLM.** For the best generic rule sets, which requirements fail on which grids, and why: slips into goals not yet due (sequence requirements), timing against the moving obstacle, or pockets where a memoryless policy is trapped. Also a ceiling: how many grids can any rule set over the vocabulary certify? This decides which levers can matter.
1. **Group training grids by properties** (Asher; another paper did this): the sets may simply be too different to solve at once. Cluster grids by structure (size `n`, goal geometry and order, obstacle density, the moving obstacle's position relative to the goal path, `BFS_steps`, or whether the generic greedy policy passes) and train one rule set per group; test within and across groups.
2. **Select for generality in the loop:** train on k grids, keep-best by a validation set of other grids (shown to the LLM only as a summary), or rotate the training grids each round so per-grid patches stop paying off.
3. **Feedback in feature space:** blame and hotspots named by feature valuations ("rule 2 loses most where `risky` holds and goal 2 is not reached") instead of coordinates, which invite coordinate patches.
4. **Close the patch language:** forbid relations between constants (`G2R = G3R + 1`) and offsets on constants (`G2C - 1`); allow only a variable compared with a constant. Optionally cap the rule count or prefer shorter rule sets in keep-best.
5. **Prompting:** say the rules will be certified on unseen, larger grids and must not special-case one; ask for a strategy in words first, then rules (as Stein et al.).
6. **Generalize from solved examples:** solve grids separately (easy), then give the LLM the working rule sets and ask for one rule set capturing what they share; verify on all training grids and refine.
7. **More search:** best-of-N initial rule sets on a training set (keep the one passing the most training grids); more rounds for training sets; training-set size 1 / 3 / 6 / 12 (later on Sonnet).

Combinations are expected: the target is the first combination that makes Haiku solve training sets and certify test grids, then ablate it down.

## Gotchas

- Windows: PRISM's JVM outlived timeouts before Oct 8 (fixed: `PrismRunner` kills the process tree); temporary model files have CRLF line endings (the PRISM-call recording normalizes them).
- `event` scheduler: the first live runs with it are this work's runs; `run.scheduler=threads` compares timings with earlier runs.
- Joint LP queries on 13x13 grids can take up to `prism.timeout_s` (15 min); a timeout counts as undecided.
- Don't `cd` into output directories (Windows locks); commit parquet, `outputs/` JSON and transfer CSVs, not logs.
- OpenRouter key: repo-root `.env` (`set -a && . ../.env` from `PRISM-Guided-Learning/`); never print or commit it.
