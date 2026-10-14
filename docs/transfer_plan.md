# Transfer search: plan

How to carry out `docs/transfer_search.md` (written Oct 13). The phases are ordered by dependency. The work without an LLM (diagnosis, baselines) runs alongside the Haiku batches.

## Decisions (Asher, Oct 13)

- **Deadline:** SEAMS (Oct 23) is a stretch goal; ICAPS (Dec 7) is the fallback. Gridworld and Pac-Man come first; the benchmark-set families and PAYNT follow if time allows.
- **Validation and test:** grids 12–19 are the validation set. A sealed test set (below) is touched only for the final configuration and the baselines.
- **"Working"** means Haiku solves at least half of the training sets and each frozen rule set certifies at least 7 of 8 validation grids on average. Then trim.
- **Gridworld families:** the random grids of `grid_20_large` share no structure, so "a family" there is weak. Generated families (below) are the main gridworld setting; the random grids stay as the harder, unrelated case.
- **The rule-list search** without an LLM is a baseline from the start.

## What the existing runs already say

These come from the transfer CSVs of every gridworld run on `grid_20_large` (14 runs). "Held-out" means a rule set that was not written for that grid.

- **No rule set has ever certified grids 0–3 or 12 held-out.** Grids 1–3 were certified only by rule sets written for them; grid 0 never. Grids 9 and 19 were certified rarely (7 and 3 held-out certifications, against 48–64 for each of 8, 10, 11 and 13–18).
- **Every current training set contains one of grids 0–3** (`[0,4,8]`, `[1,5,9]`, `[2,6,10]`, `[3,7,11]`; set 1 also has grid 9). So far a training set could only be solved by a rule written for that grid. Haiku's 0 of 12 solved sets may be a property of the sets rather than of Haiku. If so, the loop rewards exactly the grid-specific patches that fail on held-out grids. Phase 1 tests this first.
- **On the test grids 12–19 the bar is 6 of 8.** Generic `closer` rule sets certify 13–18. The hand-written template certifies 13–15 and 17–19 (it is trapped on 16). Together they cover 7 of 8; nothing transfers to grid 12, where goal 3 lies next to goal 1. Beating both needs grid 12, or 16 and 19 at once. That is a question of which policies the rules can express more than of prompting.
- **Every generic Haiku rule set uses `any`.** The worst case then takes the worst of the allowed actions, while the template commits to a fixed direction order. Part of the gap may come from `any`, and that is cheap to measure.
- **Eight test grids and three repeats make a coarse signal.** One grid is 12.5 points, and six of the eight almost always pass or always fail. After days of iteration on grids 12–19 they are a validation set, not a test set.

## Protocol additions

- **Validation and test.**
  - Grids 12–19 stay the iteration metric, under the name "validation".
  - A sealed test set is used only for the final configuration and the baselines: the 30 grids of sizes 9–13 in `grid_100_balanced.csv` that are not in `grid_20_large.csv` (6 per size), plus new 15x15, 17x17 and 20x20 grids from `datagen.py`, which only certify frozen rules (no LLM runs on them).
  - Optionally, move part of the 30 spare grids into validation to give a finer signal.
- **Parallel runs.** The brief overrides CLAUDE.md's one-run-at-a-time rule here. Wall-clock times are then not comparable, so tokens are the cost measure. PRISM processes are capped across runs (31 GB of RAM, `prism.java_max_mem: 4g` each).
- **Haiku settings** as in the brief: thinking on, 16k output, 5 rounds, three unseeded repeats. Sonnet only after asking.

## Phase 0: infrastructure (no LLM)

1. **`transfer.py --targets 12-19`** (ranges and lists); the default stays all targets.
2. **Launcher `src/run_search.py <conditions…> --unseeded 3 [--targets …] [--max-parallel K]`.** It runs one `run_ablation.py` process per condition, then `transfer.py` on each finished run, then the summary. Runs that already have a transfer CSV are skipped, so a batch can be extended.
3. **PRISM slots across processes.** `PrismRunner` takes a slot (a lock file in `out/.prism_slots/`, limit from an environment variable the launcher sets) before starting a JVM, so 20 parallel runs do not start 80 JVMs. The limit is unset by default, so a single process behaves as now.
4. **Summary `viz/transfer_summary.py configs/plot/transfer_search.yaml`** writes `out/results/transfer_search/SUMMARY.md` and a figure. Per condition, as the mean and range over repeats:
   - training sets solved;
   - validation grids certified per rule set (of 8);
   - requirements met on validation grids;
   - input and output tokens, and dollars, per set.

   Plus a grid × condition table of how often each validation grid is certified. Baselines are rows in the same tables.
5. **Baseline interface `src/baselines/`.**
   - `base.py`: a baseline is fitted on training instances and returns, for each target, a rule set or a PRISM policy module. Certification uses `PolicyVerifier`, as in `transfer.py`.
   - One module per method, configs in `configs/baselines/`.
   - Results go to `out/results/baselines/<method>/<dataset>.csv`: the transfer CSV's columns plus `method`.
   - Third-party code lives in `third_party/<name>/` with a `SOURCE.md` (URL, commit or DOI, version, license). Nothing in `core/` imports it.
6. **Tests** for each piece (fakes, tiny grids). The saved-run replays stay unchanged.

## Phase 1: diagnosis (no LLM; idea 0)

1. **Failure taxonomy.** Covers the template, the 14/20 rule set and the multi-instance `closer` sets. Per grid: which requirement fails, and how:
   - a slip into a goal not yet due (sequence);
   - obstacle timing (avoidance);
   - reach near 0 (trapped).

   Shown next to a table of grid properties: size, `BFS_steps`, distances between consecutive goals and from each goal to the later ones, patrol cells on or next to shortest paths, and pockets. Output: `out/results/transfer_search/diagnosis.md`.
2. **`any` against a fixed order.** Rewrite each generic set's `c -> any` as `c -> d` in a fixed direction order (the 4 rotations, then all 24 orders if they differ), and re-certify on all 20 grids. This measures what the worst case over `any` costs.
3. **Search over the rule language: a ceiling, and the first baseline without an LLM.** Beam search over short decision lists:
   - Each rule is a conjunction over `wall`, `hazard`, `risky`, `toward`, `closer`, `obstacle` and the goal stage, with a fixed action or `any`.
   - A list is scored on a set of grids by worst-case requirements met.

   It answers three things:
   - whether each current training set can be solved by any short generic list;
   - the best validation score such a list reaches;
   - which grids no list over these features certifies (out of reach without new features or memory).

   The search starts with PRISM (1–3 s per check). If that is too slow, a numpy model of the gridworld scores candidates (best and worst values of the induced MDP; sequence requirements through the goal flags) and PRISM certifies only the finalists.
4. **Feature gaps.** For failure classes that no feature separates, list candidate generic sensors, for example "the move goes away from a goal not yet due" or "the patrol reaches the target cell next step". Adding a feature is a decision point: each one is hard-coding until the trimming phase removes it.

Outcome: which ideas can matter at all, which training sets to use, and whether the vocabulary caps out below the template's 6 of 8.

## Phase 1b: gridworld families (no LLM)

Parameterized layouts in the sense of 1–2–3–Go!'s parameterized MDPs: one generator per family, with the size `n` (and a seed for small variations) as parameters. Instances keep the dataset format (`n, grid, goals, static, moving, BFS_steps`), so `domains/gridworld` is unchanged. Generator: `domains/gridworld/data/families.py`.

- **Candidates:**
  - rooms joined by doors (U-traps around the doors for greedy navigation);
  - a serpentine corridor (moving towards a goal is often wrong);
  - goals placed next to each other (sequence slips, as in grid 12);
  - a patrol across the only passage between two goals (timing with `obs_idx`);
  - each optionally with scattered obstacles from the seed, to give many members per size.
- **Kept** only if, at every size checked:
  - every instance is jointly solvable (the joint query);
  - the template or generic greedy rules fail on some members, so the family is not trivial;
  - some single rule list certifies every member checked (hand-written, or found by the Phase 1 search), so transfer is possible.
- **Splits:** train on small sizes, validate on middle sizes, test on the largest, and on unseen seeds at every size. Positions that scale with `n` (goals in corners, doors in the middle) are where 1–2–3–Go!'s trees over raw coordinates cannot follow.

## Phase 2: Haiku batches

- **Start** from vocabulary v2 + `closer`, the only one that transfers today. Run without `closer` once something works.
- **Each idea** is a config flag, off by default, with a condition file `haiku_ts_<idea>`. It is documented in `docs/config.md`, and in `docs/semantics.md` when it changes the loop.
- **Cost:** a batch of about 8 conditions × 3 repeats costs about $0.75 and should take under an hour with its transfer checks.

**Batch 1: cheap flags, one factor each against the baseline.**
- **Training sets by property** (idea 1), config only (`run.train_sets`):
  - sets that the Phase 1 search can solve with one generic list;
  - clusters from the property table;
  - one set of all 12 training grids;
  - sets of 1, 3, 6 and 12 grids (idea 7).
- **Generality prompt** (idea 5): say that the rules will be certified on unseen, larger grids and must not single one out; ask for the strategy in words first, then the rules. Flags `prompt.generality`, `prompt.strategy_first`.
- **Closed patch language** (idea 4): variables compared with constants only; no relations between constants and no offsets on them; an optional rule cap. Flags `rules.patches: closed`, `rules.max_rules`.
- **Fixed tie-breaks,** if Phase 1.2 shows that `any` costs grids: an instruction and an example (a prompt flag).
- **Best-of-N first rounds** (idea 7): N initial answers; keep the one that passes the most members. Flag `planner.initial_samples`.
- **More rounds** (idea 7): 10.

**Batch 2: loop changes, built while batch 1 runs.**
- **Selection by held-out grids, or rotation** (idea 2): keep-best scored on held-out training grids (the LLM sees only a summary), or different members each round.
- **Feedback in feature space** (idea 3): blame and hotspots grouped by feature valuation and goal stage rather than by cell, and over all members rather than the worst one only.
- **Generalizing from solved rule sets** (idea 6): a prototype on the saved per-grid rule sets of `haiku_rl2_grid_large_closer`. One call asks what they share; the normal training-set loop then starts from that answer.

**Then:**
1. Combine whatever moved the validation numbers, and repeat.
2. Remove the winning combination's factors one at a time.
3. Drop `closer`, and any features Phase 1 added, and see what survives.

After each batch: a `DECISIONS.md` entry, the summary page, a commit and a short progress note.

## Phase 3: gridworld baselines (no LLM; alongside Phase 2)

All on the same training grids, the same validation and test grids, and in the same table:
- **The hand-written template** (`template_policy.py`, both variants).
- **Search over the rule language** (Phase 1.3), trained on the LLM's training sets. The "why LLMs" claim needs the LLM to beat it, or to match it with far fewer checks.
- **Trees in the style of 1–2–3–Go!:** decision trees learned from the optimal strategies of the training grids, frozen, then certified.
  - Two versions: (a) over the raw state variables (`x`, `y`, goal flags, `obs_idx`), as in the paper; (b) over our features, the strongest version of their idea without an LLM.
  - Single objective, as in the paper: reach every goal in order without a collision. A monitor flag turns it into reachability so that the strategy can be exported.
  - Trees are learned with the artifact's dtControl if it runs; otherwise with CART (sklearn) on the paper's settings, recorded as such.
- **A strategy synthesized on one instance, applied to the others** (Oct 8 plan): atomic rules over `x`, `y`, the goal flags and `obs_idx`.
- **The per-instance optimum** (the joint query on each instance): the ceiling, not a transfer method.

## Phase 4: beyond gridworld

- **Pac-Man.**
  - Train on several horizons at once (e.g. 8, 10, 12), then certify the frozen rules on horizons 8–25, and 30 if PRISM manages.
  - Compare with 1–2–3–Go!'s 0.869 at horizon 25. A certified crash probability of at most 0.56 at 25 is a clear win on a family it fails on.
- **The 1–2–3–Go! artifact.**
  - Run it in Docker (Docker Desktop must be running) or in WSL Ubuntu, and reproduce two of its published numbers first.
  - Then pick families from the Quantitative Verification Benchmark Set: at least one it fails on (csma or Mars rovers) and one it does well on.
  - Each becomes a domain (`domains/<name>/`) with a test that reproduces a published value, a vocabulary and thresholds. How to set thresholds per instance is a decision point.
- **Families of MDPs** (PAYNT, Policies Grow on Trees; `third_party/paynt/`).
  - Run on same-size gridworld subfamilies (one size's grids as a sketch).
  - Compare its robust per-state policies with our rules on held-out grids of the same size. It does not apply across sizes.
  - A positioning experiment, after the above.

## Phase 5: final runs

- The winning configuration and every baseline on the sealed test set and the larger grids, with 5 repeats.
- Sonnet, after asking.
- Figures, and updates to `DECISIONS.md`, `docs/positioning.md`, `docs/semantics.md` and `docs/config.md`.

## Decision points (stop and ask)

- Whether `closer`, and any feature added in Phase 1.4, stays in the configuration we claim.
- Any Sonnet run.
- If the search over the rule language matches the LLM on validation without an LLM: the "why LLMs" story changes.
- How to set thresholds for the benchmark-set families.
