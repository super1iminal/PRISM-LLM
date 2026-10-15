# 07 · One bit of memory in the rule language (`-memory`)

Read `plans/00-overview.md` first.

## Goal

Rules may read and write one boolean memory bit. Memory runs are kept apart from memoryless ones by name and by hard checks. First a PRISM-only check whether any instance needs the bit; then Haiku runs in their own experiment.

## Why

`docs/positioning.md` (Check 1): on 2–3 grids per set, memoryless local rules get trapped, cycling in a pocket that slips cannot leave (the U-trap of reactive navigation). One bit is the smallest change that can break such a cycle, for example by switching between two preference orders, or by following a wall until the goal direction opens up.

## Design (write it into `docs/semantics.md` before coding; Asher reviews this section)

- **Config:** `rules.memory: 0 | 1` (bits), default 0. With 0, everything is byte-identical to now.
- **Language:**
  - Conditions may use the boolean `mem`.
  - Each rule may also say what happens to the bit, through an optional field in the answer schema: `"memory": "set" | "clear" | "keep"`, default `keep`. Only with `rules.memory: 1`.
  - Rule listings and examples show the field in the same JSON-per-line shape.
- **Semantics:**
  - The rules define a finite-state controller with the bit as its state, initially false.
  - At a decision state with observation v and bit m, the deciding rule is the first that matches (v, m). It fixes the allowed actions and the bit's next value.
  - At an uncovered (v, m) any action is allowed and the bit is kept. The completion chooses actions only; the bit belongs to the rules.
  - Transitions not labelled with a policy action keep the bit.
  - The induced MDP is the product of the model with the bit. Best and worst case, coverage and soundness carry over unchanged, with coverage counted over distinct (v, m).
  - For `any` rules, the update belongs to the deciding rule; `fix_π` copies it to each rule it creates.
- **PRISM encoding:**
  - The policy module gets `mem : bool init false;` and one command per rule and action: `[a] e_{j,a} -> (mem' = u_j);`.
  - It keeps the uncovered command `[a] !(c_1 | … | c_n) -> true;`.
  - First-match guards treat `mem` as one more visible variable. The guards stay disjoint, so at most one command per label is enabled.
- **Everything that reads states must see `mem`:**
  - the verifier's `policy_valuation` and decision states;
  - the mass analysis, blame and hotspots;
  - the failing trace;
  - `core/rules.py`'s evaluator;
  - `transfer.py`.
- **Tools that cannot handle memory refuse it with a clear error:** `src/tie_analysis.py`, `src/polish.py`, `src/determinize.py`, `open_ties` (`--rules open`), and the numpy gridworld model. They need no support yet.

## The `-memory` tag (hard checks, each with a test)

- `src/config.py` `validate`: `rules.memory > 0` requires the condition name to end in `-memory`, and `-memory` requires `rules.memory > 0`.
- `src/experiments/store.py`: a run whose config has memory must sit in a directory whose name contains `-memory`. Otherwise the store raises; it never silently indexes such a run.
- `src/experiments/base.py`:
  - an experiment file that includes any memory cell must be named `*-memory.yaml` and set `memory: true`;
  - in any other experiment, a cell that resolves to memory is an error;
  - RQ1 has no memory cells unless Asher says so;
  - a memory experiment may hold a memoryless reference cell: that comparison is deliberate, and its report says so in the first line.
- Cell-slug directories (`<condition>+<slug>`) keep the tag, since the condition name carries it. Check that `src/run_ablation.py` and `src/run_search.py` do too.

## Steps

1. **Feasibility, PRISM only (no LLM).**
   - Take the trapped grids (`domains/gridworld/data/*_template.csv`, the "trapped" rows in `docs/positioning.md`) and the crossing family members our lists fail.
   - Hand-write one or two 1-bit rule lists: the template plus a bit that flips its preference order when it bumps into a wall, and a wall-following variant.
   - Certify them with the new encoding. If **no** instance passes with the bit where the best memoryless list we have fails, report that and stop before the LLM runs: memory has no headroom here.
2. **Implement the design**, with tests:
   - encoding and evaluation on the tiny grid;
   - a hand-made product check: the bit doubles the decision states, and a list that never sets it gives the same values as the memoryless list;
   - the schema field only with `rules.memory: 1`;
   - every guard above;
   - the replays unchanged.
3. **Prompt.** One block in the policy section, shown only with memory: what `mem` is, that it starts false, the update field, and one example of a use (escaping a loop). Keep it generic in `core/templates/`; domains need no change.
4. **Conditions and runs** (Haiku, from the main checkout, after merging):
   - `haiku_ts_order-memory` and `haiku_fam_order-memory`: the core plus `rules.memory: 1`.
   - Pac-Man (`haiku_mi_pacman_long-memory`) only if step 1 or a quick reason says the bit could matter there.
   - `configs/experiments/rq2_memory-memory.yaml`, with `memory: true`: the memoryless core as the labelled reference, and the memory cell, 3 repeats (5 if it looks promising).
   - Results go to `out/results/ablations/<condition>-memory…`.
5. **Report and docs.**
   - The report, with the first line saying it compares memory against memoryless on purpose.
   - `docs/semantics.md` ("Memory" subsection) and `docs/config.md` rows.
   - The naming rule in `docs/experiments.md`.
   - A `DECISIONS.md` line on the design choice (the bit is kept at uncovered states).

## Done when

Either step 1's negative result is reported, or memory is merged with every guard tested, and its experiment has run and is reported. Committed.

## Files

Owns `src/core/rules.py`, `src/core/verifier.py`, the planner's answer schema in `src/core/planner.py`, the memory prompt block, and the guards in `src/config.py`, `src/experiments/base.py` and `src/experiments/store.py`. Shares `src/config.py`, `configs/run/default.yaml`, `docs/config.md`, `docs/semantics.md` and possibly `src/core/prompts.py` and `src/core/analysis.py` with plan 05: one block each, rebase before merging, and run the full test suite including the replays.

## Report back

The step-1 result (which instances the bit rescues), the design as merged, and the run numbers.
