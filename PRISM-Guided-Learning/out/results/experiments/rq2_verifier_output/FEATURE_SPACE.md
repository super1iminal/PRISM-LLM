# Feedback in feature space (plan 04)

Hand-written notes beside the generated `REPORT.md`: the audit, the smoke test and the instance-specific-rule count.

## Step 1: where the feedback names a state

Read from `src/core/prompts.py`, the core templates, the `haiku_ts_features` prompts (grids, the only runs with `feedback.space: features`) and REFINE/EXTEND prompts rendered with scripted answers on the tiny grid and on Pac-Man horizon 8.

| Place | `states` (default) | `features` before plan 04 |
|---|---|---|
| Blame per rule, "Most in:" | valuations: `x=0 & y=1 & g1=false & … & obs_idx=1`; Pac-Man `xP=3 & yP=1 & dP=0 (right) & xG0=2 & …` | `Prompts._state`: boolean variables, state features that hold, per action the action features that hold; integers left out |
| Uncovered hotspots (EXTEND) | valuations | `_state`, states that read the same merged |
| Failing trace (`feedback.trajectory`) | valuations, every step and the end | already `_state` (the trace renders through it), so position-free |
| Per-member summary, results table, coverage counts, hidden-member count, best-order report | no states | no states |
| Blamed rule's text | the LLM's own condition | same (it names the rule, not a state) |
| Invalid-answer re-ask | parse errors only | same |
| Problem, maps, constants | the task | the task (stays) |

Findings:
- **Pac-Man crashed in feature mode.** `moves_left` is an integer state feature, and `_state` evaluated every state feature as a boolean condition (`RuleError: condition 'moves_left' is not a boolean expression`) at the first REFINE/EXTEND prompt. So feature-space feedback has never run on Pac-Man.
- Gridworld (grids and families) was already position-free: `x`, `y` and `obs_idx` are integers and were left out; what remains is the goal flags and per action `wall`, `hazard` (or `patrol`/`early`), `risky`, `toward`, `closer`, `obstacle`.
- Pac-Man has no boolean variables: every variable is an integer (positions `xP`, `yP`, `xG0`, `yG0`, `xG1`, `yG1`; headings `dP`, `dG0`, `dG1`; `steps`). In feature space it reads `moves_left=…` and per action `wall`, `nearer_g0`, `nearer_g1` (`closer_g*` are hidden in the core condition).
- For plan 05 (the trace's rendering, not positions): moves the policy does not choose (Pac-Man's ghost labels `g0`, `g1`; gridworld's forced states) read "`g0` (no rule decides here)", as if the state were uncovered; in feature space, consecutive steps in which only the ghosts move read the same.
- Not about positions: when the best case passes and the worst fails with every situation covered (open `any` choices, only with `rules.any_order: none` or `report`), REFINE blames against the best case and its blame section comes out empty (seen on Pac-Man horizon 8). The core settings resolve `any` each round, so this does not arise there.

Decisions:
- Integer-valued state features are shown as `name=value`; integer variables stay out, headings included: under `rules.general` a rule can only write the numbers 0 and 1, so a heading code could not be used.
- `feedback.space: features` keeps its name: on gridworld, the only domain it ran on, nothing changes (checked by replaying a `haiku_ts_features` training set, step 2), and on Pac-Man it never ran.

## Step 3: smoke test

Haiku, the core condition of each benchmark with `feedback.space=features`, the first training set, 3 rounds (`run.limit=1`, `planner.max_rounds=3`), outside the run store. Every REFINE prompt of rounds 2-3 was read; none names a position, and the feedback reads as intended:
- Pac-Man: `moves_left=12 | right: nearer_g1; up: nearer_g0, nearer_g1; left: nearer_g0; down: wall`. Hotspots of one rule often differ only in `moves_left`. Round 3's answer dropped two conditions (crash 0.776 to 0.675 on the focus member).
- Families: `g1=false & g2=false & g3=false | up: none; right: toward, closer; down: none; left: wall`. The answers added `!patrol & toward` conditions; `avoid_moving_seg1` rose from 0.52 to 0.75.
- Grids: the same form; several blamed states are after the last goal (`g1=true & g2=true & g3=true | up: hazard; right: risky; …`). Round 3's answer still wrote a position rule on its own, `!g1 & x + 1 = G1R & y = G1C & !wall & !hazard`: feature-space feedback does not stop the LLM from reading positions, it only stops naming them.

## Step 4: feature space against coordinates

The cell "States named by their features" (`feedback.space=features`, otherwise the core configuration) against "Full feedback (core)", which names states by coordinates. Validation targets, 5 Haiku runs each, certified per rule list; Δ with its 90% bootstrap interval over runs (`REPORT.md`):

| Benchmark | Coordinates | Feature space | Δ [90%] | Ties open (coord. / features) |
|---|---|---|---|---|
| Random grids (of 8) | 6.25 (5.50–7.00) | 5.75 (5.50–6.00) | −0.50 [−0.90, −0.10] | 5.50 / 4.30 |
| Families (of 20 in its group) | 9.67 (5.67–15.00) | 7.53 (4.00–12.00) | −2.13 [−5.20, +1.00] | – / 1.60 |
| Pac-Man (of 6, horizon 30 added by plan 02) | 2.40 (1.00–3.00) | 2.60 (2.00–3.00) | +0.20 [−0.40, +1.00] | 2.40 / 2.60 |

- Grids: a small loss whose interval excludes 0; every feature-space run certified 5.5–6.0 per list, the coordinate runs 5.5–7.0. Library 8 of 8 in both.
- Families: lower on each family (rooms 6.8 against 8.4, crossing 2.8 against 4.6, adjacent 13.0 against 16.0), within the families' large run-to-run spread.
- Pac-Man: level. Horizon 30 (plan 02 added it to the targets after these runs were certified) is being certified for the feature-space runs; seed 1 does not pass it, and at 0.56 nothing passes beyond horizon 18, so the counts should not move.
- Cost is the same: output tokens per set 10,834 against 11,315 (grids), 9,416 against 9,947 (families).

## Step 5: instance-specific rules, measured

`src/rule_specificity.py` (rule parser, no LLM) over every run of the cells below; table in `instance_specific.md`, per run in `instance_specific.csv`. A condition is counted once per list. "Specific": it compares a position with a constant or another position (`y = G3C`), offsets a constant (`G2C - 1`), or relates constants (`G2R = G3R + 1`). "Rounds 2+": the lists the LLM wrote after feedback (every round's candidate, not only the kept one; the kept list per round, as `budget_transfer.kept_lists` rebuilds it, is a selection among these). "Rewritten": in each REFINE prompt, the share of listed rules whose condition is gone from the answer, for the rules the blame names and for the others.

| Benchmark | Feedback | Final: conditions | specific | Rounds 2+: specific / conditions | Blamed / other rules rewritten |
|---|---|---|---|---|---|
| Grids | coordinates | 9.2 | 0.2 | 0.2 / 9.3 | 9% / 11% |
| Grids | feature space | 8.8 | 0.0 | 0.1 / 9.3 | 13% / 20% |
| Grids | restarts only | 6.7 | 0.0 | 0.0 / 7.2 | – |
| Families | coordinates | 6.9 | 0.0 | 0.0 / 7.7 | 19% / 20% |
| Families | feature space | 7.9 | 0.0 | 0.0 / 8.2 | 23% / 16% |
| Families | restarts only | 5.6 | 0.0 | 0.0 / 6.2 | – |
| Pac-Man | coordinates | 6.8 | 0.0 | 0.0 / 5.8 | 24% / 9% (of 29 blamed) |
| Pac-Man | feature space | 5.0 | 0.0 | 0.0 / 5.3 | 13% / 9% (of 15 blamed) |
| Pac-Man | restarts only | 5.8 | 0.0 | 0.0 / 6.0 | – |

Reading: **the hypothesis does not hold in this form.** With general mode, the generality instruction and the feature vocabulary, the LLM almost never writes a position condition, an offset or a relation between constants, under any feedback: at most 0.2 of about 9 conditions per grid list, none on families or Pac-Man. Coordinates in the feedback do not turn into position rules, so naming states by features has nothing to remove. What feedback does change is the size of the lists: 9.2 conditions against 6.7 for restarts only on grids, 6.9 against 5.6 on families, with or without coordinates. If the feedback overfits the training members, it does so through more and narrower feature conjunctions, not through positions. The blame also barely steers what is rewritten: blamed and other rules are rewritten at similar rates on gridworld (Pac-Man has too few blamed rules to read).

## Step 6: what goes into the core

Plan 04's rule: feature space enters the core unless clearly worse, the lower end of its 90% interval below −1 target on some benchmark. **Met on families** (−5.20), not on grids (−0.90, though the interval excludes 0) or Pac-Man. So feature space does not enter the core automatically; this goes to Asher before plan 09 decides (`DECISIONS.md`, open questions). Two things for that decision: on families the rule is met by every RQ2b cell (random blame −7.31, regret −4.80, trace −4.98), so it mostly reads the families' spread; but feature space is lower on all three families and on grids, and higher on none.

## The other RQ2b cells

Launched after the feature-space cell (`--max-parallel 2`), in the plan's order: results table only and no blame (3 repeats each benchmark), random blame (to 5), then regret and the failing trace on Pac-Man (3 each; the trace cell is plan 05's coordinate reference, which the store will not repeat). Their numbers go into `REPORT.md` as they finish; this page is updated once all are in.
