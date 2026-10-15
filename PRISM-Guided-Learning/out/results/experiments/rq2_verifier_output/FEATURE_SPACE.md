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
