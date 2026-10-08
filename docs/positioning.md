# Positioning against related work (phase 2 draft, 2026-10-08)

Builds on `docs/related_work_survey.md` (phase 1; entry labels such as I.3 refer to it). This page states what we can claim, what each claim is up against, and what three PRISM-only checks and three paper readings showed before the transfer experiment.

## Bottom line

Our position is an intersection. No work found combines all four of:
- an LLM-written rule policy;
- a stochastic model with several probability or reward thresholds;
- exact certification on every instance, held-out and larger ones included;
- refinement from quantitative verifier feedback.

Each neighbour has two or three of these. "Nobody combined them" only persuades with an experiment in which the combination does what the neighbours cannot. Three things the checks below settled:

1. **The rule-list form is not ours.** Ordered decision lists for stochastic planning, learned on small instances, date back to Fern, Yoon and Givan (JAIR 2006). The claim is about who writes the list and what certifies it.
2. **The UUV family is trivial as defined.** The paper's own `stay` controller meets every requirement across a 0.5×–3× sweep of failure rates in both scenarios. "One controller across changing conditions" needs a different UUV family, or a different domain.
3. **Gridworld transfer is neither trivial nor hopeless.** A hand-written template over relative features certifies on half of the grids. It fails on the rest for reasons a better rule list could partly fix (slips into future goals, the moving obstacle), but on 2–3 grids per set it gets trapped for good. So gridworld leaves room for LLM-written rules to beat a human template, but not everywhere.

## Claims and their nearest work

| Claim | Nearest work and what it lacks | Our evidence now | Needed |
|---|---|---|---|
| Rules certified on every instance checked, held-out and larger ones included, in a stochastic model | **1–2–3–Go!** (I.3): trees over raw state variables, learned from optimal policies of small instances, value computed exactly on each larger one. Single reachability objective; predicates `x > c`; model parameters cannot appear in predicates; no loop, no LLM. | None yet (rules use absolute coordinates) | Head-to-head on a family where thresholds move with the instance (our gridworld: goal positions are constants, not state variables). Compare threshold satisfaction per instance, not optimality, which is 1–2–3–Go!'s target. |
| LLM-written generalized policies with probabilistic semantics | Silver (B.1), Stein (B.2), Prompt Prove Patch (B.4): deterministic PDDL; generalisation tested, not certified. Fern et al. (I.5): decision lists for stochastic PPDDL, learned, judged by simulation, no thresholds. | Per-instance rules on gridworld and UUV | The transfer experiment. Pitch against Prompt Prove Patch: in a deterministic domain, running the policy certifies it on an instance; in a stochastic one it takes model checking. |
| Quantitative verifier feedback, compared with resampling at equal budget | Stechly (C.1), Olausson (C.2): resampling about as good as feedback, deterministic verifiers, budgets in rounds. Stein (B.2): more initial programs vs more debugging steps, no consistent winner. Not a repair-vs-restart test as such. | qwen: pure resampling ahead at equal tokens (exploratory). Haiku on large grids: running, four planned comparisons. | Report whatever the Haiku comparison shows. Novel as a measurement (survey gap 5), not as a method claim. |
| Partial rules certified over all completions | Dräger et al. (H.1): permissive multi-strategies, per-state tables, no LLM. CAV 2026 shields (H.3): strong safety and strong permissiveness cannot both hold for probabilistic safety. | Barely exercised: catch-all rules are the norm; extend ran 4 times in 77 rounds; every Sonnet final policy covers every situation | Only a headline if transfer exercises it (frozen rules leaving situations uncovered on a new instance, still certified). Word it as soundness, never maximal permissiveness. |
| Readable, compact policies | dtControl (H.2), dtPaynt (G.4): small trees with exact values on one model | Rule counts (Sonnet 5.5: 11–35 rules for up to 1,400 situations) | Rule count against a dtPaynt tree at the same guarantee. Not done. |
| One certified controller across changing conditions (SEAMS) | Policies Grow on Trees (F.1), SMPMC (F.2): exact robust policies over finite families sharing a state space | The UUV failure-rate family is covered by `stay` (below) | A family where no simple policy passes and members differ in ways finite-family tools cannot handle (state space, step-bounded deadlines, several thresholds). |

## Check 1: a hand-written template over relative features (gridworld)

`domains/gridworld/data/template_policy.py`: one template for every grid. It goes to the next unreached goal, treating static obstacles, the moving obstacle's patrol cells and not-yet-allowed goals as blocked. Its decision depends only on the direction to that goal and four local blocked flags, so it is a ~10-rule list over relative features. A `cautious` variant also avoids moves whose slips land on a hazard. Results (exact checks where a value was within 0.03 of its threshold; results in `domains/gridworld/data/*_template.csv`):

| Dataset | greedy | cautious | either | LLM rules per instance (absolute coordinates) |
|---|---|---|---|---|
| `grid_20_balanced` (4×4–8×8) | 6/20 | 9/20 | 10/20 | Haiku 5.5 16/19 solvable, Sonnet 5.5 19/19 |
| `grid_20_large` (9×9–13×13) | 7/20 | 10/20 | 10/20 | Haiku 5.5 14/20 per run (restart after 2 stalls), Sonnet 5.5 3/3 (13×13 smoke test) |

Failures of the better variant per grid:
- `grid_20_balanced`: sequence requirements only, 7 grids (2 within 0.1 of the threshold); permanently trapped, 2; moving obstacle, 1.
- `grid_20_large`: sequence only, 5; trapped, 3; moving obstacle, 2.

Trapped means a goal is reached with probability near 0: the deterministic preference order cycles in a pocket that slips cannot leave. This is the U-trap of reactive navigation. Fern et al. name the same limit ("route-finding in maps/grids") and propose search.

Reading:
- The domain is not trivial: a human template fails on half of the grids, while per-instance LLM rules solve most.
- Some failures are within reach of better local rules (keeping clear of future goals and the patrol). The trapped grids need information local features do not carry.
- A feature such as "the next step on a shortest path" would make transfer trivial and leave the LLM nothing to do, so it must not be offered.
- Design consequence: expose instance constants (goal coordinates, grid size) in the rule language so the LLM writes relative conditions itself (`x < G1X`), plus generic sensors (`wall_up`), and report the trapped grids as out of reach of memoryless local rules.

## Check 2: is the UUV failure-rate family trivial?

`domains/uuv/data/failure_sweep.py` scales every thruster-failure probability by k ∈ {0.5, 0.75, 1, 1.5, 2, 3} in both scenarios (12 members). Each member keeps the paper scenario's slack: threshold = the member's best value over all controllers minus (for energy, plus) the slack at k = 1.
- `stay` meets every requirement of all 12 members.
- `always_med` meets every North Sea member.
- `always_low` and `always_high` fail every member.

The family is trivial: a fixed 4-rule policy from the paper covers it. A SEAMS story built on it would be "the LLM rediscovers `stay`". Options: tighter or differently shaped thresholds (whether a single controller still exists is then a multi-environment question), or families that vary the visibility ranges, deadlines or inspection lengths, where the right altitude thresholds move. Each option needs this check before any LLM run.

## Check 3: the papers our claims lean on

- **1–2–3–Go! (VMCAI 2025), results.**
  - Setup: 21 model–property pairs from the Quantitative Verification Benchmark Set plus Mars Exploration Rovers, all single reachability objectives.
  - Results: near-optimal on 13 of 21. It fails on csma (the property needs variables of every module), on Pac-Man (a horizon-like parameter) and on Mars rovers when only a probability changes.
  - Predicates are axis-aligned on raw state variables, chosen by Gini index; parameters never appear in predicates; tree sizes are not reported. The induced chain was built exactly in all but 4 cases.
  - No refinement loop and no LLM. The survey's entry is accurate. Its failure modes define where our rules must win, and our gridworld is such a case by construction.
- **Shields to Guarantee Probabilistic Safety (CAV 2026), introduction and definitions.**
  - Strong safety means every policy the shield allows is safe; strong permissiveness means every safe policy is allowed. No shield for probabilistic safety has both.
  - Our worst case over all completions is strong safety for the policies our rules allow; we claim no permissiveness, so the result does not touch us.
  - Their specification is a single reachability bound; ours are several thresholds, each holding for every completion and therefore jointly.
  - Read up to Section 4.1 only; Theorem 4.2's exact conditions not read.
- **Fern, Yoon, Givan (JAIR 2006), full text.**
  - Decision lists of action-selection rules: the first rule that allows an action wins; rules are written in a taxonomic concept language.
  - Learned by approximate policy iteration with random-walk bootstrapping on small instances; applied to larger ones.
  - Stochastic PPDDL domains (ground logistics, coloured blocks world, boxworld, including IPPC's hand-tailored track); judged by simulated success ratio, with no thresholds and no verification.
  - Their stated limits: the policy language for route-finding in grids, and reactive policies without search.

## What to say and not say

- Say "certified on every instance we check, including held-out larger ones". Do not say "correct for all sizes".
- Say "sound": every policy the rules allow meets every threshold. Do not say "maximally permissive".
- Do not claim the decision-list representation, or that rule lists are new for stochastic planning.
- Do not claim that PRISM or the family literature handles only one instance at a time (F.1, F.2, I.3).
- Do not claim feedback beats resampling unless the planned Haiku comparisons show it.
- Do not claim compactness against raw strategy tables; compare with dtControl or dtPaynt trees.
