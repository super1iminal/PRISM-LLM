# Decisions log

Decisions made while generalizing, to review together. Each entry has the decision and why. Items marked **[REVIEW]** are the ones I'm least sure about.

## 📌 OPEN: the story. Why an LLM, if PRISM can already synthesize the policy?
Asher has to pick one for the one-pager ("Why LLM > Synthesis", due to Marsha with the ceilings and the experiment designs). Nothing in the current runs settles it. The ceilings make the question sharper: PRISM's optimum is 1.0 on almost every gridworld requirement.
- **Expressiveness gaps (Aren).** Find a problem whose requirements or structure PRISM can't express, or can only express approximately or through a very complicated encoding (his example: unrolling recursion into a finite model). Marsha added data structures, time, and other language features. The challenge is a problem that needs the LLM's expressiveness *and* benefits from PRISM's guarantees. Right now our spec is one-to-one with PRISM. Candidate: the UUV paper's full three-phase mission (we model one phase). Keeps the current pipeline (the LLM writes policies) but needs a new problem.
- **Model engineering (Marsha).** Start from the problem description, not a PRISM model (Aren: our inputs are "designed to fit PRISM"). The LLM builds the model, PRISM's counterexamples drive revisions, and the result should be analyzable *and* reasonable, not "correct but abstracted into uselessness". The chain of revisions doubles as a justification of the model. This changes what the LLM produces (models, not policies).
- A third candidate from our notes: **restricted observation** (memoryless policies over what the agent can see are hard to synthesize; PRISM's POMDP engine rejected our obstacle requirements).
- Inputs and notes: `docs/plan.md`, "Why an LLM?".

## Setup / answers from kickoff
- New approach: **one LLM call over the whole state space** (no per-goal decomposition). Rules may refer to progress variables such as `g1`.
- Rule overlap: **ordered decision list, first match wins**.
- Feedback for missing rules **locates** high-mass uncovered states but never reveals PRISM's optimal action.
- qwen3 runs with **thinking off**, on **20 grids**.

## Cleanup
- The old planners (Vanilla, VanillaPlus, Feedback, FeedbackMinus, RL), their plotting scripts and the predecessor paper's code were removed from this line of work; all of it is still on `main`.
- The surviving approach (FeedbackSimplified) lives in `src/legacy/`, gridworld-only, kept as the baseline. Its behaviour is unchanged except instrumentation (per-iteration policies, raw outputs, tokens) and the Phase A obstacle-visibility switch.
- `grid_20_balanced.csv` = the first 4 grids of each size (4–8) from `grid_50_balanced.csv`, so still balanced by size.

## Generalization: inputs (what a case study provides)
- A case study is a directory `domains/<name>/` with a `Domain` subclass (instance loading and template context) plus Jinja2 templates. See `domains/README.md`.
- **MDP fully user-specified in PRISM** (`model.prism.j2`). Policy actions are the *action labels* of the commands. The planner never generates dynamics, only a `policy` module that synchronizes on those labels.
- **Requirements** are user-specified as PRISM path formulas with a bound (`>=`/`<=`) and a threshold, plus English text for the prompt. Best and worst case are `Pmax`/`Pmin` (swapped for `<=`).
- **English description** (`description.md.j2`) and **visual** (`visual.txt.j2`) are separate user templates injected into generic core prompts. A domain can override any core prompt template by dropping a file with the same name in its directory.
- **Policy-visible variables** are declared in the spec (ranges read from the model). For gridworld these are `x, y, g1..gN`. Config `domain.visible_extra` adds more; since Phase A the obstacle phase `obs_idx` is visible to rules and legacy alike (`pre_phase_a` hides it, as in the runs before Phase A).
- The gridworld MDP was rewritten as a symbolic PRISM template (formulas for move/slip/bounce) rather than enumerating cells. `tests/test_gridworld_equivalence.py` and `src/regression.py` confirm it is equivalent to the legacy DTMC (5e-10 across 100 policies under interval iteration).

## Generalization: symbolic policies
- Output: `{"rules": [{"condition", "action"}]}` via JSON-schema structured output. The action is an enum of the domain's labels.
- Condition language: a small PRISM-compatible expression subset (`= != < <= > >= + - & | ! =>`, parentheses, `true/false`), parsed and type-checked in Python. It is lenient about common LLM spellings (`==`, `&&`, `and`, `True`, a trailing `-> action`). I added the trailing-action case after qwen put `-> right` inside conditions during the smoke tests.
- Invalid answers are re-asked with the error message, up to 2 extra calls per attempt. If still invalid, the attempt counts as an empty policy.
- **First-match compilation**: rule *i*'s effective guard is `c_i & !c_j` for only those earlier rules *j* that can overlap it. Overlaps are found by enumerating the policy-visible space when it is ≤ 200k valuations, and otherwise every earlier rule is negated. This keeps 500-atom legacy policies compact.
- Old per-state policies are expressible as one atomic rule per state (`core.rules.atomic_rules`, `domains/gridworld/legacy_translate.py`).
- The rule listing shown back to the LLM uses the same JSON-per-line shape as its output. The example rules also moved to JSON form (qwen copied the `cond -> action` notation into the condition field).

## Refinement loop
- Success means **every requirement holds in the worst case**. The best-case pass rate is reported separately.
- The switch follows the slides. If best case fails anything → **refine** (the LLM returns a complete new rule list). If only worst case fails → **extend** (the LLM returns *only new rules*, **appended** after existing ones). Appending means existing decisions are unchanged, so worst case can only go up. Best case can go down, which then triggers refine.
- **Keep-best** like legacy: score = (#worst-case failures, #best-case failures, total worst shortfall, total best shortfall). Feedback is always built from the best policy so far.
- `max_attempts=5` generation rounds, the same number as legacy. Legacy makes one call per goal per round (3 per round; since Phase A, one per goal and obstacle phase). Symbolic makes one per round, plus re-asks for invalid answers.
- Per instance, one extra PRISM run computes the unconstrained optimum of each requirement on the bare MDP. It is used for refine-mode blame and reported as an upper bound on what is achievable.

## Feedback: probability mass
(Kept after the Sept 24 meeting, including V^opt. One-step regret is a deferred ablation, S5.)
- I first tried a performance-difference decomposition (visits × local advantage). It gives ~0 mass when the adversary *stalls* the agent in loops through uncovered states, and those loops were the main failure mode in testing. Without discounting or absorbing targets, the lemma's residual term doesn't vanish.
- Used instead: **mass(s) = expected visits to s within H=100 steps × probability still at stake at s**, reported as a share of the total.
  - extend: strategy = greedy worst-case completion, stake = best(s) − worst(s), counted only on uncovered states and aggregated by policy-visible valuation (top 10 shown).
  - refine: strategy = greedy best-case completion, stake = optimum(s) − best(s), charged to the rule deciding s (top 10 rules, each with its top 3 states).
- Greedy strategies are computed from PRISM's per-state value vectors (`filter(printall, ...)`) and the exported transition matrix. For LTL (non-reachability) formulas those vectors are PRISM's projection from the product automaton, so the ranking is a heuristic there.
- Feedback **locates** states and rules but never states PRISM's optimal action (per the kickoff answer).
- The prompt table shows best, worst and threshold for every requirement, plus coverage (reachable situations vs uncovered).

## Regression
- Two checks:
  1. *stored*: new best case at default PRISM settings vs the values the legacy run reported (tolerance 1e-9). These use the same solver, so they match to rounding (observed 1e-15).
  2. *exact*: legacy DTMC recomputed vs new MDP best **and** worst, all with **interval iteration** (sound error bounds, epsilon 1e-9, tolerance 1e-7). This is the real model-equivalence test.
- Why interval iteration: at default settings (and even at epsilon 1e-10 with plain value iteration), PRISM's worst case (Pmin) for the nested-until sequence properties stops early, by up to 4e-3 at default settings and 5e-6 at 1e-10. At 1e-10 one legacy policy didn't converge at all. With interval iteration, best and worst agree to 1e-10.
- Side finding: the legacy run's own reported sequence-property values carry up to ~1e-3 solver error at default settings. That's negligible for the paper's conclusions, but worth knowing.
- Solver fallback: for one of the 100 legacy policies (grid 9, attempt 4), interval iteration does not converge at any epsilon (non-progressing loops). That policy falls back to Gauss-Seidel at epsilon 1e-12, where legacy and new agree to 2e-11. PRISM's `-exact` engine can't be used because it doesn't support LTL formulas.
- Every iteration's policy of every sample is checked, not just the final one (the legacy planner now records the full policy at every iteration).

## Comparison
- Same model, same 20 grids, `max_attempts=5`, 2 concurrent workers for both runs, run one after the other (not concurrently) so they don't contend for the GPU.
- LLM time is client-side wall time for both, so it includes queueing behind the other worker. Tokens are contention-free and the fairer cost measure.
- The prompts differ in content by design. Legacy has two long worked examples with reasoning. Symbolic has a short rule-syntax example block. I did not port the worked examples. Examples are an ablation instead (S4 symbolic, L2 legacy; `docs/ablations.md`, deferred).
- The legacy "final" probabilities are those of its kept-best policy (reconstructed with its own keep-best rule for runs that predate the `final_prism_probs` field).

## UUV case study (`domains/uuv/`)
- Source: Päßler et al., iFM 2023 (arXiv:2308.14663). The paper omits most probabilities, so the MDP follows its ProFeat artifact (branch `scp-ifm_artifact` of `remaro-network/auv_profeat`, Apache-2.0), not the later extended model on that repo's main branch (with sonar/camera).
- **No core changes.** ProFeat features become state variables (`follow`, `alt`). The policy is the paper's feature controller: the actions `low`/`med`/`high` pick the next altitude in search-phase states. Forced transitions use a `[step]` label, which the policy module doesn't synchronize on.
- Visibility limits are modelled by **clamping**: requesting a higher altitude than allowed gives the highest allowed one. So every action is enabled everywhere (no deadlocks), and each state's set of distinct choices is exactly the paper's. Check: the bare MDP reproduces all reported numbers for both scenarios (Pmin F done = 1; Table 2 energy/time min/max; Pmin G safe = 0.65 / 0.32). The paper's `time`/`energy` reward structures are kept in the model for this check. The planner doesn't use them.
- Policy-visible: `s`, `alt`, `water_visib` (what the paper's controller monitors). Hidden: `follow`, `d_insp`, `t_failed`. `t_failed` is always 0 while searching.
- Requirements: `no_thruster_failure` = `G !"thruster_failure"` (the paper's `G "safe"`), and `done_in_time` = `F<=T "done"` (the paper's time reward turned into a deadline). An energy budget (reward requirement) was added later in A3, on branch `energy`, to be merged after the gridworld batch.
- **[REVIEW] Thresholds are tight by design.** In this model the controller has little leverage. Over all controllers, safety ranges over [0.654, 0.674] (North Sea) and [0.321, 0.355] (Caribbean). Done-in-time ranges over [0.52, 0.82] (T=30) and [0.06, 0.87] (T=70), and the low end is the adversary switching altitude back and forth. Thresholds (North Sea 0.670 / 0.80, Caribbean 0.345 / 0.862) are set so that "hold your altitude, go highest when a search starts" passes. Always-low, always-high (and, in the Caribbean, always-med) each fail at least one requirement. The best rule policy found by local search clears both by only ~0.003 / 0.002. Reproduce with `domains/uuv/data/calibrate.py`. Looser thresholds would make any sensible complete policy pass, so the case study would then mostly test coverage.
- PRISM's multi-objective query (`multi(Pmax [F<=T], P>=p [G ..])`) gave a value below a concrete policy's (0.795 vs 0.807), so it wasn't used for calibration.
- `done_in_time` is step-bounded, so the per-state vectors used by the mass analysis assume the full T steps remain from every state. The ranking is a heuristic there, as for LTL.

## UUV results (qwen3:14b, uuv_paper, 5 attempts, thinking off)
- `out/results/symbolic_uuv`, figure `viz/figures/uuv.png` (`viz/plot_domain.py`, which works for any domain and plots final policies against the domain's `reference_policies.json` and the range any controller achieves).
- North Sea: success in 2 attempts (0.6723 / 0.8064). Caribbean: fails. The final policy behaves like always-high (0.348 / 0.860 vs 0.862 needed).
- There is no legacy baseline for UUV (legacy is gridworld-only). The non-LLM reference to beat is `stay`, the only reference policy that passes both scenarios.
- Comparison with the paper (`viz/plot_uuv_summary.py` → `viz/figures/uuv_summary.png` and `.md`):
  - The paper doesn't synthesize a controller. Its managing subsystem is nondeterministic, and its best case per requirement is PRISM's separate maximum. No single controller reaches both best cases (always-high maximizes safety but misses the North Sea deadline).
  - Ours is within 0.002 (safety) and 0.012 (deadline) of those separate maxima.
  - On the paper's Table 2 measures, our North Sea policy needs 26.04 energy and 23.93 time to finish, against the paper's best possible 24.78 and 23.66.
  - Size: 25 rules vs ≥1,620 (North Sea) and ≥8,850 (Caribbean) states that an optimal PRISM strategy must decide (reachable states with more than one distinct choice). For the step-bounded deadline, the optimal strategy also depends on the step count.
  - Transfer: the North Sea rules reused unchanged on the Caribbean keep safety (0.347) but not the deadline (0.824).
- Figures regenerated Oct 5 with the energy requirement, which this run predates (`uuv.png` shows no energy dot for it; `uuv_summary.md` re-verifies its rules against every requirement). Our North Sea policy is within budget (26.04 ≤ 26.5); the Caribbean one is not (62.99 > 62.5). `stay` still passes everything in both scenarios (26.43, 62.17).

## UUV optimization (branch `optimization`, archived on GitHub, not merged)
- Sep 24, before the energy requirement and Phase A, so its numbers don't match the current setup. Target: match `stay` (passes both scenarios). **Not reached.** North Sea passed reliably; Caribbean failed in all 12 runs by ~0.002. Figure and runs (`out/results/opt/e*/r*`, 1–2 repeats each) live on the branch.
- E1 `keep` action: unused by qwen. E4 more explanation (switching arithmetic, requirement drivers): worse (0/4), reverted. E7 altitude facts in action text: no change.
- E2 found a **domain bug**: the description gave the visibility thresholds as decimals (`< 5.67`), qwen copied them into conditions, and the integer rule language rejected them. **Fixed on this branch** (E3): the description lists the integer levels of each band.
- **[REVIEW] Not carried over, need a decision:**
  - E5 (core): refine/extend feedback lists every previous attempt's worst case and flags an answer that verifies identically to an earlier one. Didn't help on UUV; never tested on gridworld.
  - E6: hide `water_visib` from the policy (limits are enforced by clamping anyway). The most useful change: qwen stopped writing per-band rules (which act like always-high) and wrote "hold the altitude" rules over 4–11 situations, but still started low or med at `s = 11`, where the Caribbean needs high. It contradicts "policy sees what the paper's controller monitors".

## Forced states (core change, made for UUV)
- Motivation: UUV states where the action has no effect (following, found, done: about 70%) showed up as uncovered and in feedback.
- `PolicyVerifier.forced_states()` finds the states of the bare MDP where every choice has the same successor distribution (probabilities rounded to 1e-12). It reuses the optimum run's exported transitions, or else runs PRISM once with no properties.
- Forced states no longer count as situations: `reachable_situations`/`uncovered_situations` count only valuations with at least one reachable decision state. They're also skipped in extend hotspots **and** refine blame, because no rule can change what happens there. `Verification.decisions` marks decision states.
- Gridworld has no forced states (checked on all 20 grids), so its coverage numbers and feedback are unchanged. UUV North Sea: a search-only policy is now complete (0 of 85 situations uncovered).

## Sept 24 meeting (Marsha, Aren)
Most outcomes became the plan (`docs/plan.md`: target, Phase A items, run order) and the ablation set (`docs/ablations.md`: retry sweep, seeds, what is deferred). Only what isn't recorded there:
- The story is open (pinned at the top). Generalization and readability are "a nice bonus", not the reason.
- **V^opt stays in the loop**; regret is an ablation (S5), not a replacement. Ceilings are "foundational", not an experiment.
- **Baselines are per case study** (e.g. RL for gridworld, the paper's controller for UUV). Rule transfer (5×5 → 8×8) is dropped.
- **[REVIEW]** Mass horizon H for UUV (the deadline?) is still open. Gridworld stays at 100.

## Results (qwen3:14b, grid_20_balanced, 5 attempts, thinking off)
- `out/results/comparison_grid20/report.md`, `viz/figures/grid20.png`. Symbolic numbers come from the **capped** run (`symbolic_grid20_capped`).
- Success 0/20 legacy vs 1/20 symbolic (worst case). Mean requirements met 4.30 vs 5.25. Shortfall 3.16 vs 2.11. Output tokens 12.2k vs 5.0k. Wall time 415 s vs 170 s. LLM calls 15 vs 5.
- Symbolic output tokens stay flat with grid size (~4–6k); legacy's grow from 5.4k (4x4) to 20.3k (8x8).
- The two are roughly tied on 4x4 grids.
- Loop usage: refine 63 attempts (35% improved the best so far), blind retry 10 (60%), extend 4 (100%). qwen nearly always wrote a catch-all rule, so policies were almost complete and extend rarely ran.
- These results were produced with the catch-all instruction and example in the prompt, which is the current prompt again (both were removed for the ablation below, then restored).
- Caveats: a single seed; the obstacle phase hidden from both; the prompts differ (legacy has worked examples); symbolic is scored on its conservative worst case. These runs predate Phase A; the 2-seed Phase A runs replace them as the main comparison.

## Catch-all ablation (qwen3:14b, grid_20_balanced, same settings)
- Three symbolic runs: catch-all instruction and example (`symbolic_grid20_capped`); instruction removed, example kept (`symbolic_grid20_nocatchall`); both removed (`symbolic_grid20_noexample`).
- The instruction alone barely mattered: rounds ending in a catch-all went from 77% to 72%, because qwen copied the example rule. Removing the example too halved it (37%; 10 of 20 final policies vs 18).
- With no catch-all at all: uncovered situations 26% of reachable (vs 9%), requirements met in worst case 4.50 (vs 5.25), shortfall 2.62 (vs 2.11). Best case 5.25, so the best-to-worst gap grew to 0.75 requirements. Still ahead of legacy (4.30, 3.16). Extend ran 11 times but improved only 2; refine success fell to 23%.
- Reading: with this model, partial policies are a liability under the conservative worst case, and extend feedback doesn't yet fill coverage well. Exposing `obs_idx` (done in Phase A) should shrink the worst-case penalty for partial policies. Worth revisiting with a stronger model.
- Figures: `viz/figures/catchall_ablation.png` (instruction only) and `viz/figures/catchall_ablation_full.png` (instruction and example).
- After the ablation, the catch-all instruction and example were **restored**, since they give qwen better results. Re-run without them when studying extend, or with a stronger model.

## Phase A (branch `phase-a`)
What was built is in `docs/plan.md` (Phase A table; ceilings and budget curves under Phase B). Decisions and findings from building it:
- **PRISM solver changed to Gauss-Seidel** (`prism.method`). With `obs_idx` readable, a rule set can make the induced chain periodic. PRISM's default value iteration then fails to converge, which would crash an instance mid-run. Found with a three-rule test policy. Gauss-Seidel and policy iteration agree on those models. Checks that reproduce old numbers pin PRISM's defaults: the UUV paper's numbers (Gauss-Seidel gives 4726.0 for the Caribbean max energy, vs the reported 4723.29) and the regression's stored-values check.
- **Joint queries use exact LP** (`prism.multi_method: lp`); value iteration has the same convergence problem there. LP can't handle step-bounded objectives (UUV's deadline). There, PRISM's value-iteration fallback wrongly says "not jointly feasible" even though the `stay` policy passes both scenarios. So the joint query returns *undecided*, and the loop falls back to the per-requirement branch. **[REVIEW]** This means the joint fix applies to gridworld but not UUV.
- **Legacy with the obstacle visible:** one call per (goal, obstacle phase); a short note in the prompt says which phase the per-cell policy is for. The legacy DTMC is solved with Gauss-Seidel when phases are observed (power method otherwise, as before). **[REVIEW]** Prompt wording of that note.
- **R4 detail:** a round that doesn't improve the kept policy has gain 0, so R4 also restarts after any non-improving round, not just slow ones.
- **Smoke-test findings (fixed before the main runs):**
  - With one fixed sampling seed, every repeat of a prompt gave the *identical* answer: R5's five rounds produced 1 distinct output, and B2 produced 3 in 5 rounds. Each call now uses a seed derived from the run seed and the call's index within the instance: reproducible, and 3/3 distinct in the re-test.
  - A real qwen rule set (16 rules reading `obs_idx`) did not converge even with Gauss-Seidel, and plain policy iteration failed too. Modified policy iteration solved it and matches Gauss-Seidel exactly on the other models tried. It is now the fallback (`prism.fallback_methods`). If every method fails, the planner scores the round as an empty policy instead of losing the instance. Legacy's verifier retries with `-bgaussseidel` then `-intervaliter` before its old silent all-zero fallback. **[REVIEW]** The legacy fallbacks are untested on a real non-converging case.

## A3: reward requirements (UUV energy), branch `energy`
- Core: a requirement may bound an expected reward (`reward: energy` in the spec, PRISM `R{"energy"}<=c [ F "done" ]`). Best and worst are `Rmin`/`Rmax`. Shortfall and blame stakes are taken relative to the threshold, so energy doesn't swamp the probability terms when they are summed. Infinite expected rewards (goal not surely reached) are capped in the mass analysis.
- UUV gets `energy_budget` (dataset column `energy_threshold`, optional), and the description now spells out the paper's energy costs. Calibrated like the other thresholds (`calibrate.py`), so `stay` passes:
  - North Sea: ≤ 26.5. `stay` uses 26.43, and `always_high` (27.13) fails it. `always_med` (25.98) still passes everything, as before.
  - Caribbean: ≤ 62.5. `stay` uses 62.17, and `always_high` (62.99) now fails on energy too.
  - Bare-MDP best and worst match the paper's Table 2 (24.78 / 59.08 min).
  - This creates a real trade-off: higher altitude is safer but costs energy.
- The joint query stays undecided for UUV (step-bounded deadline), so UUV keeps the per-requirement branch.
- Gridworld prompts render byte-identically to before (checked), so the gridworld runs are unaffected. The prompt wording for blame/hotspots mentions cost only when a reward requirement is failing.
- **[REVIEW]** Energy thresholds and the description of the energy costs.
- The mass horizon for UUV is still open.

## Batch 2 settings (deferred ablations, implemented Oct 3)
- **UUV mass horizon = the mission deadline** (Asher, Oct 3): `feedback.horizon: domain` asks `Domain.horizon(instance)`, so North Sea uses 30 and Caribbean 70. Since Oct 5 (Asher) the UUV run's own config file (`configs/run/conditions/U1.yaml`) sets it. Before, `default.yaml` held a per-domain map (`feedback.horizon_by_domain: {uuv: domain}`), but the run config should not name domains: each kind of run gets its own config file. Every condition resolves to the same settings as before.
- **Story:** A (expressiveness) for now, maybe B later. Nothing story-specific is implemented until there's a domain.
- **S1 (table feedback):** after a failure, every round shows the results table and the previous rules and asks for a complete new list. No blame, no REFINE/EXTEND framing, no appending. Retry policy unchanged.
- **S5 (regret blame):** one-step regret of the rule's action on the best-case values, with Q from the bare MDP. Successors the policy never reaches take their optimum value, which slightly favours deviating (a heuristic). Extend ranks uncovered states by the local gap best − worst, without occupancy. The prompt wording describes the signal it shows. **[REVIEW]**
- **V1 (random blame)** is a placebo: same number of rules/states as B2, equal shares, and B2's wording. **V2** drops the blame section but keeps the framing.
- **L1:** legacy's retry uses the same `stall:k` rule as symbolic; a blind-retry round regenerates every (goal, phase) from the initial prompt, then keep-best applies as usual.
- **Run order:** batch 1 (B2, R1–R5) finished first; then the symbolic part of batch 2 (S1, S4, S5, V1, V2), U1, then all legacy runs (B1, L1, L2). Legacy B1 was moved behind the symbolic batch-2 runs, following "all symbolic first, then legacy".

## Config folders (Oct 5)
- `configs/` has one folder per tool (Asher): `run/` (`default.yaml` + `conditions/<name>.yaml`), `ablation/`, `regression/` (each with a `default.yaml`) and `plot/` (one file per figure or report). The run config says what a run does; domains are defined in `domains/<name>/`, not in configs.
- Each `default.yaml` is the only place its defaults live (no Python defaults; a missing key is an error). The old Python copy of the run defaults had drifted (`domain.visible_extra`).
- Figures and reports read run facts (dataset, model, round budget) from each run's `config.json` instead of hard-coding them; metrics use each domain's `Requirement` thresholds instead of legacy's gridworld lookup. Committed figures regenerate byte-identical except the ablation summary's title (model name as recorded) and "Caribbean Sea" in its UUV table.

## Default retry policy changed (Oct 3)
- **Default `planner.retry` is now `gain:0.05`** (restart when the last round cut total worst-case shortfall by less than 0.05, including no improvement). It won the retry sweep: 4.95 requirements met vs 4.45 for the old default `stall:2` (p = 0.055) and 4.72 for pure resampling (p = 0.31; better on 10 grids, worse on 6).
- Every condition that ran with the old default now pins `retry: "stall:2"` in its condition file (`configs/run/conditions/<name>.yaml` since Oct 5), so its name still means what was run (checked against each run's `config.json`).
- **D7:** the new default with 7 rounds, 2 seeds, to see whether more rounds keep paying off (all budget curves were still rising at round 5).
- Legacy runs were stopped on Oct 3 (lower priority than the symbolic results); `docs/ablation_not_run.png`.
