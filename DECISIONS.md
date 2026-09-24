# Decisions log (symbolic-policies branch)

Decisions made while generalizing, to review together. Each entry has the decision and why. Items marked **[REVIEW]** are the ones I'm least sure about.

## Setup / answers from kickoff
- New approach: **one LLM call over the whole state space** (no per-goal decomposition). Rules may refer to progress variables such as `g1`.
- Rule overlap: **ordered decision list, first match wins**.
- Feedback for missing rules **locates** high-mass uncovered states but never reveals PRISM's optimal action.
- qwen3 runs with **thinking off**, on **20 grids**.

## Cleanup
- Branch `symbolic-policies` was created from `generalize` (52c899b).
- Removed: Vanilla, VanillaPlus, Feedback and FeedbackMinus planners, RL (`RLCounterfactual`, `GridWorldStepper`), `UniformPlanner`, `Repairer.py`, `PlotRQs.py`, `PlotResults.py`, `moving-obstacle-plan.md` and `prism_portable.yml` (replaced by `requirements.txt`). All of it is still on `main`.
- The surviving "current approach" (FeedbackSimplified) now lives in `src/legacy/`. It's entirely gridworld-specific, and it's kept only as the baseline for the regression and comparison runs. It isn't part of the new approach.
- The legacy planner has **not changed behaviour**. I only added instrumentation: every verified per-iteration policy, raw LLM outputs and token counts are stored for the regression step. `-exportstates` to a fixed shared path was removed from the legacy PRISM call. It was a side effect only, and it wasn't safe to run in parallel.
- The LLM is a small `core/llm.py` wrapper around the `ollama` Python client with JSON-schema structured output, which replaces LangChain. Settings: `num_ctx=16384`, `num_predict=8192`, `think=False`, and the model's default temperature and sampling.
- The PRISM path is `$PRISM_PATH`, else `prism` on PATH (no more hardcoded path). All output paths are now absolute, relative to `PRISM-Guided-Learning/`.
- The gridworld datasets moved to `domains/gridworld/data/`. `grid_20_balanced.csv` = the first 4 grids of each size (4–8) from `grid_50_balanced.csv`, so still balanced by size.
- The local Ollama model was missing its weight blob (manifest present, 9.3 GB blob gone), so I re-pulled `qwen3:14b-q4_K_M` with your OK.

## Generalization: inputs (what a case study provides)
- A case study is a directory `domains/<name>/` with a `Domain` subclass (instance loading and template context) plus Jinja2 templates. See `domains/README.md`.
- **MDP fully user-specified in PRISM** (`model.prism.j2`). Policy actions are the *action labels* of the commands. The planner never generates dynamics, only a `policy` module that synchronizes on those labels.
- **Requirements** are user-specified as PRISM path formulas with a bound (`>=`/`<=`) and a threshold, plus English text for the prompt. Best and worst case are `Pmax`/`Pmin` (swapped for `<=`).
- **English description** (`description.md.j2`) and **visual** (`visual.txt.j2`) are separate user templates injected into generic core prompts. A domain can override any core prompt template by dropping a file with the same name in its directory.
- **Policy-visible variables** are declared in the spec (ranges read from the model). For gridworld these are `x, y, g1..gN`. The obstacle phase `obs_idx` is hidden, matching the legacy per-cell policies.
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
- `max_attempts=5` generation rounds, the same number as legacy. Legacy makes one call per goal per round (3 per round). Symbolic makes one per round, plus re-asks for invalid answers.
- Per instance, one extra PRISM run computes the unconstrained optimum of each requirement on the bare MDP. It is used for refine-mode blame and reported as an upper bound on what is achievable.

## Feedback: probability mass **[REVIEW]**
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
- The prompts differ in content by design. Legacy has two long worked examples with reasoning. Symbolic has a short rule-syntax example block. I did not port the worked examples. **[REVIEW]**
- The legacy "final" probabilities are those of its kept-best policy (reconstructed with its own keep-best rule for runs that predate the `final_prism_probs` field).

## UUV case study (`domains/uuv/`)
- Source: Päßler et al., iFM 2023 (arXiv:2308.14663). The paper omits most probabilities, so the MDP follows its ProFeat artifact (branch `scp-ifm_artifact` of `remaro-network/auv_profeat`, Apache-2.0), not the later extended model on that repo's main branch (with sonar/camera).
- **No core changes.** ProFeat features become state variables (`follow`, `alt`). The policy is the paper's feature controller: the actions `low`/`med`/`high` pick the next altitude in search-phase states. Forced transitions use a `[step]` label, which the policy module doesn't synchronize on.
- Visibility limits are modelled by **clamping**: requesting a higher altitude than allowed gives the highest allowed one. So every action is enabled everywhere (no deadlocks), and each state's set of distinct choices is exactly the paper's. Check: the bare MDP reproduces all reported numbers for both scenarios (Pmin F done = 1; Table 2 energy/time min/max; Pmin G safe = 0.65 / 0.32). The paper's `time`/`energy` reward structures are kept in the model for this check. The planner doesn't use them.
- Policy-visible: `s`, `alt`, `water_visib` (what the paper's controller monitors). Hidden: `follow`, `d_insp`, `t_failed`. `t_failed` is always 0 while searching.
- Requirements (probabilistic only, since core has no reward requirements): `no_thruster_failure` = `G !"thruster_failure"` (the paper's `G "safe"`), and `done_in_time` = `F<=T "done"` (the paper's time reward turned into a deadline). **[REVIEW]** Energy isn't a requirement. Adding it would need reward-bounded requirements in core (`R{..}<=c`), which I didn't add without asking.
- **[REVIEW] Thresholds are tight by design.** In this model the controller has little leverage. Over all controllers, safety ranges over [0.654, 0.674] (North Sea) and [0.321, 0.355] (Caribbean). Done-in-time ranges over [0.52, 0.82] (T=30) and [0.06, 0.87] (T=70), and the low end is the adversary switching altitude back and forth. Thresholds (North Sea 0.670 / 0.80, Caribbean 0.345 / 0.862) are set so that "hold your altitude, go highest when a search starts" passes. Always-low, always-high (and, in the Caribbean, always-med) each fail at least one requirement. The best rule policy found by local search clears both by only ~0.003 / 0.002. Reproduce with `domains/uuv/data/calibrate.py`. Looser thresholds would make any sensible complete policy pass, so the case study would then mostly test coverage.
- PRISM's multi-objective query (`multi(Pmax [F<=T], P>=p [G ..])`) gave a value below a concrete policy's (0.795 vs 0.807), so it wasn't used for calibration.
- `done_in_time` is step-bounded, so the per-state vectors used by the mass analysis assume the full T steps remain from every state. The ranking is a heuristic there, as for LTL.
- States where the action has no effect (following, found, done: about 70% of UUV states) were first counted as situations, so they showed up as uncovered and in feedback. This is fixed in core (see "Forced states" below).

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

## Forced states (core change, made for UUV)
- `PolicyVerifier.forced_states()` finds the states of the bare MDP where every choice has the same successor distribution (probabilities rounded to 1e-12). It reuses the optimum run's exported transitions, or else runs PRISM once with no properties.
- Forced states no longer count as situations: `reachable_situations`/`uncovered_situations` count only valuations with at least one reachable decision state. They're also skipped in extend hotspots **and** refine blame, because no rule can change what happens there. `Verification.decisions` marks decision states.
- Gridworld has no forced states (checked on all 20 grids), so its coverage numbers and feedback are unchanged. UUV North Sea: a search-only policy is now complete (0 of 85 situations uncovered).
- Cost: one extra PRISM run per verifier when the optimum isn't computed first (tests only; the planner computes the optimum first). The test suite went from ~28 s to ~38 s.

## Prompt template changes (after the gridworld runs)
- `_problem.md.j2`: the boolean example uses the domain's first boolean policy variable, and is left out when there is none. The rendered gridworld text is identical (`g1`).
- `_results.md.j2`: "reachable situations (combinations of the state variables)" became "... where the action matters". This is the only wording change in gridworld prompts compared with the runs above.

## TODO
- Per-domain switch to expose extra variables to rules (e.g. `obs_idx` in gridworld), which makes the worst case exact. Keep it off for legacy comparisons.

## Results (qwen3:14b, grid_20_balanced, 5 attempts, thinking off)
- `out/results/comparison_grid20/report.md`, `viz/figures/grid20.png`. Symbolic numbers come from the **capped** run (`symbolic_grid20_capped`).
- Success 0/20 legacy vs 1/20 symbolic (worst case). Mean requirements met 4.30 vs 5.25. Shortfall 3.16 vs 2.11. Output tokens 12.2k vs 5.0k. Wall time 415 s vs 170 s. LLM calls 15 vs 5.
- Symbolic output tokens stay flat with grid size (~4–6k); legacy's grow from 5.4k (4x4) to 20.3k (8x8).
- The two are roughly tied on 4x4 grids.
- Loop usage: refine 63 attempts (35% improved the best so far), blind retry 10 (60%), extend 4 (100%). qwen nearly always wrote a catch-all rule, so policies were almost complete and extend rarely ran.
- These results were produced with the catch-all instruction and example in the prompt, which is the current prompt again (both were removed for the ablation below, then restored).
- Caveats: a single seed; the prompts differ (legacy has worked examples); symbolic is scored on its conservative worst case.
- Suggested ablations: the same examples in both prompts; symbolic loop restricted to atomic rules; multiple seeds.

## Catch-all ablation (qwen3:14b, grid_20_balanced, same settings)
- Three symbolic runs: catch-all instruction and example (`symbolic_grid20_capped`); instruction removed, example kept (`symbolic_grid20_nocatchall`); both removed (`symbolic_grid20_noexample`).
- The instruction alone barely mattered: rounds ending in a catch-all went from 77% to 72%, because qwen copied the example rule. Removing the example too halved it (37%; 10 of 20 final policies vs 18).
- With no catch-all at all: uncovered situations 26% of reachable (vs 9%), requirements met in worst case 4.50 (vs 5.25), shortfall 2.62 (vs 2.11). Best case 5.25, so the best-to-worst gap grew to 0.75 requirements. Still ahead of legacy (4.30, 3.16). Extend ran 11 times but improved only 2; refine success fell to 23%.
- Reading: with this model, partial policies are a liability under the conservative worst case, and extend feedback doesn't yet fill coverage well. The TODO to expose `obs_idx` to rules would shrink the worst-case penalty for partial policies. Worth revisiting with a stronger model.
- Figures: `viz/figures/catchall_ablation.png` (instruction only) and `viz/figures/catchall_ablation_full.png` (instruction and example).
- After the ablation, the catch-all instruction and example were **restored**, since they give qwen better results. Re-run without them when studying extend, or with a stronger model.
