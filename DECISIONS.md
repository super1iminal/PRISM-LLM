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

## TODO
- Per-domain switch to expose extra variables to rules (e.g. `obs_idx` in gridworld), which makes the worst case exact. Keep it off for legacy comparisons.

## Results (qwen3:14b, grid_20_balanced, 5 attempts, thinking off)
- `out/results/comparison_grid20/report.md`, `viz/figures/grid20.png`. Symbolic numbers come from the **capped** run (`symbolic_grid20_capped`).
- Success 0/20 legacy vs 1/20 symbolic (worst case). Mean requirements met 4.30 vs 5.25. Shortfall 3.16 vs 2.11. Output tokens 12.2k vs 5.0k. Wall time 415 s vs 170 s. LLM calls 15 vs 5.
- Symbolic output tokens stay flat with grid size (~4–6k); legacy's grow from 5.4k (4x4) to 20.3k (8x8).
- The two are roughly tied on 4x4 grids.
- Loop usage: refine 63 attempts (35% improved the best so far), blind retry 10 (60%), extend 4 (100%). qwen nearly always wrote a catch-all rule, so policies were almost complete and extend rarely ran.
- These results were produced with `stall_limit=2` and the catch-all nudge in the prompt. Both have since changed (`stall_limit=1`, nudge commented out in `_problem.md.j2`), so a re-run would differ.
- Caveats: a single seed; the prompts differ (legacy has worked examples); symbolic is scored on its conservative worst case.
- Suggested ablations: the same examples in both prompts; symbolic loop restricted to atomic rules; multiple seeds.
