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
- **Kept** `out/results/100-balanced-paper-results/` (the paper's numbers and plots). It's data, not code, and handy for reference. **[REVIEW]** Delete it if you want the branch fully barebones.
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
- **Policy-visible variables** are declared in the spec (ranges read from the model). For gridworld these are `x, y, g1..gN`. The obstacle phase `obs_idx` is hidden, matching the legacy per-cell policies. Consequence **[REVIEW]**: the worst-case adversary *can* see `obs_idx` (it resolves uncovered states per full MDP state), so worst case is conservative relative to observation-based completions. That's sound, just pessimistic.
- The gridworld MDP was rewritten as a symbolic PRISM template (formulas for move/slip/bounce) rather than enumerating cells. `tests/test_gridworld_equivalence.py` and `src/regression.py` confirm it is equivalent to the legacy DTMC to within 1e-8.

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
- **Blind retry**: after `stall_limit=2` consecutive non-improving attempts, the next attempt uses the fresh initial prompt. This follows the "retry is good / try@k" slide. **[REVIEW]** In the smoke test qwen returned an identical answer to an identical refine prompt, so `stall_limit=1` may be better.
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
- Two checks, since PRISM's default convergence (legacy used `-power` with default epsilon) leaves ~1e-5 error in stored values:
  1. *stored*: new pipeline (default settings) vs values the legacy run reported, tolerance 1e-4.
  2. *exact*: legacy DTMC recomputed vs new MDP, both at epsilon 1e-10, tolerance 1e-7. This is the real model-equivalence test.
- Every iteration's policy of every sample is checked, not just the final one (the legacy planner now records the full policy at every iteration).

## Comparison
- Same model, same 20 grids, `max_attempts=5`, 2 concurrent workers for both runs, run one after the other (not concurrently) so they don't contend for the GPU.
- LLM time is client-side wall time for both, so it includes queueing behind the other worker. Tokens are contention-free and the fairer cost measure.
- The prompts differ in content by design. Legacy has two long worked examples with reasoning. Symbolic has a short rule-syntax example block. I did not port the worked examples. **[REVIEW]**
- The legacy "final" probabilities are those of its kept-best policy (reconstructed with its own keep-best rule for runs that predate the `final_prism_probs` field).
