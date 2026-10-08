# Decisions log

Design decisions and results, to review together. Each entry has the decision and why. Items marked **[REVIEW]** are the ones I'm least sure about.

Paper target: **SEAMS 2027** (research track Oct 23; check the official site) or **ICAPS 2027** (Dec 7 abstract), decided at Marsha's **Oct 10 checkpoint**: if rules transfer on gridworld with a stronger model and relative features, SEAMS; if not, ICAPS, where generalized policies are a core topic. Write first, then finish. Rajan helps with implementation.

## Story: Option A, narrowed (Marsha, Oct 8 email)
- **Committed to Option A; Option B (model engineering) is parked.** B has the attribution problem (if the LLM writes model and policy, we can't tell which failed, and there is no ground truth), would compete with LLM-as-domain-generator work at ICAPS, and would split effort across two unfinished papers.
- **The claim:** one readable controller for a parametric family of instances, with probabilistic guarantees checked per instance: "certified on every instance we check, including held-out larger ones", not correctness for all sizes. "PRISM and related work handle one instance at a time" won't survive review (families of MDPs, multi-environment MDPs, LLM-written generalized policies exist). Large instances alone aren't a gap either: checking a policy on a big model costs about as much as synthesizing the optimum on it.
- **Main design step: relative features.** Rules over absolute coordinates (`x = 1 & y = 2 -> left`) can't transfer; rules need features such as the direction to the next goal or an adjacent obstacle.
- **Experiments:** gridworld: generate rules on a few grids, freeze them, verify on held-out and larger grids (up to about 20x20). UUV: one rule set for both North Sea and Caribbean, plus a sweep over failure rates ("one certified controller across changing conditions, without re-synthesis on board"). No ATC (no PRISM model, too little time). Baselines: the per-instance PRISM optimum, and a strategy synthesized on one instance applied to the others.
- **Motivation (SEAMS):** adaptive systems that need a certified controller as conditions change, not agents misusing tools.
- **On hold** until Asher's related-work survey is done.
- An **energy budget is expressible in PRISM** (the paper's Table 2 uses rewards). Reward requirements close a gap in *our core*, not in PRISM, so they don't support an expressiveness argument by themselves.

## Positioning checks before the transfer experiment (Oct 8)
Details in `docs/positioning.md` (the related-work survey's phase 2; the survey is `docs/related_work_survey.md`).
- **The position is an intersection.** No work combines LLM-written rules, a stochastic model with several thresholds, exact certification on every instance (held-out and larger ones included) and quantitative feedback; every neighbour has two or three of these. The closest is **1–2–3–Go!** (VMCAI 2025): trees over raw state variables, learned from small instances, certified per larger instance; single reachability objective; parameters can't appear in predicates; no loop, no LLM.
- **The rule-list form is not ours:** Fern, Yoon and Givan (JAIR 2006) learned first-match decision lists for stochastic PPDDL domains on small instances, judged by simulation.
- **Hand-written template on gridworld** (`domains/gridworld/data/template_policy.py`; one goal-relative, wall-avoiding rule list for every grid): certifies 10/20 grids of `grid_20_balanced` and 10/20 of `grid_20_large`, against 16/19 and 14/20 for Haiku 5.5's per-instance rules. On 2–3 grids per set it is trapped for good (memoryless navigation in a pocket); the other failures are sequence requirements (slips into future goals) and the moving obstacle. Transfer on gridworld is neither trivial nor hopeless.
- **The UUV failure-rate family is trivial** (`domains/uuv/data/failure_sweep.py`): with each member's thresholds keeping the paper scenario's slack, the paper's `stay` controller meets every requirement for failure rates 0.5×–3× in both scenarios, and `always_med` covers every North Sea member.
- **Wording:** "certified on every instance we check" and "sound" (every allowed policy meets every threshold), never "maximally permissive" (CAV 2026: strong safety and strong permissiveness can't both hold for probabilistic safety).
- **[REVIEW]** Transfer design: expose instance constants (goal coordinates, grid size) and generic sensors (`wall_up`) to the rule language so the LLM writes the relative conditions itself; no shortest-path feature (it would make the policy trivial); compare with 1–2–3–Go! on threshold satisfaction per instance.
- **[REVIEW]** UUV: drop the failure-rate family, or find one where no simple policy passes (vary visibility ranges, deadlines, inspection lengths), checked with `failure_sweep.py`-style PRISM runs before any LLM run.

## Setup (answers from kickoff)
- The symbolic approach makes **one LLM call over the whole state space** (no per-goal decomposition). Rules may refer to progress variables such as `g1`.
- Rule overlap: **ordered decision list, first match wins**.
- Feedback for missing rules **locates** high-mass uncovered states but never reveals PRISM's optimal action.
- qwen3 runs with **thinking off**, on **20 grids**.

## Scope of this branch
- This branch holds the symbolic approach and one baseline. The predecessor's other planners (Vanilla, VanillaPlus, Feedback, FeedbackMinus, RL), their plotting scripts and the predecessor paper's code are on `main`.
- The baseline is FeedbackSimplified in `src/legacy/` (gridworld only): the predecessor's planner plus instrumentation (per-iteration policies, raw outputs, tokens) and the obstacle-visibility switch (`domain.visible_extra`).
- `grid_20_balanced.csv` = the first 4 grids of each size (4–8) from `grid_50_balanced.csv`, so it is balanced by size.
- `grid_20_large.csv` = the first 4 grids of each size 9–13 from `grid_100_balanced.csv` (same generator; its sizes 4–8 start with exactly `grid_20_balanced`'s grids). Made because Sonnet 5.5 reaches the ceiling on `grid_20_balanced`, leaving no headroom for the ablations.

## Generalization: inputs (what a case study provides)
- A case study is a directory `domains/<name>/` with a `Domain` subclass (instance loading and template context) plus Jinja2 templates. See `domains/README.md`.
- **MDP fully user-specified in PRISM** (`model.prism.j2`). Policy actions are the *action labels* of the commands. The planner never generates dynamics, only a `policy` module that synchronizes on those labels.
- **Requirements** are user-specified as PRISM path formulas with a bound (`>=`/`<=`) and a threshold, plus English text for the prompt. Best and worst case are `Pmax`/`Pmin` (swapped for `<=`).
- **English description** (`description.md.j2`) and **visual** (`visual.txt.j2`) are separate user templates injected into generic core prompts. A domain can override any core prompt template by dropping a file with the same name in its directory.
- **Policy-visible variables** are declared in the spec (ranges read from the model). For gridworld these are `x, y, g1..gN`. Config `domain.visible_extra` adds more: by default the obstacle phase `obs_idx`, for rules and legacy alike (`pre_phase_a` hides it).
- The gridworld MDP is a symbolic PRISM template (formulas for move/slip/bounce) rather than an enumeration of cells. `tests/test_gridworld_equivalence.py` and `src/regression.py` confirm it is equivalent to the legacy DTMC (5e-10 across 100 policies under interval iteration).

## Generalization: symbolic policies
- Output: `{"rules": [{"condition", "action"}]}` via JSON-schema structured output. The action is an enum of the domain's labels.
- Condition language: a small PRISM-compatible expression subset (`= != < <= > >= + - & | ! =>`, parentheses, `true/false`), parsed and type-checked in Python. It is lenient about common LLM spellings (`==`, `&&`, `and`, `True`, and a trailing `-> action`, which qwen writes inside conditions).
- Invalid answers are re-asked with the error message, up to 2 extra calls per attempt. If still invalid, the attempt counts as an empty policy.
- **First-match compilation**: rule *i*'s effective guard is `c_i & !c_j` for only those earlier rules *j* that can overlap it. Overlaps are found by enumerating the policy-visible space when it is ≤ 200k valuations, and otherwise every earlier rule is negated. This keeps 500-atom legacy policies compact.
- Legacy per-state policies are expressible as one atomic rule per state (`core.rules.atomic_rules`, `domains/gridworld/legacy_translate.py`).
- The rule listing shown back to the LLM uses the same JSON-per-line shape as its output, and so do the example rules (qwen copies the `cond -> action` notation of examples into the condition field).

## Refinement loop
- Success means **every requirement holds in the worst case**. The best-case pass rate is reported separately.
- The switch follows the slides. If best case fails anything → **refine** (the LLM returns a complete new rule list). If only worst case fails → **extend** (the LLM returns *only new rules*, **appended** after existing ones). Appending means existing decisions are unchanged, so worst case can only go up. Best case can go down, which then triggers refine.
- **Keep-best** like legacy: score = (#worst-case failures, #best-case failures, total worst shortfall, total best shortfall). Feedback is always built from the best policy so far.
- `max_attempts=5` generation rounds, the same number as legacy. Legacy makes one call per goal and obstacle phase per round (one per goal when the phase is hidden). Symbolic makes one per round, plus re-asks for invalid answers.
- Per instance, one extra PRISM run computes the unconstrained optimum of each requirement on the bare MDP. It is used for refine-mode blame and reported as an upper bound on what is achievable.

## Feedback: probability mass
- V^opt stays in the loop (Sept 24 meeting); one-step regret is the S5 ablation.
- Rejected: a performance-difference decomposition (visits × local advantage). It gives ~0 mass when the adversary *stalls* the agent in loops through uncovered states, the main failure mode in testing. Without discounting or absorbing targets, the lemma's residual term doesn't vanish.
- Used: **mass(s) = expected visits to s within H steps × probability still at stake at s** (H = `feedback.horizon`: 100 for gridworld, the mission deadline for UUV), reported as a share of the total.
  - extend: strategy = greedy worst-case completion, stake = best(s) − worst(s), counted only on uncovered states and aggregated by policy-visible valuation (top 10 shown).
  - refine: strategy = greedy best-case completion, stake = optimum(s) − best(s), charged to the rule deciding s (top 10 rules, each with its top 3 states).
- Greedy strategies are computed from PRISM's per-state value vectors (`filter(printall, ...)`) and the exported transition matrix. For LTL (non-reachability) formulas those vectors are PRISM's projection from the product automaton, so the ranking is a heuristic there.
- Feedback **locates** states and rules but never states PRISM's optimal action (per the kickoff answer).
- The prompt table shows best, worst and threshold for every requirement, plus coverage (reachable situations vs uncovered).

## Regression
- Two checks:
  1. *stored*: the symbolic pipeline's best case at default PRISM settings vs the values the legacy run reported (tolerance 1e-9). Same solver, so they match to rounding (observed 1e-15).
  2. *exact*: legacy DTMC recomputed vs the domain MDP's best **and** worst, all with **interval iteration** (sound error bounds, epsilon 1e-9, tolerance 1e-7). This is the real model-equivalence test.
- Why interval iteration: at default settings (and even at epsilon 1e-10 with plain value iteration), PRISM's worst case (Pmin) for the nested-until sequence properties stops early, by up to 4e-3 at default settings and 5e-6 at 1e-10. At 1e-10 one legacy policy doesn't converge at all. With interval iteration, best and worst agree to 1e-10.
- Side finding: the legacy run's own reported sequence-property values carry up to ~1e-3 solver error at default settings. That's negligible for the paper's conclusions, but worth knowing.
- Solver fallback: for one of the 100 legacy policies (grid 9, attempt 4), interval iteration does not converge at any epsilon (non-progressing loops). That policy falls back to Gauss-Seidel at epsilon 1e-12, where legacy and symbolic agree to 2e-11. PRISM's `-exact` engine can't be used because it doesn't support LTL formulas.
- Every iteration's policy of every sample is checked, not just the final one (the legacy planner records the full policy at every iteration). Settings: `configs/regression/default.yaml`.

## Comparison
- Same model, same 20 grids, `max_attempts=5`, 2 concurrent workers for both runs, run one after the other (not concurrently) so they don't contend for the GPU.
- LLM time is client-side wall time for both, so it includes queueing behind the other worker. Tokens are contention-free and the fairer cost measure.
- The prompts differ in content by design. Legacy has two long worked examples with reasoning. Symbolic has a short rule-syntax example block; the worked examples are not ported. Examples are an ablation (S4 symbolic, L2 legacy; `docs/ablations.md`).
- The legacy "final" probabilities are those of its kept-best policy (replayed with its own keep-best rule for runs without the `final_prism_probs` field).

## UUV case study (`domains/uuv/`)
- Source: Päßler et al., iFM 2023 (arXiv:2308.14663). The paper omits most probabilities, so the MDP follows its ProFeat artifact (branch `scp-ifm_artifact` of `remaro-network/auv_profeat`, Apache-2.0), not the extended model on that repo's main branch (with sonar/camera).
- **Domain-only modelling.** ProFeat features become state variables (`follow`, `alt`). The policy is the paper's feature controller: the actions `low`/`med`/`high` pick the next altitude in search-phase states. Forced transitions use a `[step]` label, which the policy module doesn't synchronize on.
- Visibility limits are modelled by **clamping**: requesting a higher altitude than allowed gives the highest allowed one. So every action is enabled everywhere (no deadlocks), and each state's set of distinct choices is exactly the paper's. Check: the bare MDP reproduces all reported numbers for both scenarios (Pmin F done = 1; Table 2 energy/time min/max; Pmin G safe = 0.65 / 0.32). The paper's `time`/`energy` reward structures are part of the model for this check; the planner's energy requirement uses `energy`.
- Policy-visible: `s`, `alt`, `water_visib` (what the paper's controller monitors). Hidden: `follow`, `d_insp`, `t_failed`. `t_failed` is always 0 while searching.
- Requirements: `no_thruster_failure` = `G !"thruster_failure"` (the paper's `G "safe"`), `done_in_time` = `F<=T "done"` (the paper's time reward turned into a deadline), and `energy_budget`, an expected-energy bound (A3 below).
- **[REVIEW] Thresholds are tight by design.** In this model the controller has little leverage. Over all controllers, safety ranges over [0.654, 0.674] (North Sea) and [0.321, 0.355] (Caribbean). Done-in-time ranges over [0.52, 0.82] (T=30) and [0.06, 0.87] (T=70), and the low end is the adversary switching altitude back and forth. Thresholds (North Sea 0.670 / 0.80, Caribbean 0.345 / 0.862) are set so that "hold your altitude, go highest when a search starts" passes. Always-low, always-high (and, in the Caribbean, always-med) each fail at least one requirement. The best rule policy found by local search clears both by only ~0.003 / 0.002. Reproduce with `src/calibrate.py --domain uuv --data uuv_paper.csv`. Looser thresholds would make any sensible complete policy pass, so the case study would then mostly test coverage.
- PRISM's multi-objective query (`multi(Pmax [F<=T], P>=p [G ..])`) gives a value below a concrete policy's (0.795 vs 0.807), so it isn't used for calibration.
- `done_in_time` is step-bounded, so the per-state vectors used by the mass analysis assume the full T steps remain from every state. The ranking is a heuristic there, as for LTL.

## UUV results (qwen3:14b, uuv_paper, 5 attempts, thinking off)
- `out/results/symbolic_uuv`: a run on the two probability requirements (no energy budget). Figures `viz/figures/uuv.png` (`viz/plot_domain.py`, which works for any domain and plots final policies against the domain's `reference_policies.json` and the range any controller achieves) and `viz/figures/uuv_summary.png`.
- North Sea: success in 2 attempts (0.6723 / 0.8064). Caribbean: fails. The final policy behaves like always-high (0.348 / 0.860 vs 0.862 needed).
- There is no legacy baseline for UUV (legacy is gridworld-only). The non-LLM reference to beat is `stay`, the only reference policy that passes both scenarios, energy budget included (26.43, 62.17).
- Comparison with the paper (`viz/plot_uuv_summary.py` → `viz/figures/uuv_summary.png` and `.md`):
  - The paper doesn't synthesize a controller. Its managing subsystem is nondeterministic, and its best case per requirement is PRISM's separate maximum. No single controller reaches both best cases (always-high maximizes safety but misses the North Sea deadline).
  - Ours is within 0.002 (safety) and 0.012 (deadline) of those separate maxima.
  - On the paper's Table 2 measures, our North Sea policy needs 26.04 energy and 23.93 time to finish, against the paper's best possible 24.78 and 23.66.
  - Energy budget (the rules re-verified against all three requirements): the North Sea policy is within budget (26.04 ≤ 26.5), the Caribbean one is not (62.99 > 62.5).
  - Size: 25 rules vs ≥1,620 (North Sea) and ≥8,850 (Caribbean) states that an optimal PRISM strategy must decide (reachable states with more than one distinct choice). For the step-bounded deadline, the optimal strategy also depends on the step count.
  - Transfer: the North Sea rules reused unchanged on the Caribbean keep safety (0.347) but not the deadline (0.824).

## UUV optimization (branch `optimization`, archived on GitHub, not merged)
- Its runs use another configuration (no energy requirement, among others), so their numbers aren't comparable with the runs above. Target: match `stay` (passes both scenarios). **Not reached.** North Sea passed reliably; Caribbean failed in all 12 runs by ~0.002. Figure and runs (`out/results/opt/e*/r*`, 1–2 repeats each) live on that branch.
- E1 `keep` action: unused by qwen. E4 more explanation (switching arithmetic, requirement drivers): worse (0/4), not kept. E7 altitude facts in action text: no change.
- E2 found a **domain bug**: with the visibility thresholds given as decimals (`< 5.67`), qwen copies them into conditions, and the integer rule language rejects them. The UUV description lists the integer levels of each band instead (E3).
- **[REVIEW] Not in this branch, need a decision:**
  - E5 (core): refine/extend feedback lists every previous attempt's worst case and flags an answer that verifies identically to an earlier one. Didn't help on UUV; never tested on gridworld.
  - E6: hide `water_visib` from the policy (limits are enforced by clamping anyway). The most useful variant: qwen writes "hold the altitude" rules over 4–11 situations instead of per-band rules (which act like always-high), but starts low or med at `s = 11`, where the Caribbean needs high. It contradicts "policy sees what the paper's controller monitors".

## Pac-Man case study (`domains/pacman/`, Oct 8)
- **Source:** the Quantitative Verification Benchmark Set's Pac-Man (`benchmarks/mdp/pacman`, version 2; Junges and Könighofer, from the probabilistic-shields paper). `model.prism.j2` is the benchmark's `pacman.nm` unchanged except for `MAXSTEPS`, set per instance. Chosen because 1–2–3–Go! fails on it (the horizon is a parameter its trees cannot use) and because Pac-Man's choices are labelled actions that our policy module can synchronize on.
- **Fidelity:** the bare MDP reproduces the benchmark's published state count (498) and minimum crash probability (0.5511) for `MAXSTEPS = 5` (`tests/test_pacman.py`).
- **Calibration** (`src/calibrate.py --domain pacman --data pacman.csv`): the minimum crash probability is 0.5511 at every horizon, so the first moves decide it; the maximum grows (0.88 at 8–10 moves, 1.0 from 12). Requirement `crash` ≤ 0.56. At 8–10 moves simple policies (keep the heading, always right) pass; from 12 moves every reference policy fails (0.69–0.89). Dataset `pacman.csv`: horizons 8, 10, 12, 14, 16. States: 6.9k (10), 97k (15), 883k (20), 3.7M (25), so the loop runs on short horizons and longer ones are for certifying frozen rules.
- **Core changes, generic:**
  - The model parser reads untyped constants (`const xSize = 11;`, PRISM's default int), which the benchmark file uses.
  - **Decision states count only choices labelled with a policy action.** Once both ghosts are gone, the arbiter's unlabelled idle step competes with the forced `[p]` move. That is environment nondeterminism the policy cannot touch, but the old check (any two choices with different successors) counted it as a decision. Gridworld (every choice is a policy action) and UUV (forced states have only `[step]`) are unaffected. `docs/semantics.md` updated.
  - `src/calibrate.py --domain <name> --data <dataset>` replaces UUV's `domains/uuv/data/calibrate.py`; it reads each domain's `data/reference_policies.json`.
- **LLM runs** (default settings, restart on slow progress; one unseeded run each; `haiku_pacman`, `sonnet5_5_pacman`):

  | horizon | 8 | 10 | 12 | 14 | 16 |
  |---|---|---|---|---|---|
  | Haiku 5.5 | 0.5511 (1 round) | 0.5511 (1) | 0.8911 | 0.8911 | 0.8911 |
  | Sonnet 5.5 | 0.5511 (1) | 0.5511 (1) | 0.5511 (1) | 0.5511 (5) | 0.8658 |

  Worst-case crash probability of the final rules (≤ 0.56 passes; the optimum is 0.5511). Sonnet's passing rule sets have 1–6 rules. Haiku's failures equal the "keep heading" reference policy (0.8911). Costs: about $0.02 (Haiku) and $0.40 (Sonnet).
- **Against the papers:** the QVBS optimum is 0.5511 at every horizon (Storm; our PRISM agrees). 1–2–3–Go! learns its tree from horizon 5, where every controller gives 0.5511, and reports 0.869 at horizon 25 and 0.99998 at 200 (Smart LSS 0.729 at 25; random policies 0.923).
- **Transfer of Sonnet's rule sets across horizons** (`src/transfer.py` from branch `rule-language`; the map is fixed, so base-vocabulary rules mean the same at every horizon; `out/results/ablations/sonnet5_5_pacman/seed_none_1/transfer_pacman_transfer.csv`): the rule sets from horizons 8, 12 and 14 stay at 0.5511 on horizons 8–14, then fail from 16 on (0.91–0.99 at 16–20, 0.99 at 25). So they do not reach 1–2–3–Go!'s 0.869 at horizon 25.

## Forced states (core, motivated by UUV)
- In UUV, the action has no effect in about 70% of states (following, found, done). Without special handling they would count as uncovered and appear in feedback.
- `PolicyVerifier.forced_states()` finds the states of the bare MDP where every choice has the same successor distribution (probabilities rounded to 1e-12). It reuses the optimum run's exported transitions, or else runs PRISM once with no properties.
- Forced states don't count as situations: `reachable_situations`/`uncovered_situations` count only valuations with at least one reachable decision state. They're also skipped in extend hotspots **and** refine blame, because no rule can change what happens there. `Verification.decisions` marks decision states.
- Gridworld has no forced states (checked on all 20 grids), so this doesn't affect it. UUV North Sea: a search-only policy is complete (0 of 85 situations uncovered).

## Sept 24 meeting (Marsha, Aren)
Outcomes: the paper target (top), the retry sweep and seeds (`docs/ablations.md`), and:
- The story is open (pinned at the top). Generalization and readability are "a nice bonus", not the reason.
- **V^opt stays in the loop**; regret is an ablation (S5), not a replacement. Ceilings are "foundational", not an experiment.
- **Baselines are per case study** (e.g. RL for gridworld, the paper's controller for UUV). Rule transfer was dropped here; the Oct 8 email reverses this, and transfer is now the main experiment (Story, top).

## Oct 8 email (Marsha): the ablation results and what to rerun
- **What the results don't show yet.** Feedback is not shown to beat restarting: pure resampling (R5) meets 4.72 requirements vs 4.45 for the ablations' reference loop (B2, restart after 2 stalls; the default retry is now restart on slow progress, R4), with about half the input tokens (8.3k vs 16.8k median). A success-vs-budget curve that rises every round is guaranteed by keep-best (any sampler that keeps its best attempt is monotone), so it is not evidence that feedback helps; pure resampling's curve has to be on the same plot. With 10 comparisons and 2 seeds, one p ≈ 0.05 is what chance gives, and nothing survives a multiple-comparison correction. Withdrawn: "restarting matters a lot", "feedback quality matters", "the budget curve is the first direct evidence that feedback helps".
- **The rerun:** only restart after 2 stalls (B2), restart on slow progress (R4, the default), pure resampling and legacy, on a stronger model. Asher's plan:
  - **Headroom:** Sonnet 5.5 solves every solvable `grid_20_balanced` grid, 15 in round 1, where all retry policies send the same prompt. It also solves 13x13 grids from `grid_20_large`; Haiku 5.5 leaves some headroom on `grid_20_balanced` (results below, "Headroom for the rerun"). Chosen (Asher): **Haiku 5.5 on `grid_20_large`** ("Haiku ablation on grid_20_large", below).
  - **Two runs per condition**, not 5. Claude endpoints take no seed, so repeats are unseeded (`run_ablation.py --unseeded N`, `seed_none_<k>/`).
  - **Legacy on qwen only** (on a hosted model it would be expensive: one call per goal and obstacle phase per round).
  - Asher chose to run it now, with the current vocabulary (absolute coordinates), rather than after relative features.
  - **Equal iterations and equal cost, both** (Asher preferred iterations, Marsha asked for tokens): pure resampling gets 10 rounds, so both comparisons come from the same runs.
- **Feedback rounds cost more than fresh ones**, so equal iterations is not equal cost. Median per round: qwen (restart after 2 stalls) 5.7k tokens for a refine round vs 2.7k for a fresh one (input + output); Haiku 5.5 on `grid_20_balanced` $0.0043 vs $0.0028, because it also thinks longer on refine prompts (7.6k vs 5.0k output tokens).
- **qwen at equal tokens** (`viz/figures/budget_tokens_qwen.png`, `configs/plot/budget_tokens_qwen.yaml`; exploratory, the runs were seen first): pure resampling is above restart after 2 stalls at every spend. It reaches 4.72 at 15.1k tokens, where restart after 2 stalls is between 3.85 (14.2k) and 4.25 (18.9k); that one reaches 4.45 only at 24.0k. Restart on slow progress is level with pure resampling where both have data (4.77 at 15.6k) and goes on to 4.95 at 19.2k. The main ablation page has the same curves as a second row, and its intro now calls its tests exploratory.
- **Tooling:** `plot_budget.py` and `ablation_summary.py` take a `cost` setting (spend = input tokens × weight + output tokens × weight) and place each round budget k at the mean spend through round k; legacy runs log only per-run token totals, so they have no spend curve. `ablation_summary.py` runs a page's `planned` comparisons (`at: rounds:N` or `cost`) and Holm-corrects them together; they replace the paired panel and get their own table.
- **Marsha's legacy check:** the legacy run so far (`legacy_grid20`) is from the old setup with the obstacle phase hidden (`pre_phase_a`). Its budget curve (4.30) against symbolic's (5.25) compares two runs in that same setup, so that pairing is matched; legacy beside the ablation conditions (obstacle visible) is not. **B1** (legacy, obstacle visible, qwen, seeds 1 and 2) started Oct 8. Some hosted-model runs and PRISM-only ceilings overlap it on the CPU (not the GPU), so its first hour's wall times may read slightly high; tokens are unaffected.

## Results (qwen3:14b, grid_20_balanced, 5 attempts, thinking off)
- `out/results/comparison_grid20/report.md`, `viz/figures/grid20.png`. Symbolic numbers come from the **capped** run (`symbolic_grid20_capped`).
- Success 0/20 legacy vs 1/20 symbolic (worst case). Mean requirements met 4.30 vs 5.25. Shortfall 3.16 vs 2.11. Output tokens 12.2k vs 5.0k. Wall time 415 s vs 170 s. LLM calls 15 vs 5.
- Symbolic output tokens stay flat with grid size (~4–6k); legacy's grow from 5.4k (4x4) to 20.3k (8x8).
- The two are roughly tied on 4x4 grids.
- Loop usage: refine 63 attempts (35% improved the best so far), blind retry 10 (60%), extend 4 (100%). qwen nearly always wrote a catch-all rule, so policies were almost complete and extend rarely ran.
- The prompt has the catch-all instruction and example (the default prompt).
- Caveats: a single seed; the obstacle phase hidden from both (the `pre_phase_a` setting); the prompts differ (legacy has worked examples); symbolic is scored on its conservative worst case. The 2-seed ablation runs (`docs/ablations.md`) have the obstacle visible.

## Catch-all ablation (qwen3:14b, grid_20_balanced, same settings)
- Three symbolic runs: catch-all instruction and example (`symbolic_grid20_capped`); instruction removed, example kept (`symbolic_grid20_nocatchall`); both removed (`symbolic_grid20_noexample`).
- The instruction alone barely matters: rounds ending in a catch-all are 77% with it and 72% without, because qwen copies the example rule. Removing the example too halves it (37%; 10 of 20 final policies vs 18).
- With no catch-all at all: uncovered situations 26% of reachable (vs 9%), requirements met in worst case 4.50 (vs 5.25), shortfall 2.62 (vs 2.11). Best case 5.25, so the best-to-worst gap is 0.75 requirements. Still ahead of legacy (4.30, 3.16). Extend ran 11 times but improved only 2; refine success is 23%.
- Reading: with this model, partial policies are a liability under the conservative worst case, and extend feedback doesn't fill coverage well. Exposing `obs_idx` (the default) should shrink the worst-case penalty for partial policies. Worth revisiting with a stronger model.
- Figures: `viz/figures/catchall_ablation.png` (instruction only) and `viz/figures/catchall_ablation_full.png` (instruction and example).
- The default prompt keeps both (`prompt.catch_all_instruction`, `prompt.examples`), since they give qwen better results. Run without them when studying extend, or with a stronger model.

## Ceilings and budget curves
- Ceilings (`src/ceilings.py`, `out/results/ceilings/`): what any controller achieves on the bare MDP. Gridworld: 19/20 grids jointly solvable, and almost every requirement's optimum is 1.0. Grid 17 is not solvable: `complete_sequence` can reach at most 0.70 against a 0.80 threshold. UUV: joint undecided (step-bounded deadline), but `stay` is a witness that both scenarios are solvable, energy budget included.
- The single-seed runs against the ceilings (`src/compare.py`): solved of solvable 0/19 legacy vs 1/19 symbolic.
- Budget curves of the single-seed runs (`viz/figures/budget_grid20.png`; the result with budget k is the kept policy after round k): symbolic starts behind (3.25 vs 4.00 requirements met after 1 round) but improves every round to 5.25; legacy plateaus at 4.30 after round 3.

## Solvers, joint queries and seeds
- **PRISM solver: Gauss-Seidel** (`prism.method`). With `obs_idx` readable, a rule set can make the induced chain periodic. PRISM's default value iteration then fails to converge, which would crash an instance mid-run (a three-rule test policy shows it). Gauss-Seidel and policy iteration agree on those models. Checks against published or recorded numbers use PRISM's defaults: the UUV paper's numbers (Gauss-Seidel gives 4726.0 for the Caribbean max energy, vs the reported 4723.29; `tests/test_uuv.py`, `configs/plot/uuv_summary.yaml`) and the regression's stored-values check.
- **Joint queries use exact LP** (`prism.multi_method: lp`); value iteration has the same convergence problem there. LP can't handle step-bounded objectives (UUV's deadline). There, PRISM's value-iteration fallback wrongly says "not jointly feasible" even though the `stay` policy passes both scenarios. So the joint query returns *undecided*, and the loop falls back to the per-requirement branch. **[REVIEW]** This means the joint branch applies to gridworld but not UUV.
- **Legacy with the obstacle visible:** one call per (goal, obstacle phase); a short note in the prompt says which phase the per-cell policy is for. The legacy DTMC is solved with Gauss-Seidel when phases are observed (power method otherwise). **[REVIEW]** Prompt wording of that note.
- **R4 detail:** a round that doesn't improve the kept policy has gain 0, so R4 also restarts after any non-improving round, not just slow ones.
- **Seeds and solver fallbacks:**
  - Each call uses a seed derived from the run seed and the call's index within the instance: reproducible, but distinct per call. A single fixed seed makes every repeat of a prompt return the identical answer (in a smoke test, R5's five rounds produced 1 distinct output, and B2 3 in 5 rounds).
  - The fallback solver is modified policy iteration (`prism.fallback_methods`). A real qwen rule set (16 rules reading `obs_idx`) converges neither with Gauss-Seidel nor with plain policy iteration; modified policy iteration solves it and matches Gauss-Seidel exactly on the other models tried. If every method fails, the planner scores the round as an empty policy instead of losing the instance. Legacy's verifier retries with `-bgaussseidel` then `-intervaliter` before its silent all-zero fallback. **[REVIEW]** The legacy fallbacks are untested on a real non-converging case.

## A3: reward requirements (UUV energy)
- Core: a requirement may bound an expected reward (`reward: energy` in the spec, PRISM `R{"energy"}<=c [ F "done" ]`). Best and worst are `Rmin`/`Rmax`. Shortfall and blame stakes are taken relative to the threshold, so energy doesn't swamp the probability terms when they are summed. Infinite expected rewards (goal not surely reached) are capped in the mass analysis.
- UUV has `energy_budget` (dataset column `energy_threshold`, optional), and the description spells out the paper's energy costs. Calibrated like the other thresholds (`src/calibrate.py`), so `stay` passes:
  - North Sea: ≤ 26.5. `stay` uses 26.43, and `always_high` (27.13) fails it. `always_med` (25.98) passes everything.
  - Caribbean: ≤ 62.5. `stay` uses 62.17, and `always_high` (62.99) fails it.
  - Bare-MDP best and worst match the paper's Table 2 (24.78 / 59.08 min).
  - This creates a real trade-off: higher altitude is safer but costs energy.
- The joint query is undecided for UUV (step-bounded deadline), so UUV uses the per-requirement branch.
- The blame/hotspot wording mentions cost only when a reward requirement is failing, so gridworld prompts never mention it.
- **[REVIEW]** Energy thresholds and the description of the energy costs.

## Batch 2 settings
- **UUV mass horizon = the mission deadline** (Asher): U1's run config (`configs/run/conditions/U1.yaml`) sets `feedback.horizon: domain`, which asks `Domain.horizon(instance)`: North Sea 30, Caribbean 70. The run config never names a domain; each kind of run has its own condition file.
- **Story:** A, narrowed by the Oct 8 email (top); B parked.
- **S1 (table feedback):** after a failure, every round shows the results table and the previous rules and asks for a complete new list. No blame, no REFINE/EXTEND framing, no appending. Retry as in B2.
- **S5 (regret blame):** one-step regret of the rule's action on the best-case values, with Q from the bare MDP. Successors the policy never reaches take their optimum value, which slightly favours deviating (a heuristic). Extend ranks uncovered states by the local gap best − worst, without occupancy. The prompt wording describes the signal it shows. **[REVIEW]**
- **V1 (random blame)** is a placebo: same number of rules/states as B2, equal shares, and B2's wording. **V2** drops the blame section but keeps the framing.
- **L1:** legacy's retry uses the same `stall:k` rule as symbolic; a blind-retry round regenerates every (goal, phase) from the initial prompt, then keep-best applies as usual.
- **Run order:** all symbolic conditions first (B2 and R1–R5, then S1, S4, S5, V1, V2, U1 and D7), then legacy.

## Config folders
- `configs/` has one folder per tool (Asher): `run/` (`default.yaml` + `conditions/<name>.yaml`), `ablation/`, `regression/` (each with a `default.yaml`) and `plot/` (one file per figure or report). The run config says what a run does; domains are defined in `domains/<name>/`, not in configs.
- Each `default.yaml` is the only place its defaults live (no Python defaults; a missing key is an error).
- Figures and reports read run facts (dataset, model, round budget) from each run's `config.json`; runs without one read as a fixed record (`results_io.PRE_CONFIG_RUNS`). Metrics use each domain's `Requirement` thresholds and bounds.

## Default retry policy
- **`planner.retry` defaults to `gain:0.05`** (restart when the last round cut total worst-case shortfall by less than 0.05, including no improvement). It won the retry sweep: 4.95 requirements met vs 4.45 for B2's `stall:2` (p = 0.055) and 4.72 for pure resampling (p = 0.31; better on 10 grids, worse on 6).
- B2, S1, S4, S5, V1, V2, U1 and `pre_phase_a` set `retry: "stall:2"` in their condition files, matching each run's `config.json`.
- **D7:** the default retry with 7 rounds, 2 seeds, to see whether more rounds keep paying off (every budget curve rises through round 5).
- The legacy conditions (B1, L1, L2) are not run: lower priority than the symbolic results (`docs/ablation_not_run.png`).

## Default run (qwen3:14b, grid_20_balanced, `configs/run/default.yaml`, no seed)
- `out/results/symbolic_grid20_default`: the default config unchanged, as an end-to-end check of the pipeline. Report `out/results/comparison_grid20_default/report.md`; figures `viz/figures/grid20_default.png` and `viz/figures/budget_grid20_default.png` (with the R4 seeds, which have the same settings).
- Requirements met 5.10 (worst case) vs 4.30 legacy; R4's seeds 4.95 (4.80 to 5.10). Per grid, +0.15 ± 0.88 against R4's seed mean, while R4's two seeds differ by ± 1.6: consistent with R4. Solved 0/20. Shortfall 1.86 vs 3.16. Output tokens 5.8k vs 12.2k. Wall time 186 s vs 415 s per grid.
- Every kept policy covers every reachable situation (best = worst). No invalid answers, errors or solver fallbacks.
- **Solver accuracy:** the final policies re-verified with interval iteration (epsilon 1e-9; Gauss-Seidel at 1e-12 for grid 19, where interval iteration does not converge in 1e8 iterations) change no met/not-met verdict. The largest difference is 0.017 (grid 4, `avoid_moving_seg1`: 0.723 stored, 0.706 exact, threshold 0.70).
  - Gauss-Seidel stops when an iteration changes little, which is not an error bound. Value iteration approaches from below, so values PRISM computes directly (`goal*`, reachability) err low, the safe side for a worst case. Values it computes as 1 − another probability err high: the avoid requirements (`G`, both cases) and the worst case of the sequence requirements (LTL: 1 − Pmax of the negation).
  - On chains that leak probability slowly, the error is large. In R4 seed 1, grid 18's kept policy (fully covering, so best = worst exactly) has `seq_2_before_3` stored as best 0.0044, worst 0.272; interval iteration gives 0.0061. Its `goal3` is stored as 0.587 and is 0.799 exactly. The verdicts are the same there (threshold 0.8), but only just for `goal3`.

## Exact check of the final policy
- **Runs report the final policy's values from interval iteration** (`prism.exact_check`, on by default; Asher), which converges to within `prism.exact_epsilon` (1e-9) of the true values, so success and every reported best and worst case can be trusted. Where interval iteration does not converge in `prism.exact_max_iters`, Gauss-Seidel at 1e-12 is used (tight, but not a bound); if that fails too, the loop's values are reported. Each result records the solver (`final_check`) and the loop's values (`loop_best`, `loop_worst`).
- The loop keeps Gauss-Seidel: interval iteration on every round would cost minutes on slowly leaking chains (R4 seed 1's grid 18 policy takes 347 s), and the loop's decisions only need a ranking. Typical policies take seconds; the default run's 20 final policies took 7 minutes in all.
- Budget curves replay the rounds, so they use the loop's values; their last point can differ slightly from a run's reported numbers.
- Runs without `prism.exact_check` in their `config.json` report the loop's values.

## LLM tasks, backends and lockstep batching
- The symbolic planner never calls a model itself. It builds an **`LLMTask`** (model, messages, JSON schema, generation parameters with the call's seed already resolved, and bookkeeping: instance, round, mode, fixup) and receives an **`LLMResult`** (`src/core/tasks.py`). A **backend** (`src/core/backends/`, chosen by `llm.backend`) turns tasks into results through one method, `execute_batch(tasks) -> results` (same order; a failed task carries `error` instead of raising). `OllamaBackend` sends a batch as concurrent requests, which only run in parallel if the server's `OLLAMA_NUM_PARALLEL` allows it. Scripted backends for tests are in `tests/fakes.py`.
- `planner.solve_steps(instance, logger)` is the refinement loop as a generator: it yields each task and is sent its result. `planner.solve()` drives it one task at a time with the planner's backend, so the loop (prompts, PRISM checks, branches, retries) is the same whichever drives it.
- **`run.scheduler`** picks how instances are interleaved. `threads` (default): `run.workers` threads, each solving one instance and calling the backend itself. `lockstep` (`src/core/scheduler.py`): `run.workers` instances in flight; each step sends the pending task of every active instance as one batch, then resumes every instance with its result (their PRISM work runs in parallel threads), and finished instances free their slot. Lockstep logs every task and result to `<run>/llm_tasks.jsonl` with its batch number. It drives the symbolic planner only; a legacy run with `lockstep` is a config error.
- The legacy planner calls the model through `core/llm.py`'s `LLMClient`, a blocking client on top of the same backends, so only the backend talks to the serving engine.
- Seeds: `seed * 1_000_003 + n`, with n counting the instance's calls (re-asks included). A backend failure ends that instance with the backend's error text.
- Per-round `llm_wall_time`: how long the round waited for its answers, including queueing and, in lockstep, waiting for the rest of the batch. `llm_time` is the tasks' own time in the backend. Lockstep wall-clock times are not comparable with threaded runs; tokens are.
- Tests (`tests/test_scheduler.py`): replays of saved D7, B2, R1 and S1 runs (each from its condition file and seed, exact check off) reproduce every round exactly (prompts, rules, PRISM values, invalid answers), one task at a time and in lockstep; batch size, slot refill, `max_batch` splitting and failure isolation; seeds; and `run_symbolic.run` writing the same parquet under both schedulers. A deliberate change to the prompts or the loop fails the replays; then point them at runs made with the new code. U1 is not replayed: its prompts use an earlier UUV description.
- Design choice: the planner is a **generator** (yield a task, receive a result), chosen over (B) a blocking client behind a barrier, (C) an explicit state machine and (D) asyncio. It keeps the loop readable top to bottom, makes each batch an explicit list, and cannot deadlock. If pausing and resuming *inside* an instance (crash recovery) becomes necessary, the generator's locals can move into a state object (option C) without changing schedulers or backends.
- **[REVIEW]** `threads` is the default, so runs stay comparable with the existing ones; switch only after a lockstep smoke run with the live model.
- **[REVIEW]** Lockstep waits for every active instance before sending a batch, so fast instances idle while slow ones run PRISM. A rolling policy (send what is ready after a short wait) is the alternative if GPU utilization matters more than fixed batches.
- **Smoke runs** against the live server (2 grids, default config, `out/results/smoke/`, not committed): threads and lockstep both complete without errors or invalid answers, with tokens in line with the default run; lockstep sends 10 tasks in 5 batches of 2.
  - Ollama here serves **one request at a time** (`OLLAMA_NUM_PARALLEL` unset): in every batch, one request generates at ~59 tokens/s and the other waits for it. So lockstep takes as long as threads (4:04 vs 4:08), and `run.workers: 2` overlaps one instance's PRISM work with the other's generation, not two generations. A request's `llm_time` includes its wait in Ollama's queue.
  - Batching only pays with parallel slots (`OLLAMA_NUM_PARALLEL` ≥ `run.workers`; a second 16k context needs roughly 2.5 GB more VRAM, more than the 4080 has free next to qwen) or an engine that batches.
- **Open:** which batched engine to target is undecided, so only the interface is general (`BackendInfo.max_batch`, `supports_schema`, `supports_seed`).
- **OpenRouter backend** (`llm.backend: openrouter`, `src/core/backends/openrouter.py`), for running without a local GPU. It uses the `openai` client against OpenRouter's OpenAI-compatible API. A batch goes out as concurrent requests, and the backend is otherwise stateless, so it is safe under both schedulers.
  - The key is read from the variable named by `llm.openrouter.api_key_env` and never enters a config. `llm.model` is an OpenRouter id (`qwen/qwen3-14b`). `num_ctx` is not sent, `num_predict` becomes `max_tokens`, the schema becomes a strict `json_schema` response format, and `llm.think` becomes `reasoning.enabled`.
  - Every request sets `require_parameters`, so only providers honouring the schema and the seed serve it. `llm.openrouter.providers` pins providers, in order, with no fallback.
  - Setup and the smoke test are in `docs/openrouter.md`. Offline tests use a stand-in client (`tests/test_openrouter.py`: request mapping, result order under random delays, per-task failures, concurrent calls).
  - **Smoke test** (`qwen/qwen3-14b` on DeepInfra, fp8, pinned; seed 1; `out/results/smoke/openrouter_*`, not committed): the live request, 1 grid with threads and 2 grids with lockstep all complete with no errors or invalid answers.
    - Thinking is off: 590–2,150 output tokens per call, as in the local runs, and the first round's prompt is 1,645 tokens (local: 1,649). DeepInfra accepts the strict schema.
    - About 75 tokens/s per request (local Ollama: ~59). Lockstep's two requests per batch run in parallel, so 2 grids take 125 s (local lockstep: 4:04); one grid with threads takes 93 s.
    - The three steps cost $0.003.
  - OpenRouter runs are a condition of their own, not comparable with the local runs (Asher: fine): providers serve FP8/BF16 weights, not `q4_K_M`, and hosted seeds are not guaranteed to reproduce samples. Name the model, provider and quantization in their results.
  - `providers: []` lets OpenRouter route each request to any provider, so experiments pin one. DeepInfra (fp8) passed the smoke test; NextBit (int4, closer to the local 4-bit weights) is the other provider of `qwen/qwen3-14b` that supports every parameter sent.

## Sonnet vs qwen (grid_20_balanced, R4's settings)
- **What varied:** only the model (Asher). `sonnet5`: Claude Sonnet 5, thinking off as for qwen. `sonnet5_5`: Claude Sonnet 5.5, thinking on (its endpoints refuse to turn it off) and a 16k output cap, since the cap covers thinking and answer. Both on OpenRouter with Anthropic's endpoint pinned, one unseeded run each (no Sonnet endpoint accepts a seed), against R4 and B2 (qwen3:14b, 2 seeds each). Page: `out/results/sonnet_vs_qwen/SUMMARY.md` (`configs/plot/sonnet_vs_qwen.yaml`; it now has Haiku 5.5 too, below).
- **Results** (worst case; solved of the 19 solvable grids):

  | | solved | req. met | Δ vs R4 [95% CI] | after 1 round |
  |---|---|---|---|---|
  | B2 (qwen) | 0/19 | 4.45 | | 2.98 |
  | R4 (qwen) | 0/19 | 4.95 | | 3.00 |
  | Sonnet 5 | 5/19 | 7.10 | +2.15 [+1.50, +2.77] | 6.10 |
  | Sonnet 5.5 | 19/19 | 8.95 | +4.00 [+3.62, +4.38] | 8.55 |

  - Sonnet 5.5 meets every requirement on every solvable grid, 15 of them in the first round. On grid 17 it meets 8 of 9, the most any controller can (its `complete_sequence` ceiling is 0.70 against 0.80).
  - The sequence requirements separate the models. `complete_sequence` is met on 0% of grids by qwen, 30% by Sonnet 5 and 95% by Sonnet 5.5 (every grid but 17).
  - **Rule count** of the final policy (median, interquartile range): B2 43 (28–60), R4 32.5 (23–56), Sonnet 5 25.5 (23–32), Sonnet 5.5 19.5 (18–27); Sonnet 5.5 solves its 19 grids with 11–35 rules. qwen fills the 64-rule cap in about a third of its answers (69 of 205 for B2, 51 of 207 for R4), Sonnet in none; extending a capped answer gives qwen policies of up to 112 rules. One B2 grid (seed 2, grid 15) keeps the empty rule set: its first answers were all invalid and no later round scored better.
  - **Rules per decision situation** (the situations a per-state lookup table needs one entry for; in gridworld, every reachable state of the bare MDP): median 0.099 (B2), 0.112 (R4), 0.088 (Sonnet 5), 0.062 (Sonnet 5.5), i.e. one rule per 9–16 situations. Rule counts stay roughly flat while grids grow from about 100 to 1,400 situations, so the ratio falls with grid size (8x8: 0.033 for Sonnet 5.5, 0.05–0.06 for qwen).
- **Checks:** each run's `config.json` matches its condition file; no errors or invalid answers; every final policy covers every reachable situation; the exact check (interval iteration) changes no verdict.
- **Costs** (per grid): Sonnet 5 15.2k input / 3.8k output tokens, Sonnet 5.5 4.5k / 5.9k (mostly thinking; it stops early when solved), qwen 13–17k / 6–7k. Tokenizers differ. About $1.37 and $1.36 per 20-grid run (token counts at $2/M input and $10/M output; $2.73 for both on the key); 7 and 10 minutes, with concurrent hosted requests, so not comparable with local timings.
- **Reading:** with a strong model the loop reaches the ceiling, so qwen's results are limited by the model, not by the loop or the verification. Sonnet 5 without thinking is already far ahead of qwen; thinking (with the newer version) closes the rest. Confounds: Sonnet 5.5 differs in both version and thinking, and each Sonnet condition is a single unseeded run.

## Sonnet 5.5 on UUV (U1's settings)
- `U1_sonnet5_5`: U1 with Sonnet 5.5's model settings, one unseeded run. Figures: `viz/figures/uuv_sonnet.png` (against qwen's U1 and the reference policies) and `viz/figures/uuv_summary_sonnet.png` (against the paper's controller).
- **North Sea:** solved in round 3 with 3 rules (qwen's U1: round 1, 25 rules), using less energy (26.01 vs 26.18, budget 26.5).
- **Caribbean:** not solved by either model; both meet only `no_thruster_failure`. qwen's policy reproduces `always_high` (done in time 0.860 vs 0.862 needed, energy 62.99 vs 62.5). Sonnet's is further off (0.824, energy 98.5), and its refine rounds made it worse (energy 170–224), so it kept round 1. The `stay` reference policy meets all three, so the scenario is solvable.
- **Rules per decision situation:** UUV's rules see `s`, `alt` and `water_visib` only, so 162 and 295 situations stand for 5,580 and 29,244 states. Sonnet 5.5's 3 and 6 rules give 0.019 and 0.020; qwen's 25 give 0.15 and 0.085. The paper's controller has 1,620 and 8,850 strategy states.
- 15.5k input / 7.5k output tokens per scenario; $0.21 for the run.
- **Reading:** Sonnet 5.5's lead on gridworld does not carry over to the Caribbean, whose thresholds are calibrated so that only a scenario-aware policy like `stay` passes; the feedback did not lead it there. One run, so read with care.

## Headroom for the rerun: Haiku 5.5 and 13x13 grids (Oct 8)
- **Haiku 5.5 on `grid_20_balanced`** (`haiku5_5`: Sonnet 5.5's settings, only the model differs; one unseeded run, `seed_none_1`). On the Claude vs qwen page (`out/results/sonnet_vs_qwen/SUMMARY.md`, heading now "Claude vs qwen").
  - Solved 16 of 19 solvable grids (Sonnet 5.5 19, Sonnet 5 5, qwen 0); requirements met 8.45 (Sonnet 5.5 8.95, Sonnet 5 7.10, R4 4.95). It misses grid 17 (unsolvable) and grids 10, 11 and 19.
  - Only 7 grids are solved in round 1; after k rounds it meets 7.15, 7.85, 8.40, 8.45, 8.45. So the loop does work there, unlike with Sonnet 5.5 (8.55 after round 1).
  - Final policies: 13–52 rules (median 21), every reachable situation covered. 4 invalid answers, all fixed by re-asks. The exact check changes no verdict.
  - 10.5k input / 18.2k output tokens per grid (mostly thinking), about $0.20 for the run at $0.10/M input and $0.50/M output; 1.8 min per grid.
- **Sonnet 5.5 smoke test on `grid_20_large`'s three 13x13 grids** (instances 16–18, those with the longest shortest paths: 30, 38 and 26 steps; `out/results/smoke/sonnet5_5_large13`, not committed): all three solved, in rounds 1, 2 and 2, with 24–35 rules over 374–1,734 situations. PRISM took 1–3 s per grid, so verification is cheap at this size. About $0.30.
- **Ceilings of `grid_20_large`** (`out/results/ceilings/gridworld_grid_20_large.md`): all 20 grids are jointly solvable, and every requirement's optimum is 1.0. The joint queries (LP on the bare MDP) took 29 minutes for the 20 grids, mostly on the larger ones; the loop's per-round checks on the induced MDP stay at seconds.
- **Reading:** sizes up to 13 leave Sonnet 5.5 no headroom. Headroom has to come from a weaker model (Haiku 5.5 has some on `grid_20_balanced`), much larger grids, or harder instances (more goals).

## Haiku ablation on grid_20_large (planned Oct 8, written before the runs)
- **Conditions** (Asher): Claude Haiku 5.5 with `haiku5_5`'s settings on `grid_20_large`, two unseeded runs each (`run_ablation.py ... --unseeded 2`): restart after 2 stalls (`haiku_large_B2`, the ablations' reference), restart on slow progress (`haiku_large_R4`, the default retry) and pure resampling with **10 rounds** (`haiku_large_R5`), so that it can be compared both at equal iterations and at equal cost. Rules over absolute coordinates (the current vocabulary). Legacy is not part of it (qwen, `grid_20_balanced`, running separately).
- **Planned comparisons, fixed before the runs.** Metric: requirements met by the kept policy (worst case, the loop's values, replayed per round), per grid averaged over the two runs; paired sign-flip test over the 20 grids; **Holm correction over these four**:
  1. restart after 2 stalls vs pure resampling at 5 rounds (equal iterations);
  2. restart on slow progress vs pure resampling at 5 rounds;
  3. restart after 2 stalls vs pure resampling at equal cost: on each grid, pure resampling gets the feedback condition's mean spend on that grid, cost at Haiku 5.5's prices ($0.10/M input, $0.50/M output tokens); every run keeps at least its first round, whose prompt is the same in every condition;
  4. restart on slow progress vs pure resampling at equal cost.
- Everything else (solved counts, rounds to solve, the two feedback conditions against each other) is exploratory.
- **Results (Oct 8): feedback beats pure resampling, in all four planned comparisons** (requirements met per grid, Holm-corrected over the four):

  | comparison | Δ met [95% CI] | p (Holm) |
  |---|---|---|
  | restart after 2 stalls vs pure resampling, 5 rounds | +1.68 [+1.00, +2.40] | < 0.001 |
  | restart on slow progress vs pure resampling, 5 rounds | +1.43 [+0.80, +2.10] | 0.001 |
  | restart after 2 stalls vs pure resampling, equal cost | +1.43 [+0.88, +2.00] | 0.001 |
  | restart on slow progress vs pure resampling, equal cost | +1.27 [+0.62, +1.98] | 0.002 |

  - Final (full budget): restart after 2 stalls 8.43 met, 14/20 solved; restart on slow progress 8.18, 12/20; pure resampling with 10 rounds 7.30, 5.5/20.
  - Round 1 is the same prompt in every condition: 4.78–5.22, so about ±0.2 is noise. The feedback conditions jump in round 2 (7.10–7.30) while resampling reaches 5.97; resampling's 10th round (7.30) does not reach the feedback conditions' 2nd.
  - Cost per grid: $0.015–0.016 (feedback), $0.021 (resampling, 10 rounds); about $1 for the six runs.
  - **Reading:** with qwen on small grids, resampling was ahead; with a stronger model on harder grids, quantitative feedback helps clearly, at equal rounds and at equal cost. Exploratory: restart after 2 stalls vs restart on slow progress (8.43 vs 8.18) is not a planned comparison.
- Page: `out/results/haiku_large/SUMMARY.md` (`configs/plot/haiku_large.yaml`). The plan first called restart after 2 stalls "the default loop"; the default retry is restart on slow progress, so the names above were corrected after the commit that fixed the plan (the comparisons are unchanged).
