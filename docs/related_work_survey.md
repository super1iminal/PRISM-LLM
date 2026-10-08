# Related-work survey, phase 1: probabilistic generate-then-verify refinement loops

Compiled 2026-10-08. This is phase 1 of 2: it finds, reads and characterises existing work. It makes no comparison with this project's approach and no novelty argument; phase 2 does that.

## How to read this

**Tiers** follow the brief:
- **High**: main track of ICML, NeurIPS, ICLR, ICAPS, AAAI, IJCAI, CAV, TACAS, ICSE or FSE.
- **Medium**: any other peer-reviewed conference or journal. Some of these are top venues in their own field (JAIR, POPL, PLDI, ASE, CHI, UAI, CoRL, Nature); entries say "outside the list" for those.
- **Low–Medium**: a workshop held at one of the listed venues. The brief ranks "workshops outside top venues" Low, so I read workshops at listed venues as one step higher.
- **Low**: other workshops, arXiv-only preprints, and papers with few results.

I did not judge any paper to be apparently AI-generated. Papers read only as abstracts were not assessed for this.

**Reading depth**:
- **Full**: the whole main text; appendices only where stated.
- **Most**: method, setup, results and limitations read; the parts named in the entry were skimmed.
- **Partial**: some of those parts were not read; the entry says which and why.
- **Abstract**: the abstract page only.
- **Snippet**: search-result text only.

Abstract- and snippet-level papers sit in short tables at the end of each cluster. Their columns say only what the abstract states.

**Labels** such as B.2 refer to entries in this survey, not to the project's ablation conditions.

**Flags** in each full entry mark what the brief asked to separate:
- a guarantee that is only empirical;
- a loop that is described but run once;
- "probabilistic" in name only;
- a guarantee that holds only for a learned model;
- abstract claims that the results do not support;
- a single run or a single model;
- no comparison against resampling at equal budget.

## Contents

1. [Target problem](#1-target-problem)
2. [Per-paper entries](#2-per-paper-entries)
   - [A. LLM output checked by probabilistic model checking](#a-llm-output-checked-by-probabilistic-model-checking)
   - [B. LLM-written generalized policies and programs for planning](#b-llm-written-generalized-policies-and-programs-for-planning)
   - [C. Verifier feedback versus resampling; LLM-Modulo](#c-verifier-feedback-versus-resampling-llm-modulo)
   - [D. LLMs with non-probabilistic formal verification; LLM-written models](#d-llms-with-non-probabilistic-formal-verification-llm-written-models)
   - [E. Statistical and conformal guarantees for LLM planners](#e-statistical-and-conformal-guarantees-for-llm-planners)
   - [F. Families of MDPs, multi-environment and robust policies (no LLM)](#f-families-of-mdps-multi-environment-and-robust-policies-no-llm)
   - [G. Synthesis and learning with probabilistic model checking in the loop (no LLM)](#g-synthesis-and-learning-with-probabilistic-model-checking-in-the-loop-no-llm)
   - [H. Permissive strategies, compact strategy representations, shields](#h-permissive-strategies-compact-strategy-representations-shields)
   - [I. Generalising policies to held-out and larger instances](#i-generalising-policies-to-held-out-and-larger-instances)
   - [J. Self-adaptive systems (SEAMS)](#j-self-adaptive-systems-seams)
   - [K. Reusing solved instances or experience in context](#k-reusing-solved-instances-or-experience-in-context)
   - [L. LLM-guided controller synthesis in control](#l-llm-guided-controller-synthesis-in-control)
   - [M. Domain resources and the project's own prior work](#m-domain-resources-and-the-projects-own-prior-work)
3. [Coverage map](#3-coverage-map)
4. [Limitations across approaches](#4-limitations-across-approaches)
5. [Gaps](#5-gaps)
6. [Roadblocks](#6-roadblocks)
7. [Search log](#7-search-log)

## 1. Target problem

The target is a generate-then-verify loop for **one Markov decision process given as a PRISM model**.

- **What is generated.** An LLM writes a symbolic controller that may be partial: an ordered list of `condition -> action` rules over declared policy-visible variables, where the first matching rule wins. In states that no rule covers, every policy action stays allowed.
- **What verifies it.** PRISM. Each requirement is a threshold on a probability (a PCTL or LTL path formula) or on an expected reward. PRISM computes the best and the worst value over every completion of the rules (Pmax and Pmin over all schedulers of the restricted MDP). It also runs a joint multi-objective feasibility query, and it re-checks the final values with interval iteration.
- **What kind of guarantee.** Exact model checking on that finite model, one instance at a time; nothing is statistical. If every worst case meets its threshold, every deployment consistent with the rules meets every requirement, so the rule set is a sound permissive strategy.
- **What goes back to the generator.**
  - A table of best value, worst value and threshold per requirement, plus how many situations the rules cover.
  - A branch. **REFINE** (rewrite the list) when no completion can pass. **EXTEND** (append rules) otherwise; appending can only shrink the set of completions, so worst cases cannot fall.
  - Blame: rules and uncovered situations ranked by expected visits times the probability at stake. PRISM's optimal action is never revealed.
  - Around the loop: the best rule set so far is kept, and the loop restarts when progress stalls.
- **Planned extension (families).** Rules over relative features are written on a few instances, then frozen. They are then certified one instance at a time on held-out and larger instances.

The inputs (meeting notes of Sept 24 and Oct 3, Marsha's email, the meeting transcript) agree on this target. Rule transfer, dropped in the Sept 24 notes, is part of the family extension that the later email proposes.

## 2. Per-paper entries

### A. LLM output checked by probabilistic model checking

Papers in which an LLM produces a policy, plan, model or action, and PRISM or Storm checks it or a model learned around it.

#### A.1 Verifying Memoryless Sequential Decision-making of Large Language Models

- **Citation**: D. Gross, H. Spieker, A. Gotlieb. arXiv:2510.06756, Oct 2025.
- **Link**: https://arxiv.org/abs/2510.06756 (code: COOL-MC repository, `llm_verification` branch)
- **Venue / tier**: preprint, no venue found. **Low**.
- **Depth**: Full (arXiv HTML).
- **Generated**: nothing is synthesised. The LLM is the policy: each reachable state is described in natural language, and the reply is parsed into an action (a default action if parsing fails).
- **Verifier**: probabilistic and exact. COOL-MC builds, state by state, the Markov chain that the LLM policy induces on a PRISM MDP; Storm checks PCTL reachability (e.g., the probability of reaching an unsafe event).
- **Loop**: none. One verification, no feedback to the LLM.
- **Guarantee**: the exact property value of one seeded, deterministic run of the LLM policy on one model. The LLM's sampling behaviour is not covered, because Ollama exposes no token probabilities.
- **Evaluation**: three small COOL-MC benchmarks (Taxi, 4×4 Frozen Lake, a stock-market model; at most 691 reachable states). Five open models of 1.5B–7B parameters via Ollama, seed 42, against DQN baselines. The LLMs mostly do worse than DQN. LLM inference dominates runtime: about 4 h for 17 states in one case, and one run timed out at 5 h.
- **Claimed vs demonstrated**: the paper presents an automated model-checking framework for LLM policies with exact safety guarantees. The exactness holds only for the seeded run on these small models. Richer PCTL properties are said to be supported, but only reachability is exercised.
- **Limitations**:
  - Authors: memoryless, deterministic policies only; scale limited by per-state inference and model size.
  - Observed: one LLM call per reachable state; the verdict depends on seed and prompt; no repair.
- **Flags**: guarantee scoped to one seeded run; single run per configuration; no loop.

#### A.2 Enhancing RL Safety with Counterfactual LLM Reasoning

- **Citation**: D. Gross, H. Spieker. ICTSS 2024, LNCS 15383, pp. 23–29 (short paper). arXiv:2409.10188.
- **Link**: https://arxiv.org/abs/2409.10188
- **Venue / tier**: peer-reviewed testing conference; a 7-page short paper. **Medium**.
- **Depth**: Full.
- **Generated**: a trained DQN policy has state-action pairs that lead directly into a safety violation. For each of them, GPT-4-turbo (greedy decoding) proposes an alternative action with an explanation.
- **Verifier**: probabilistic and exact. COOL-MC and Storm check the Markov chain the policy induces on a PRISM MDP, using PCTL reachability of unsafe events.
- **Loop**: one round. Verify; collect the pairs one step before a violation; the LLM proposes alternatives; override them; verify once more.
- **Guarantee**: the exact value of the repaired policy (the DQN plus the overrides) on the model. No threshold is claimed to be met.
- **Evaluation**:
  - Setup: one private "cleaning agent" environment, built to avoid training-data contamination; four PCTL queries. The baseline replaces a violating action with the DQN's second-best action.
  - With a natural-language description of the environment, the probability of running out of energy drops from 0.603 to 0.406 (baseline 0.660). The other three queries tie the baseline.
  - Describing the environment as PRISM code does worse; a hand-tailored description reaches 0.265.
  - The authors judged three quarters of 44 explanations sensible.
- **Claimed vs demonstrated**: the claim that LLMs can explain and improve RL safety rests on one toy environment, one model, one run and one repair round. Results vary strongly with the description.
- **Limitations**:
  - Authors: repairs only the step before a violation; memoryless policies; state-space size.
  - Observed: no non-LLM repair baseline, such as the model checker's own optimal action.
- **Flags**: loop described, run once; single run and model.

#### A.3 VeriPlan: Integrating Formal Verification and LLMs into End-User Planning

- **Citation**: Lee, Porfirio, Wang, Zhao, Mutlu. CHI 2025, article 247. arXiv:2502.17898.
- **Link**: https://arxiv.org/abs/2502.17898
- **Venue / tier**: CHI main track (a top HCI venue, outside the list). **Medium**.
- **Depth**: Most (method, study design, results and limitations read; qualitative themes skimmed).
- **Generated**:
  - GPT-4 writes an everyday plan: a schedule, a patient escort or a cooking task.
  - LLM agents fit the user's constraints into six LTL templates.
  - An LLM translates the constraints and the plan into PRISM.
  - The constraints are translated back into natural language for the user to confirm.
- **Verifier**: PRISM and Stormpy, used qualitatively: any violated rule rejects the plan. User-set "flexibility" sliders make rules soft: before each check, whether each rule is enforced is sampled according to its slider. That sampling is the only probabilistic element.
- **Loop**: generate, check, return the violated rules to the LLM (and show them to the user), regenerate. Fixed at three iterations; users may edit rules between runs.
- **Guarantee**: the LLM-translated plan satisfies the LLM-translated rules. Both translations are only reviewed by the user.
- **Evaluation**: within-subjects user study, n = 12, three scenarios. Ablations remove the sliders, the translator, or both. Measures are Likert scales (USE, FATE). The full system is rated significantly higher on perceived performance, usefulness and satisfaction. No objective rate of valid plans is reported.
- **Claimed vs demonstrated**: improved reliability is supported by perceived ratings only; the paper does not count how often plans became valid.
- **Limitations**:
  - Authors: limited constraint templates; PRISM/Stormpy LTL support; three scenarios; n = 12.
  - Observed: checking one linear plan amounts to trace checking; translation correctness is not measured.
- **Flags**: "probabilistic" in name only (a probabilistic checker used qualitatively); loop capped at three rounds.

#### A.4 Automated Generation of MDPs Using Logic Programming and LLMs for Robotic Applications

- **Citation**: Saccon, De Martini, Saveriano, Lamon, Palopoli, Roveri. IEEE Robotics and Automation Letters (RA-L), 2025. arXiv:2511.23143.
- **Link**: https://arxiv.org/abs/2511.23143
- **Venue / tier**: journal. **Medium**.
- **Depth**: Most (introduction, framework, experiments and results read; graph-construction details and discussion skimmed).
- **Generated**:
  - The LLM (GPT-4o, later GPT-5-mini; temperature 0, seed 42, few-shot) writes a Prolog knowledge base from a natural-language scenario: initial state, predicates, PPDDL-like actions with probabilistic effects, reward predicates and goal labels.
  - An MDP in PRISM is then built deterministically by reachability from the knowledge base.
  - Storm synthesises the policy (maximum goal probability or minimum expected cost). It is exported as a state-action table and run in ROS 2.
- **Verifier**: Storm, as the synthesiser of an optimal policy for the generated MDP, not as a checker of anything the LLM wrote. The Prolog interpreter catches syntax and instantiation errors; experts judge each knowledge base.
- **Loop**: none automatic. Experts correct errors, and nothing from model checking goes back to the LLM.
- **Guarantee**: the policy is optimal for the generated MDP. Whether that MDP matches the scenario is checked only by expert inspection.
- **Evaluation**:
  - Three use cases (human–robot assembly, including a real robot; AGV traffic; a probabilistic gripper), with five variants each (sizes, probabilities, new actions); MDPs of 17–1024 states.
  - GPT-4o wrote fully correct knowledge bases in 5/5 assembly variants, 4/5 AGV variants (one minor error) and 3/5 gripper variants (syntax slips).
  - GPT-5-mini, tried afterwards, reached a perfect score in the additional tests reported.
  - The synthesised policies are simulated against faulty and ε-greedy baselines.
- **Claimed vs demonstrated**: the claim that LLMs plus formal methods make probabilistic planning more accessible rests on small, hand-picked scenarios with expert-validated knowledge bases.
- **Limitations**: observed: small models; no automatic check of the model against the description; one generation per test at temperature 0.
- **Flags**: no loop; model faithfulness judged by experts only.

#### A.5 Pro2Guard: Proactive Runtime Enforcement for LLM Agent Safety via Probabilistic Prediction

- **Citation**: Wang, Poskitt, Sun, Wei. arXiv:2508.00500 (v2, Jan 2026; v1 was titled "... via Probabilistic Model Checking").
- **Link**: https://arxiv.org/abs/2508.00500
- **Venue / tier**: preprint, no venue found. **Low**.
- **Depth**: Full main text (v2); the proof appendix was not read.
- **Generated**: nothing offline. At runtime an LLM agent (a ReAct-style household agent, or the Apollo driving stack) proposes actions. When the predicted risk passes a threshold, the system either re-prompts the LLM with a risk alert (state, violated property, probability) so it revises its plan, or stops the agent and hands over to a human.
- **Verifier**: probabilistic, but on a learned model. A Markov chain is learned from execution traces over hand-chosen predicates (Laplace smoothing). PRISM computes the probability of eventually reaching unsafe abstract states; traffic laws with bounded response use a monitor product. Values are cached per abstract state.
- **Loop**: monitoring at every step, with one reflection prompt per trigger.
- **Guarantee**: an (ε, δ) PAC bound on reachability probabilities in the learned chain, using a sample-size rule from prior work. It holds only if the real process is Markov over the chosen abstraction. Nothing is guaranteed about behaviour after an intervention.
- **Evaluation**:
  - Apollo: 7 law-violation scenarios, measuring warning lead time against REDriver.
  - SafeAgentBench household tasks (unsafe % / completion %): no enforcement 40.63/59.38; AgentSpec 19.79/59.38; stop at θ = 0.1, 2.60/10.42; reflect at θ = 0.1, 14.07/47.74.
  - 12% fewer tokens than AgentSpec; 5–30 ms overhead with caching.
- **Claimed vs demonstrated**: the abstract's headline enforcement and completion rates (93.6% and 80.4%) do not appear in the v2 results table, and the text gives a third unsafe rate (11.47%). The PAC claim rests on the Markov-abstraction assumption. Scenario counts are small.
- **Limitations**: observed: what gets checked is a learned behaviour model, not an artifact the LLM wrote; the predicates are chosen by hand.
- **Flags**: guarantee relative to a learned model; abstract claims do not match the results table.

#### A.6 Abstract-level entries

| Paper | Venue / tier | Depth | What the abstract says |
|---|---|---|---|
| Koohestani. AgentGuard: Runtime Verification of AI Agents. arXiv:2509.23864 | AgenticSE workshop @ ASE 2025 · Low | Abstract | Watches an agent's inputs and outputs, learns an MDP online and runs probabilistic model checking at runtime. No properties or results are given in the abstract. |
| Je-Gal, Yi, Lee. A-LAMP: Agentic LLM-Based Framework for Automated MDP Modeling and Policy Generation. arXiv:2512.11270 | NeurIPS 2025 workshop · Low–Medium | Abstract | Natural-language task → MDP formulation → environment code → RL-trained policy, with checkable stages. No formal verification. |
| Gross, Spieker, Gotlieb. Bounded PCTL Model Checking of LLM Outputs (LLMCHECKER). arXiv:2509.18836 | ICTAI 2025 · Medium | Abstract | Checks PCTL properties of the LLM's own token generation, restricted to bounded top-k tokens. Verifies the model, not an artifact it writes. |
| Spieker, Gross, Gotlieb. Probabilistic Model Checking of Autoregressive Neural Sequence Models. arXiv:2609.00838 | ICTSS 2026 · Medium | Abstract | Under-approximates generation by a Markov chain, checks it with PRISM, and narrows certified intervals with a CEGAR loop; examples are a GPT-2 process-planning model and a molecule generator. Verifies the model. |
| Tran-Truong, Le. Measuring the Unmeasurable: Markov Chain Reliability for LLM Agents (TraceToChain). arXiv:2604.24579 | Preprint · Low | Abstract | Fits agent execution traces to an absorbing Markov chain (Laplace-smoothed estimates, goodness-of-fit checks, credible and bootstrap intervals) and derives pass@k-style reliability metrics from it; seven agent frameworks. The abstract names no model checker. |
| Dhodapkar, Pishori. SafetyDrift: Predicting When AI Agents Cross the Line Before They Actually Do. arXiv:2603.27148 | Preprint (submitted to COLM) · Low | Abstract | Models agent safety trajectories as absorbing Markov chains and computes the probability of a violation within a given number of steps in closed form; 357 traces from 40 tasks; a monitor built on it detects 94.7% of violations 3.7 steps early. |
| Dantas, Cordeiro, Nowroozi, Tihanyi. Toward Safe LLM Agents: A Survey of Specification, Verification, and Enforcement. arXiv:2608.14590 | Preprint (survey) · Low | Abstract | Systematic review of 38 studies (2022–2026). Translating natural language into formal specifications reaches only 24–35% semantic correctness; runtime monitoring is the most mature enforcement; blocking most unsafe actions can still leave few tasks completed safely; no approach is sound, scalable, semantically correct and task-safe at once. |

### B. LLM-written generalized policies and programs for planning

An LLM writes one program, policy or heuristic for a whole planning domain. Every full entry here uses deterministic, fully observable PDDL domains and goal reachability.

#### B.1 Generalized Planning in PDDL Domains with Pretrained Large Language Models

- **Citation**: T. Silver, S. Dan, K. Srinivas, J. B. Tenenbaum, L. P. Kaelbling, M. Katz. AAAI 2024, 38(18):20256–20264. arXiv:2305.11014.
- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/30006
- **Venue / tier**: AAAI main track. **High**.
- **Depth**: Full, including Appendix A; the program listings in Appendix B were skimmed.
- **Generated**: a domain-specific Python function `get_plan(objects, init, goal)`, a generalized plan, written by GPT-4 through the ChatGPT interface (copied by hand). The prompt first asks for a domain summary and a simple strategy without search. Two training tasks are shown, truncated.
- **Verifier**: deterministic. The program runs on the training tasks (exceptions, 30 s timeout), plans are syntax-checked, and VAL validates them and suggests repairs.
- **Loop**: up to four debugging re-prompts. Each carries one feedback type: a traceback, a timeout with traceback, a syntax error with an operator reminder, or a VAL semantic hint. History accumulates; if the program still fails after four rounds, the last version is used. No restarts.
- **Guarantee**: none formal. Programs are valid on the training tasks; generalisation is measured on 30 larger held-out tasks per seed.
- **Evaluation**:
  - Setup: 7 domains (6 from the PG3 benchmark plus a new "Heavy"), 10 seeds.
  - Ablations: no chain of thought, no debugging, anonymised names, GPT-3.5. Baselines: PG3, policy evaluation, plan comparison, random.
  - GPT-4 does well on Delivery 0.90, Forest 1.0, Gripper 0.90, Ferry 0.80 and Heavy 0.60 (PG3 0.0), and fails on Miconic 0.01 and Spanner 0.10. PG3 scores 1.0 on 6 of 7.
  - Without debugging results are much worse (Delivery 0.10); with anonymised names they collapse. One debugging step gives most of the gain.
  - Outcomes are nearly all-or-nothing per seed. The programs run faster than LAMA.
- **Claimed vs demonstrated**: the importance of automated debugging is shown against no debugging with a single sample, not against restarting or resampling at equal budget. The authors mention restarts and pass@k as untested options. Generalisation is not proven.
- **Limitations**:
  - Authors: the domains are easy to hand-code; training tasks may not convey the task distribution; results depend on names; errors in the overall strategy need a restart rather than local fixes.
  - Observed: deterministic, fully observed domains; IPC-style domains carry contamination risk (acknowledged); one LLM used through a chat interface.
- **Flags**: generalisation empirical only; no equal-budget resampling baseline.

#### B.2 Improved Generalized Planning with LLMs through Strategy Refinement and Reflection

- **Citation**: K. Stein, D. Hodel, D. Fišer, J. Hoffmann, M. Katz, A. Koller. ICAPS 2026, 36(1):677–686. arXiv:2508.13876.
- **Link**: https://arxiv.org/abs/2508.13876 (proceedings: https://ojs.aaai.org/index.php/ICAPS/article/view/42886)
- **Venue / tier**: ICAPS main track. **High**.
- **Depth**: Most (introduction, method, experiments and results including Tables 1–2 read; the costumed/anonymised results table partly read; appendices not read).
- **Generated**: natural-language descriptions of the domain and tasks, then a pseudocode strategy, then a Python generalized plan. Models: GPT-4o, Llama-3.3-70B, DeepSeek-V3.2, Qwen3-30B-Thinking; greedy decoding.
- **Verifier**: deterministic. VAL checks plans for 6 small debugging tasks. The pseudocode is checked indirectly: the LLM follows it to produce plans, and VAL checks those plans. Programs run with a 45 s limit.
- **Loop**:
  - Strategy debugging with reflection (the LLM locates the faulty pseudocode step and gives a reason): up to 5 rounds, keeping the strategy with the best debugging coverage.
  - Code debugging with reflection, using both positive and negative examples.
  - Several initial programs from reordered prompts.
  - Two configurations with about the same maximum number of programs are compared: 3 initial programs × 6 debugging steps, and 5 × 3. The best program is chosen on the debugging tasks.
- **Guarantee**: none formal. Coverage is measured on larger evaluation tasks under 4 input orderings. A manual analysis argues that the programs with 100% coverage solve every instance the generator can produce.
- **Evaluation**:
  - Setup: 17 domains (Silver et al.'s 7 plus 10 more), 3 runs, with "costumed" and anonymised variants.
  - The best configuration (5 × 3 with DeepSeek) averages 82% coverage.
  - Removing multiple programs, strategy debugging or code reflection each hurts some domains.
  - Neither 3 × 6 nor 5 × 3 wins consistently; the authors attribute this to the kind of mistake involved.
  - About 15M tokens (about 5.5 USD) for DeepSeek.
- **Claimed vs demonstrated**: the gains over Silver et al. hold across four LLMs. Generalisation beyond the test set rests on manual analysis. Three runs per configuration.
- **Limitations**: observed: deterministic domains; verification uses only six small tasks.
- **Flags**: guarantee empirical only.

#### B.3 Provably Complete Generalized Planning with LLMs

- **Citation**: Stein, Jain, Hoffmann, Koller. arXiv:2609.27105, Sep 2026.
- **Link**: https://arxiv.org/abs/2609.27105
- **Venue / tier**: preprint. **Low**.
- **Depth**: Most (introduction, pipeline, anti-cheating checks, invariants, experiments, related work and conclusion read; the PDDL-to-Lean formalisation skimmed).
- **Generated**: GPT-5.6-Sol (temperature 1, high reasoning) writes three things:
  - a Lean function `solve(instance)` implementing a given pseudocode strategy (taken from Stein et al. 2026 and checked or revised by the authors);
  - a Lean proof that, for every instance satisfying user-written domain constraints, `solve` returns an applicable plan that reaches the goal;
  - proofs that actions preserve the state-validity invariants.
- **Verifier**: the Lean kernel, which checks the proof deductively for all valid instances of any size. VAL also checks 6 small debugging tasks during the plan stage. Anti-cheating checks: no custom notation or macros, the exact theorem statement, standard axioms only, and an independent re-check.
- **Loop**:
  - Plan debugging on Lean errors and VAL results: up to 4 rounds.
  - Proof debugging, in one of two modes. Basic mode regenerates the whole proof, up to 6 rounds. Iterative mode writes a proof sketch and repairs it declaration by declaration, stopping after 6 consecutive failures on one declaration.
- **Guarantee**: machine-checked completeness, relative to the human-written constraint specification and the deterministic PDDL-to-Lean translation.
- **Evaluation**: 13 IPC-style domains. Plans for all 13 (10 on the first try). Proofs for 12: 7 in basic mode, 5 more in iterative mode; Transport fails. 9–54 minutes per domain, with up to 66 LLM-written declarations.
- **Claimed vs demonstrated**: the claims hold for the stated constraint specifications. The strategies come from prior work, so the loop proves strategies rather than discovering them. One model; constraint specifications are written by hand.
- **Limitations**: observed: deterministic domains and goal reachability only; correctness is relative to the specification and the translation.
- **Flags**: single model; guarantee relative to hand-written specifications.

#### B.4 Prompt, Prove, Patch: The Neuro-Symbolic Loop for General Policy Synthesis

- **Citation**: Drexler, Fritzsche, Musayev, Ståhlberg. ICAPS 2026 Workshop on Reliability in Planning and Learning (RIPL).
- **Link**: https://icaps26.icaps-conference.org/files/workshops/ripl/DrexlerRIPL26.pdf
- **Venue / tier**: workshop at a listed venue. **Low–Medium**.
- **Depth**: Full main text and Appendix A (the prompt); Appendix B (policy listings) skimmed.
- **Generated**: a general policy for a classical-planning domain. It consists of description-logic features (a concept and role grammar with role composition and transitive closure) and rules. Each rule maps Boolean or numeric feature conditions to feature-change effects; the policy is a relation over transitions, in the style of Bonet and Geffner. GPT-5.5 (medium thinking) writes it inside Codex, as one continuous agent session per domain that writes files and calls tools.
- **Verifier**: deterministic and exhaustive.
  - The proof tool grounds each training task, builds the graph of states reachable under the policy, and proves that every maximal path is finite and ends in a goal; when several transitions are accepted, the choice is adversarial.
  - Counterexamples are open states (reachable non-goal states with no accepted transition) or cycles. There is no dead-end analysis.
  - Helper tools: an enumerator of simple features that separate conflicting states (with a value table), greedy execution on validation tasks, and a GBFS+hFF solvability screen.
- **Loop**:
  - Start from an empty policy. Propose, prove on all training tasks, receive a counterexample (and optionally the feature table), revise. Stop when the training proof passes and validation execution succeeds.
  - The full history stays in context. Iterations per domain range from 1 to 329 (Floortile).
  - The LLM mostly ignored the enumerated features and wrote its own, more complex ones.
- **Guarantee**: a proof over the finite training tasks, covering all behaviour the policy accepts. Generalisation to larger test tasks is empirical (greedy execution). The authors name proofs over parameterised families as future work.
- **Evaluation**:
  - Setup: 15 classical domains from IPC generators; 100 training, 100 validation and 100 test tasks per domain (Gripper 5/5/10). Test tasks are strictly larger along at least one scaling dimension. IPC tasks are also used.
  - Proved on the training tasks in 14 of 15 domains (Floortile 97/100).
  - Test: greedy execution solves 1370 of 1410 generated tasks, against LAMA 1207 and Dual-BFWS 1170 (planners given as reference points). On IPC tasks: 440/449, against 404 and 394.
  - Driverlog fails because the feature language cannot bind each driver to its own target (60/100 test). Satellite is slow because features are expensive to evaluate.
- **Claimed vs demonstrated**:
  - "Formally proven" applies to the training tasks only.
  - One run per domain, one model, no seeds.
  - No ablation of the loop (no one-shot or no-counterexample baseline), and no comparison with earlier general-policy learners on the same splits.
  - The LLM helped choose the task-parameter ranges, which were then curated by hand. No token or cost figures.
- **Limitations**:
  - Authors: Driverlog expressivity; no solvability analysis, so dead ends surface late; generalisation evidence comes from validation only; PDDL names may give cues (they suggest obfuscation).
  - Observed: deterministic domains and goal reachability only; the model's prior knowledge of well-known IPC domains is not separated from the loop's effect.
- **Flags**: proof limited to training tasks, generalisation empirical; single run and model; no loop ablation.

#### B.5 GenePlan: Evolving Better Generalized PDDL Plans using Large Language Models

- **Citation**: A. Murray, D. Dervovic, A. Pozanco, M. Cashmore. ICAPS 2026 (the arXiv comment says accepted to ICAPS 2026, and the paper has an ICAPS proceedings page). arXiv:2603.09481.
- **Link**: https://arxiv.org/abs/2603.09481 (proceedings: https://ojs.aaai.org/index.php/ICAPS/article/view/42885)
- **Venue / tier**: ICAPS. **High**.
- **Depth**: Full main text; appendices (domain PDDL, evolved planners, Sokoban) skimmed by heading.
- **Generated**: Python generalized planners `get_plan(objects, init, goal)` written by GPT-4o (and GPT-4o-mini) and evolved in the style of FunSearch. The initial population comes from chain-of-thought prompting (Silver et al.) or from a given planner.
- **Verifier**: deterministic. VAL validates every plan on 5–10 training tasks. Fitness is the mean plan length, with a large failure score for an unsolved task. An abstract-syntax-tree whitelist screens code before it runs.
- **Loop**:
  - Evolutionary search: a population of 10, 10 offspring per generation, 10 generations.
  - Parents are sampled by a softmax over fitness, with a temperature that decays as the population grows.
  - The prompt shows the sampled parents, each with its score, error message or VAL failure, and asks for crossover and mutation.
  - The 10 best of parents and offspring survive each generation, and the best planner is returned.
- **Guarantee**: none formal. Plan quality is measured on 30 test problems per domain.
- **Evaluation**:
  - Domains: 8, six from prior work and two new ones ("research", "trading") meant to limit contamination.
  - Baselines: chain-of-thought with GPT-4 and GPT-4o (Silver et al.); Fast Downward LAMA at 300 s and 1800 s; optimal A* with LM-cut (30 min).
  - Variants: a natural-language domain summary, GPT-4o-mini, anonymised names, and a no-evaluator variant in which every candidate gets the same score.
  - Average IPC quality score: 0.91, against 0.93 for LAMA at 1800 s and 0.64 for chain-of-thought with GPT-4o. The evolved planners solve 100% of test tasks in all 8 domains.
  - Anonymised names give 0% everywhere. The no-evaluator variant performs close to chain-of-thought (e.g., 16.7% solved on Miconic) at a similar dollar cost per domain.
  - Friedman and Nemenyi tests: GenePlan and LAMA at 1800 s are statistically indistinguishable; GenePlan beats chain-of-thought.
  - Cost: about $1.82 per domain with GPT-4o, about 645 s to generate, 0.49 s per test task.
  - Sokoban, which has no simple strategy: no tasks solved.
- **Claimed vs demonstrated**: parity in plan quality with LAMA at 1800 s holds on these 8 domains, from one GenePlan run per domain (no repeated seeds reported). Test-task sizes relative to training are described only as varying.
- **Limitations**:
  - Authors: domains without simple strategies; no early stopping; plan length is the only metric.
  - Observed: deterministic domains; generalisation not proven.
- **Flags**: guarantee empirical only; single run per domain.

#### B.6 Language Models for Generalised PDDL Planning: Synthesising Sound and Programmatic Policies (LMPlan)

- **Citation**: D. Z. Chen, J. Zenn, T. Cinquin, S. A. McIlraith. RLC 2025 Workshop on Programmatic RL. arXiv:2508.18507.
- **Link**: https://arxiv.org/abs/2508.18507
- **Venue / tier**: workshop at a venue outside the list. **Low**.
- **Depth**: Full main text; appendices (costs, generation times, correlation analysis) not read.
- **Generated**: Python programs, once per domain: a value function used as a heuristic for greedy best-first search, and a policy that picks among the applicable actions. Models: DeepSeek-R1, Gemini 2.0 Flash, Gemini 2.5 Flash. The prompt includes the domain, 2 training problems and an example program for Gripper.
- **Verifier**: none external. Soundness comes from construction:
  - the policy wrapper executes only actions applicable in the current state (a random applicable action if the program returns anything else), and returns a plan only once the goal is reached;
  - search with a heuristic that never wrongly reports infinity is sound and, on finite state spaces, complete.
- **Loop**: none; no feedback. Each LM is called 10 times per domain, and the best value function and policy are selected by a time-based score on 10 small training problems.
- **Guarantee**: every returned plan is valid. The policy has no termination or goal-reaching guarantee (the authors say so).
- **Evaluation**:
  - Setup: IPC 2023 learning track, 10 domains with 90 test problems each (900); test problems are up to 100× larger than the validation ones; 1800 s and 8 GB per problem.
  - Baselines: GBFS with hFF, LAMA, WL-GOOSE, and the reported numbers of Corrêa et al.'s LLM-generated heuristics.
  - Policy rollout solves 563, against 557 for LAMA. It solves every problem in 6 domains and does poorly on Childsnack, Floortile and Rovers.
  - Search with the value function solves 397, the combined two-queue search 424, and the portfolio chosen by validation 630.
  - Anonymised names have mixed effects: better on some domains, worse on others.
- **Claimed vs demonstrated**:
  - "Provably sound without external verifiers" describes the execution wrapper, which is sound for any program; it says nothing about reaching the goal.
  - Coverage comes from one selection run.
  - Corrêa et al.'s numbers are copied from their paper and were measured on different hardware.
- **Limitations**:
  - Authors: policies have no completeness or termination guarantee; plan quality is sometimes far worse (up to about 100× on Transport); some domains fail; name-anonymisation effects are unexplained.
  - Observed: deterministic domains; no repair loop.
- **Flags**: "sound" refers to execution only; guarantee empirical only; no loop.

#### B.7 Property-Guided LLM Program Synthesis for Planning

- **Citation**: A. G. Pereira, A. B. Corrêa, J. Seipp. arXiv:2605.16142, May 2026.
- **Link**: https://arxiv.org/abs/2605.16142
- **Venue / tier**: preprint. **Low**.
- **Depth**: Most (main text, Sections 1–7, read; appendices not read).
- **Generated**: Python heuristic functions for PDDL domains, written by Gemini 3.1 Pro.
- **Verifier**: a deterministic property check on the training tasks.
  - A heuristic is "direct" if every state reachable from the initial state through strictly improving transitions has a strictly improving successor. Hill climbing then reaches the goal without search.
  - A depth-first search per training task stops at the first violation, with a 30 s cap per task; a timeout counts as a pass.
  - A counterexample is the violating state, its heuristic value and the values of its successors.
- **Loop**: a repair prompt with the counterexample and the full history; at most 10 iterations; 3 runs.
- **Guarantee**: directness proven on the training tasks within the time cap only; behaviour on test tasks is empirical.
- **Evaluation**:
  - Setup: IPC 2023 learning track, 10 domains, 900 test tasks outside the training distribution.
  - On average 3.4 candidates per run, against a fixed 25 in the sample-and-select method of Corrêa et al. (NeurIPS 2025). Evaluating candidates costs 10.75 minutes per domain, against 206 CPU-hours.
  - Direct heuristics with hill climbing solve 623.3 of 900, against 573 for sample-and-select with GBFS and 298 for hFF with GBFS.
  - Ablation: the same loop fed numeric coverage instead of counterexamples reaches 572.7 (GBFS) and 412.3 (hill climbing).
  - One of 2,700 test evaluations failed because of a property violation.
- **Claimed vs demonstrated**: the property is checked on training tasks under a time cap that counts timeouts as passes; results average 3 runs.
- **Limitations**:
  - Authors: needs a property that is cheap to check and yields actionable counterexamples; checked only on training tasks; unconstrained Python is hard to interpret or verify.
  - Observed: deterministic domains.
- **Flags**: property checked on training tasks only.

#### B.8 Abstract-level entries

| Paper | Venue / tier | Depth | What is known |
|---|---|---|---|
| Corrêa, Pereira, Seipp. Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code. arXiv:2503.18809 | NeurIPS 2025, as cited by B.7 · High | Not read; facts from B.6 and B.7 | Samples a fixed number of heuristic programs per domain (25, per B.7), selects on training tasks, and uses the winner in greedy best-first search. No feedback loop. |
| Katz, Kokel, Srinivas, Sohrabi. Thought of Search. arXiv:2404.11833 | NeurIPS 2024 · High | Abstract | The LLM writes the successor function and goal test once, and search is then sound and complete. 100% on 4 search problems with few LLM calls; a human corrects the code. |
| Cao, Katz, Kokel, Srinivas, Sohrabi. Automating Thought of Search (AutoToS). arXiv:2408.11326 | Preprint · Low | Abstract | Unit-test feedback (generic and domain-specific) replaces the human; 100% on the evaluated domains with few calls. |
| Cui, Xia, Shen, Luo, He, Liang. Abstraction Generation for Generalized Planning with Pretrained LLMs. arXiv:2602.10485 | Preprint · Low | Abstract | LLMs write abstract features and qualitative numerical planning (QNP) abstractions of a domain from training tasks; an automated debugger detects abstraction errors and sends them back. Some LLMs produce useful abstractions when debugged. |
| Sohrabi, Ananthakrishnan, Kokel, Srinivas, Katz. Learning and Reusing Policy Decompositions for Hierarchical Generalized Planning with LLM Agents (HCL-GP). arXiv:2605.06957 | Preprint · Low | Abstract | Learns parameterised policies for LLM agents, extracts reusable components from successful runs into a library, and retrieves them by semantic search. AppWorld: 98.2% normal, 97.8% challenge; +15.8 points over static synthesis on challenge tasks; open models 62.5% with reuse against near zero without. No formal verification. |
| Deng et al. PlanU: Large Language Model Reasoning through Planning under Uncertainty. arXiv:2510.18442 | NeurIPS 2025 · High | Abstract | LLM planning in stochastic environments: Monte Carlo tree search where each node stores a quantile distribution of returns, with an exploration score for uncertain nodes. It plans per task; no generalized policy and no verification. |

### C. Verifier feedback versus resampling; LLM-Modulo

Work that tests whether a verifier's feedback helps an LLM more than simply sampling again, and the framing of LLMs as generators inside a verifier loop.

#### C.1 On the Self-Verification Limitations of Large Language Models on Reasoning and Planning Tasks

- **Citation**: K. Stechly, K. Valmeekam, S. Kambhampati. ICLR 2025. arXiv:2402.08115.
- **Link**: https://arxiv.org/abs/2402.08115
- **Venue / tier**: ICLR main track. **High**.
- **Depth**: Full main text (Sections 1–6); appendices not read.
- **Generated**: GPT-4 answers for Game of 24 and graph colouring (100 planar graphs), and plans for Blocksworld and Mystery Blocksworld.
- **Verifier**: either GPT-4 itself (a verdict plus a critique) or sound deterministic verifiers (SymPy, an edge check, VAL). Nothing probabilistic.
- **Loop**:
  - Back-prompting with the full history, up to 15 rounds.
  - With the sound verifier, the feedback content is varied: binary ("wrong"), the first error, or all errors.
  - "Sampling": the base prompt is re-asked without history until the verifier accepts, with k = 15 or 25; self-consistency is also tested.
- **Guarantee**: correctness of the outputs comes from the external verifier.
- **Evaluation** (100 instances per domain):
  - LLM self-critique does worse than standard prompting; the LLM verifier's false-negative rate reaches 96–97% on colouring and Mystery Blocksworld.
  - The sound verifier gives large gains, and the level of feedback detail barely matters: Blocksworld 60% binary, 87% first error, 83% all errors; colouring 38/37/34%.
  - Plain resampling with the sound verifier does about as well: Blocksworld 68% at k = 15 and 72% at k = 25; colouring 40% and 44%; Game of 24 28% and 42%, against 36–38% with feedback.
- **Claimed vs demonstrated**: the claims are scoped to these domains and GPT-4. Budgets are matched in rounds, not tokens; the token argument (resampling prompts do not grow) is made qualitatively.
- **Limitations**: observed: deterministic verifiers; one model (GPT-4, 2023–24); short answers rather than structured artifacts such as programs or policies; no quantitative verifier output.
- **Flags**: none beyond the scope noted.

#### C.2 Is Self-Repair a Silver Bullet for Code Generation?

- **Citation**: T. X. Olausson, J. P. Inala, C. Wang, J. Gao, A. Solar-Lezama. ICLR 2024. arXiv:2306.09896.
- **Link**: https://arxiv.org/abs/2306.09896
- **Venue / tier**: ICLR main track. **High**.
- **Depth**: Full main text (Sections 1–6); appendices not read.
- **Generated**: Python programs for HumanEval and a 300-task subset of APPS, written by CodeLlama-13B-instruct, GPT-3.5 and GPT-4 at temperature 0.8.
- **Verifier**: unit tests.
- **Loop**: a repair tree. The model draws several initial samples; each failing sample gets several joint feedback-and-repair samples, one level deep. This is compared with independent sampling at an equal number of programs (pass@k, where k counts every program in the tree). An appendix repeats the comparison with token-based cost and reports the same trends.
- **Guarantee**: tests only.
- **Evaluation**:
  - At matched budget, self-repair is often only marginally better and sometimes worse, especially at small budgets.
  - Spending the budget on diverse initial samples beats deeper repair. For GPT-4 on APPS, 10 initial samples × 1 repair reaches 1.05× pass@20, while 2 × 10 reaches 0.97× pass@22.
  - The quality of the feedback is the bottleneck. Feedback from a stronger model beats both baselines, and human feedback raises GPT-4's repair success from 33.3% to 52.6% (1.58×).
- **Claimed vs demonstrated**: well supported. Estimates are bootstrapped from one large repair tree per task, which the authors flag as a possible source of artefacts.
- **Limitations**: observed: deterministic tests; only one level of repair.
- **Flags**: none.

#### C.3 Position: LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks

- **Citation**: S. Kambhampati, K. Valmeekam, L. Guan, M. Verma, K. Stechly, S. Bhambri, L. Saldyt, A. Murthy. ICML 2024 position track, PMLR 235:22895–22907. arXiv:2402.01817.
- **Link**: https://proceedings.mlr.press/v235/kambhampati24a.html
- **Venue / tier**: ICML, position track; an argument rather than new experiments. **High** venue.
- **Depth**: Most (Sections 1–5 read; the extraction cut off the last related-work paragraph).
- **Content**:
  - LLMs act as candidate generators and approximate knowledge sources inside a generate–test–critique loop. The loop has hard critics (sound and model-based: VAL, simulators, tests) and soft critics (style, possibly another LLM).
  - A meta-controller turns critiques into the next prompt.
  - LLMs also help humans build the critics' models, reformulate candidates and refine specifications.
  - Soundness comes from the hard critics; completeness depends on how diverse the candidates are.
- **Evidence** (secondary, from cited studies): back-prompting GPT-4 with VAL reaches 82% on Blocksworld within 15 rounds, 70% on Logistics and about 10% on Mystery Blocksworld. Travel planning improves 6× over baseline with critics and 10 back-prompt cycles.
- **Claimed vs demonstrated**: a framework; the numbers come from cited work. No equal-budget resampling comparison here (Stechly et al. supply one).
- **Limitations**: observed: every cited result uses deterministic critics.
- **Flags**: position paper; evidence secondary.

#### C.4 Abstract-level entries

| Paper | Venue / tier | Depth | What the abstract says |
|---|---|---|---|
| Valmeekam, Marquez, Sreedharan, Kambhampati. On the Planning Abilities of Large Language Models: A Critical Investigation. arXiv:2305.15771 | NeurIPS 2023 (spotlight) · High | Abstract (numbers also via C.3) | Autonomous GPT-4 produces about 12% executable plans. Used as heuristic guidance for sound planners, or back-prompted with VAL, LLM output improves. |
| Huang, Chen, Mishra, Zheng, Yu, Song, Zhou. Large Language Models Cannot Self-Correct Reasoning Yet. arXiv:2310.01798 | ICLR 2024 · High | Abstract | Self-correction without external feedback does not improve reasoning and can hurt it. |
| Jha, Jha, Lincoln, Bastian, Velasquez, Ewetz, Neema. Neuro Symbolic Reasoning for Planning: Counterexample Guided Inductive Synthesis using LLMs and Satisfiability Solving. arXiv:2309.16436 | Preprint · Low | Abstract | The LLM is the CEGIS learner; Z3 checks plans and returns counterexamples; Blocksworld. |
| Li, Parsert, Polgreen. Guiding Enumerative Program Synthesis with Large Language Models. arXiv:2403.03997 | CAV 2024 · High | Abstract | An LLM alone falls short of the state of the art on SyGuS. An LLM guiding weighted probabilistic enumeration, in a feedback loop, beats the LLM alone, the enumerator alone and the competition winner. |
| Dantas, Cordeiro, Sun, Junior. The 4/δ Bound: Designing Predictable LLM-Verifier Systems for Formal Method Guarantee. arXiv:2512.02080 | Preprint · Low | Abstract | Models a four-stage LLM–verifier pipeline as an absorbing Markov chain. If each stage succeeds with probability at least δ, the pipeline terminates with probability 1 within at most 4/δ expected iterations; 90k simulated trials. A guarantee about the loop, not the artifact. |
| Göbel, Lorang, Zips, Glück. Agentic LLM Planning via Step-Wise PDDL Simulation: An Empirical Characterisation. arXiv:2603.06064 | Preprint · Low | Abstract | Claude Haiku 4.5 as an interactive search policy over a PDDL simulator, on 102 IPC Blocksworld instances, 180 s each. Fast Downward 85.3%, direct LLM planning 63.7%, agentic 66.7% at 5.7× the tokens per solution. The authors relate the small gain to step feedback that the agent must assess itself, unlike compiler or test feedback. |

Kumar and Cohen's in-context error correction (L-ICL) is listed under [K](#k-reusing-solved-instances-or-experience-in-context).

### D. LLMs with non-probabilistic formal verification; LLM-written models

LLM output checked by deterministic model checkers, automata or validators, and work in which the LLM writes the model itself.

#### D.1 Fine-Tuning Language Models Using Formal Methods Feedback: A Use Case in Autonomous Systems (DPO-AF)

- **Citation**: Y. Yang, N. P. Bhatt, T. Ingebrand, W. Ward, S. Carr, Z. Wang, U. Topcu. MLSys 2024. arXiv:2310.18239.
- **Link**: https://arxiv.org/abs/2310.18239
- **Venue / tier**: MLSys main track. **Medium**.
- **Depth**: Partial. Sections 1–5.2 and the start of 5.3 were read; the HTML extraction stopped there, so the rest of 5.3 and the appendix (specifications, world models) were not read.
- **Generated**: Llama2-7B writes step-by-step driving instructions in natural language (e.g., "turn right at the traffic light"). The LLM aligns them to given propositions and actions, and they are parsed into a finite-state controller (GLM2FSA).
- **Verifier**: NuSMV, on the product of a hand-built automaton model of the world and the controller, against 15 LTL traffic rules. Qualitative and deterministic, with counterexample traces. Alternatively, an empirical check: the fraction of Carla simulation traces that satisfy each rule.
- **Loop**: not prompt refinement. The number of rules each sampled response satisfies ranks pairs of responses; about 3000 such preference pairs train the model by DPO with LoRA. Checkpoints are re-evaluated every 20 epochs.
- **Guarantee**: each controller satisfies the LTL rules on the abstract model, assuming actions succeed and perception is correct. Transfer to a real vehicle is argued, not measured.
- **Evaluation**: driving tasks. The share of satisfied specifications rises from about 60% to about 90% on both training and validation tasks. Per-rule satisfaction in Carla also rises. Five seeds that differ only in data order.
- **Claimed vs demonstrated**: one domain, one base model, hand-written specifications and models. Real-world transfer is argued only.
- **Limitations**: observed: counterexamples are produced but used only to rank responses.
- **Flags**: guarantee relative to a hand-built abstract model; single model.

#### D.2 Large Language Models as Planning Domain Generators

- **Citation**: J. Oswald, K. Srinivas, H. Kokel, J. Lee, M. Katz, S. Sohrabi. ICAPS 2024, 34(1):423–431. arXiv:2405.06650.
- **Link**: https://ojs.aaai.org/index.php/ICAPS/article/view/31502
- **Venue / tier**: ICAPS main track. **High**.
- **Depth**: Partial. Sections 1–4 were read up to the action-reconstruction-error analysis, where the extraction stopped; the conclusion was not read.
- **Generated**: PDDL action schemas, one action at a time, from a natural-language description of each action ("base", "flipped" or "random" description classes) and a given predicate list with glosses. The prompt holds 3 examples from other domains. Seven LLMs (LLaMA-2 7B/13B/70B base and chat, StarCoder) with greedy decoding.
- **Verifier**: deterministic.
  - A parser checks syntax, and semantic integration checks follow.
  - "Heuristic domain equivalence": the top 100 plans (from the K* planner) for 2 problems per domain are cross-validated with VAL between the original and the generated domain.
  - This soundly refutes equivalence but can confirm it only heuristically.
- **Loop**: none; this is an evaluation framework.
- **Guarantee**: none.
- **Evaluation**: 9 domains, 2 of them created after the models' training cutoff; 60 prompts per action. The best model (LLaMA-2-70B) writes valid PDDL 94% of the time and heuristically equivalent domains 25% of the time. Chat models do worse than base models; the description class matters little.
- **Claimed vs demonstrated**: consistent with the results. Automatic evaluation needs a reference domain to compare against.
- **Limitations**: observed: older open models; no repair loop.
- **Flags**: no loop.

#### D.3 Abstract-level entries

| Paper | Venue / tier | Depth | What the abstract says |
|---|---|---|---|
| Guan, Valmeekam, Sreedharan, Kambhampati. Leveraging Pre-trained LLMs to Construct and Utilize World Models for Model-based Task Planning. arXiv:2305.14909 | NeurIPS 2023 · High | Abstract | The LLM writes PDDL models, which are corrected with PDDL validators and human feedback (via a natural-language translation) before sound planners use them; 40+ actions, 48 tasks. |
| Yang, Gaglione, Neary, Topcu. Automaton-Based Representations of Task Knowledge from Generative Language Models (GLM2FSA). arXiv:2212.01944 | Preprint (submitted to JAIR) · Low | Abstract | Builds a finite-state automaton from an LLM's textual task knowledge, verifies it against user specifications, and refines the queries to the LLM from verification outcomes such as counterexamples. |
| Yang, Bhatt, Ward, Hu, Biswas, Topcu. Joint Verification and Refinement of Language Models for Safety-Constrained Planning. arXiv:2410.14865 | Preprint · Low | Abstract | Converts LLM-written robot programs into automata and verifies them against safety specifications. Proves that any composition of verified programs stays safe, and fine-tunes on verification outcomes: 30% more specification-compliant programs, half the training time. |
| Ramani, Tawosi, Alamir, Borrajo. Bridging LLM Planning Agents and Formal Methods: A Case Study in Plan Verification. arXiv:2510.03469 | AgenticSE workshop @ ASE 2025 · Low | Abstract | The LLM translates natural-language plans into Kripke structures and LTL, and NuSMV checks them; on a PlanBench subset GPT-5 reaches 96.3% F1 as a classifier. Whether the models are semantically correct is left open. |
| Castle, Rubeck. Agentic Synthesis against Counterexample-Supplemented Sketches (CESS). arXiv:2607.15854 | Preprint · Low | Abstract | A human sketch plus a coding agent. Counterexamples approved by an operator become general rules; edits are scoped, regression sets and deterministic replay are kept. One synthetic app, one model; the authors caution against generalising. |
| Ma et al. Eureka: Human-Level Reward Design via Coding Large Language Models. arXiv:2310.12931 | ICLR 2024 · High | Abstract | The LLM writes reward code, refined by evolutionary search using RL training statistics as feedback; 29 environments; beats expert rewards on 83% of tasks. The evaluator is empirical. |
| Romera-Paredes et al. Mathematical discoveries from program search with large language models (FunSearch). Nature, 2023 | Nature · Medium (outside the list) | Not re-read | The LLM proposes programs, an automated evaluator scores them, and an evolutionary program database keeps the best. The evaluator scores deterministically and proves nothing. |
| Zuzuárregui, Carpin. As You Wish: Mission Planning with Formal Verification using LLMs in Precision Agriculture. arXiv:2606.18519 | ICRA 2026 (per arXiv) · Medium (outside the list) | Abstract | Adds LTL-based feedback loops to an LLM mission planner for field robots, using two different commercial LLMs for specification and for verification. Reports that LLM-written LTL formulas are a weak point, and how the design addresses it. |
| Liu, Jiang, Zhang, Liu, Zhang, Biswas, Stone. LLM+P: Empowering Large Language Models with Optimal Planning Proficiency. arXiv:2304.11477 | Preprint · Low | Abstract | The LLM translates a natural-language problem into PDDL, a classical planner solves it, and the plan is translated back; optimal plans for most benchmark problems where the LLM alone fails. |
| Tihanyi, Jain, Charalambous, Ferrag, Sun, Cordeiro. A New Era in Software Security: Towards Self-Healing Software via Large Language Models and Formal Verification (ESBMC-AI). arXiv:2305.14752 | Preprint · Low | Abstract | Bounded model checking (ESBMC) finds vulnerabilities and counterexamples (stack trace, line, error type); an LLM repairs the C code; the fix is re-verified. 50,000 programs from FormAI. |

### E. Statistical and conformal guarantees for LLM planners

Guarantees that hold over a distribution of tasks rather than for a checked artifact.

#### E.1 Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners (KnowNo)

- **Citation**: A. Z. Ren et al. CoRL 2023, PMLR 229. arXiv:2307.01928.
- **Link**: https://proceedings.mlr.press/v229/ren23a.html
- **Venue / tier**: CoRL main track (outside the list). **Medium**.
- **Depth**: Partial. Sections 1–4.2 were read; the extraction stopped during the hardware results, and the appendices were not read.
- **Generated**: PaLM-2L proposes candidate next steps as multiple choice. The robot acts if the conformal prediction set holds a single option, and otherwise asks a human.
- **Verifier**: none formal. Split conformal prediction calibrates the LLM's next-token scores on a calibration set (400 scenarios, δ = 0.01), with calibration at sequence level for multi-step plans.
- **Loop**: none; the human in the loop supplies help, not refinement.
- **Guarantee**: statistical. Task success is at least 1 − ε over scenarios drawn i.i.d. from the calibration distribution, assuming the human helps correctly. Prediction sets are minimal if the scores are calibrated.
- **Evaluation**: PyBullet tabletop tasks with attribute, numeric and spatial ambiguity; multi-step rearrangement on hardware; a bimanual task. Baselines: simple sets, ensemble sets, prompt sets, binary help, no help. KnowNo meets the target success rate and asks for help 10–24% less.
- **Claimed vs demonstrated**: consistent within the i.i.d. assumption.
- **Limitations**: observed: the guarantee is distributional and needs a calibration set from the deployment distribution.
- **Flags**: statistical guarantee under an i.i.d. assumption; no loop.

#### E.2 Abstract-level entries

| Paper | Venue / tier | Depth | What the abstract says |
|---|---|---|---|
| Wang, Tong, Tan, Vorobeychik, Kantaros. Conformal Temporal Logic Planning using Large Language Models (HERACLEs). arXiv:2309.10092 | ACM Transactions on Cyber-Physical Systems (accepted, per arXiv) · Medium | Abstract | Missions are LTL formulas over sub-tasks written in natural language. A symbolic planner orders the sub-tasks, an LLM writes the robot actions, and conformal prediction links the two, reaching user-defined mission success rates in theory and experiments. |
| Sundarsingh, Wang, Deshmukh, Kantaros. ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees. arXiv:2504.21022 | Preprint · Low | Abstract | Builds LTL formulas through a sequence of LLM question-answering steps, using conformal prediction to decide when to ask an auxiliary model or the user; reaches user-defined translation accuracy. |

Pro2Guard ([A.5](#a5-pro2guard-proactive-runtime-enforcement-for-llm-agent-safety-via-probabilistic-prediction)) gives a PAC bound on a learned Markov chain. Gros et al. ([I.1](#i1-per-domain-generalizing-policies-on-validation-instances-and-scaling-behavior)) and Schnitzer et al. ([F.4](#f4-certifiably-robust-policies-for-uncertain-parametric-environments)) give statistical guarantees without LLMs.

### F. Families of MDPs, multi-environment and robust policies (no LLM)

One policy is synthesised, or checked, for a whole set of models at once.

#### F.0 What "family" means in this line of work

What "family" means in the Junges and Češka papers and related work, and what followed Policies Grow on Trees.

| Line of work | What a family is | What is synthesised | What members share |
|---|---|---|---|
| Češka et al., TACAS 2019 (title only, from reference lists); PAYNT, CAV 2021 ([G.1](#g1-paynt-a-tool-for-inductive-synthesis-of-probabilistic-programs)) | A PRISM or JANI program with holes, each with a finite set of options. Every assignment of the holes gives one Markov chain. | One hole assignment that meets the specification, or a proof that none does. | The program's variables. Members may differ in transitions and reachable states. |
| Andriushchenko et al., UAI 2022 and JAIR 2025 ([G.2](#g2-inductive-synthesis-of-finite-state-controllers-for-pomdps), [G.3](#g3-an-oracle-guided-approach-to-constrained-policy-synthesis-under-uncertainty)) | The design space of controllers for one POMDP (e.g., all finite-state controllers with k memory nodes); each controller induces a Markov chain. JAIR 2025 generalises this as a "coloured MDP". | One controller. | The POMDP. |
| Policies Grow on Trees, ATVA 2024 ([F.1](#f1-policies-grow-on-trees-model-checking-families-of-mdps)) | A finite index set (hole assignments of an MDP sketch); each index gives one MDP. | A policy tree: the index set is split into subfamilies, and each leaf holds one memoryless policy that wins on every member of that subfamily. | State space and action sets. Transitions differ, and holes may change the transition graph. |
| rfPG, IJCAI 2025 ([F.3](#f3-robust-finite-memory-policy-gradients-for-hidden-model-pomdps-rfpg)) | A hidden-model POMDP: a finite set of POMDPs. | One finite-state controller with the best worst-case value. | States, actions, observations and the observation function. |
| SMPMC, AAAI 2026 ([F.2](#f2-constrained-and-robust-policy-synthesis-with-satisfiability-modulo-probabilistic-model-checking)) | Controllable parameters (the policy) and uncontrollable parameters (initial states, members, perturbations) over one coloured MDP. | Controllable values such that the specification holds for every uncontrollable value, under structural constraints such as "is a decision tree of size n". | The coloured MDP. |
| Multi-environment MDPs (Raskin and Sankur 2014; van der Vegt et al. 2023; [F.5](#f5-abstract-level-entries)) | A finite set of MDPs over one state and action space; the controller does not know which one it is in. | One policy for all members. It may need memory: exponential memory in the worst case for almost-sure reachability (van der Vegt et al.). | States and actions. |
| Uncertain parametric MDPs (Badings et al. 2022; Schnitzer et al. 2025, [F.4](#f4-certifiably-robust-policies-for-uncertain-parametric-environments)) | A parametric MDP with an unknown distribution over parameter values; each value gives one MDP. | Schnitzer et al.: a policy plus a PAC bound on the risk that an unseen member falls below a stated value. Badings et al.: a PAC bound on whether some policy reaches a value. | States and actions, and by default the transition graph. |
| Parameterized MDPs (Křetínský and colleagues: 1–2–3–Go!, VMCAI 2025, [I.3](#i3-123go-policy-synthesis-for-parameterized-markov-decision-processes-via-decision-tree-learning-and-generalization)) | A PRISM model whose parameters (number of modules, variable bounds, deadlines) give instances of different sizes. Not called a family there. | One decision tree over the state variables, learned from the optimal policies of a few small instances and applied to instances of any size. | State-variable names and action labels, not the state space. |

In every row except the last, members share a state space (or an observation space), and the family is either finite and enumerated through holes or indices, or a distribution over parameters. Those definitions do not cover instances whose state spaces differ, such as one model at different sizes. Two lines of work handle such instances: parameterized MDPs, where a policy over state variables is carried from small instances to larger ones (last row), and planning, which treats them as instances of one domain ([B](#b-llm-written-generalized-policies-and-programs-for-planning), [I](#i-generalising-policies-to-held-out-and-larger-instances)).

**Forward citations of Policies Grow on Trees** (Semantic Scholar, 2026-10-08): three papers, SMPMC ([F.2](#f2-constrained-and-robust-policy-synthesis-with-satisfiability-modulo-probabilistic-model-checking)), rfPG ([F.3](#f3-robust-finite-memory-policy-gradients-for-hidden-model-pomdps-rfpg)) and KAPS ([F.5](#f5-abstract-level-entries)). None involves an LLM.

#### F.1 Policies Grow on Trees: Model Checking Families of MDPs

- **Citation**: R. Andriushchenko, M. Češka, S. Junges, F. Macák. ATVA 2024. arXiv:2407.12552.
- **Link**: https://arxiv.org/abs/2407.12552 (artifact: doi 10.5281/zenodo.12569976)
- **Venue / tier**: ATVA, a peer-reviewed formal-methods conference outside the list. **Medium**.
- **Depth**: Full, including appendices. Seed paper.
- **Generated**: no LLM. A policy tree. Inner nodes split the family's index set (hole assignments of an MDP sketch written in PRISM). Each leaf holds either one memoryless deterministic policy (a state-to-action table) that wins on every MDP in that subfamily, or "unsatisfiable".
- **Verifier**: Storm, exact, in two ways:
  - **Game abstraction**: player 1 picks the action and player 2 picks which family member executes it. If the game value meets the threshold, player 1's policy is robust for the whole subfamily.
  - **Quotient MDP**: if its maximum is below the threshold, every member is unsatisfiable.
  - Value iteration for MDPs and policy iteration for games; not interval iteration.
- **Loop**: divide and conquer by abstraction refinement.
  - Look for a robust policy; test for unsatisfiability.
  - Otherwise split on the indices where the optimal game or quotient policy is inconsistent (optimistic or pessimistic splitting), and recurse.
  - Post-processing merges compatible policies (those that agree on jointly reachable states) and re-verifies them.
- **Guarantee**: sound and complete for finite families, with at most n splits. Each leaf policy provably wins on every member of its subfamily. One objective: indefinite-horizon reachability with a threshold.
- **Evaluation**:
  - 15 sketch benchmarks from the literature (av, dodge, dpm, obstacles, rover, uav, virus, rocks); families of 4·10³ to 4·10⁸ members; quotient MDPs of up to 10⁵ states.
  - Baselines, both run with Storm: one-by-one enumeration, and one all-in-one MDP.
  - On most benchmarks the number of iterations stays below 2% of the family size.
  - On dodge-3, 246 policies cover 9.4·10⁷ MDPs in about 24 minutes; one-by-one enumeration would be about 1129× slower (extrapolated). The all-in-one MDP runs out of memory on many benchmarks.
  - Ablations: a randomised player-2 abstraction does worse; splitting variants are compared.
- **Claimed vs demonstrated**: the claims match the evidence. Speedups for runs beyond 2 h are extrapolated and marked as such.
- **Limitations**:
  - Authors:
    - The game abstraction is too pessimistic when player 2 can adapt to the agent. On Rocks-6-4 a single policy covering every member exists but is not found; virus has the same problem.
    - Post-processing does not minimise the number of policies.
    - Only memoryless robust policies; robust policies may need memory, and deciding whether one exists is NP-complete.
    - Policy size is not addressed (left to dtControl).
  - Observed:
    - Reachability only: no LTL, multiple objectives or reward bounds.
    - The family must be finite and enumerated through the holes of one sketch with a shared state space.
    - Policies are per-state tables, and nothing is claimed for members outside the family.
- **Flags**: none.

#### F.2 Constrained and Robust Policy Synthesis with Satisfiability-Modulo-Probabilistic-Model-Checking

- **Citation**: L. Heck, F. Macák, M. Češka, S. Junges. AAAI 2026. arXiv:2511.08078.
- **Link**: https://arxiv.org/abs/2511.08078
- **Venue / tier**: AAAI main track. **High**.
- **Depth**: Most. Sections 1–3 and 5–7 were read. Section 4 (the CDCL(T) theory solver and quantifier handling) and the appendices were skimmed by heading.
- **Generated**: no LLM. An assignment to the controllable parameters, which is the policy. It must meet arbitrary first-order structural constraints, for example: representable as a decision tree with n nodes; observation-based or finite-memory; the same action on tiles of the same colour. It must also be robust over the uncontrollable parameters: initial states, members of a multi-environment MDP, perturbations.
- **Verifier**: Storm as a custom theory in Z3 (a user propagator). Model checking partial assignments gives bounds (optimistic value iteration, precision 1e-4) that prune the search. Model-based quantifier instantiation handles the "exists policy, for all environments" alternation.
- **Loop**: CDCL(T) search: SAT branching combined with model-checking bounds and conflicts. The authors present it as a tighter alternative to CEGIS-style generalisation.
- **Guarantee**: complete over the finite parameter space. It can find a provably smallest robust decision-tree policy, or prove that none of size n exists. Values hold to the 1e-4 precision of optimistic value iteration. Specifications are undiscounted, indefinite-horizon expected rewards, with reachability encoded as a reward.
- **Evaluation**:
  - 372 benchmarks: sets of Markov chains, fixed-memory controllers for POMDPs, multi-environment MDPs (from Policies Grow on Trees and van der Vegt et al.), multi-environment POMDPs (from rfPG). 30-minute timeout.
  - SMPMC solves 262 (75 that no other method solves). PAYNT's abstraction refinement solves 129, PAYNT's CEGIS 78, and an SMT encoding over linear real arithmetic 92.
  - Robust decision trees of up to 15 nodes.
  - On Rocks-6-4 it finds a 5-node robust decision-tree policy (a spiral that exploits slipping), which Policies Grow on Trees could not find.
- **Claimed vs demonstrated**: consistent. Three repeated runs. The monolithic SMT baseline for the new problem class is the authors' own implementation.
- **Limitations** (authors and observed):
  - Parameter spaces and model sizes must be finite and explicitly bounded; plain synthesis runs on coloured MDPs of up to about 1M states.
  - Decision-tree predicates are over state variables with fixed encodings.
  - No search heuristics yet.
  - Robustness is over a finite, enumerated family.
- **Flags**: none.

#### F.3 Robust Finite-Memory Policy Gradients for Hidden-Model POMDPs (rfPG)

- **Citation**: M. F. L. Galesloot, R. Andriushchenko, M. Češka, S. Junges, N. Jansen. IJCAI 2025, pp. 8518–8526. arXiv:2505.09518.
- **Link**: https://arxiv.org/abs/2505.09518
- **Venue / tier**: IJCAI main track. **High**.
- **Depth**: Full main text (Sections 1–6, arXiv HTML v3). The appendices (the controller-size heuristic, benchmark details, extended tables) were not read.
- **Generated**: no LLM. A stochastic finite-state controller, softmax-parameterised; its size is chosen by a heuristic on a sample of POMDPs. It should be robust over a hidden-model POMDP: a finite indexed set of POMDPs that share states, actions, observations and the observation function, and differ in transitions and rewards.
- **Verifier**: deductive robust policy evaluation with PAYNT. The controller induces a family of Markov chains, which PAYNT handles as one quotient MDP. This gives the exact worst-case member and robust value without enumerating the family.
- **Loop**: alternate two steps until a 1 h timeout, keeping the best robust controller (progress is not monotone):
  1. Robust evaluation, which finds the worst-case POMDP.
  2. Ten projected subgradient-ascent steps on that POMDP.
- **Guarantee**: the reported robust value is exact; it lower-bounds that controller's value on every member of the family. No optimality guarantee: the method is local, and the problem is undecidable in general.
- **Evaluation**:
  - Six hidden-model POMDPs: Obstacles, Network, Avoid, Rover, and DPM with up to 131k POMDPs. One of them is an MDP family from Policies Grow on Trees.
  - Baselines are Saynt and gradient-ascent variants, run on random subsets of 10 POMDPs.
  - A variant trained on 10 POMDPs is compared with the baselines, all evaluated on the full family, which tests generalisation to unseen members.
  - 10 seeds. Ablation: random POMDP selection instead of the worst case (domain randomisation).
  - rfPG is best on most benchmarks.
- **Claimed vs demonstrated**: supported. "Generalises to unseen POMDPs" means unseen members of the same finite family, evaluated exactly.
- **Limitations**: observed:
  - Reachability-style rewards only.
  - Finite families with shared state and observation spaces.
  - Controller parameters are numeric tables.
- **Flags**: none.

#### F.4 Certifiably Robust Policies for Uncertain Parametric Environments

- **Citation**: Y. Schnitzer, A. Abate, D. Parker. Conference version: "Learning Provably Robust Policies in Uncertain Parametric Environments", TACAS 2025. Extended version: arXiv:2408.03093 (v5, Mar 2025).
- **Link**: https://arxiv.org/abs/2408.03093
- **Venue / tier**: TACAS 2025, per the extended version's own reference to its conference version. **High**.
- **Depth**: Most. The introduction, setup, Sections 3.2–3.3, experiments and conclusion were read. Sections 3.1 and 3.4, the proofs and the benchmark appendices were not.
- **Generated**: no LLM. A robust memoryless policy for an uncertain parametric MDP: a parametric MDP with an unknown distribution over parameter values, whose transition probabilities are also unknown and seen only through trajectories. Two learners:
  - **Robust interval-MDP learning**: learn an interval MDP for each training environment, merge the intervals, and take the optimal policy of the merged model.
  - **Robust meta-RL**: max-min policy gradient across the training environments.
- **Verifier**: an extension of PRISM.
  - For each environment in a separate verification set, an interval MDP is learned from trajectories that contains the true MDP with confidence 1 − γ (γ = 10⁻⁴). Robust value iteration on it gives a lower bound on the policy's value in that environment.
  - A new scenario-optimisation theorem turns these lower bounds into a PAC bound on the risk that the policy's value in an unseen environment falls below the stated guarantee (overall confidence 1 − β, β = 10⁻²).
  - A second theorem trades a higher risk for a higher guarantee by discarding the k worst samples.
- **Loop**: none. Environments are split into training and verification sets, with no iteration between them.
- **Guarantee**: with probability at least 1 − β over the sampled environments, the chance that an unseen environment from the same distribution yields a value below the guarantee is at most ε.
  - Sampled environments are assumed i.i.d.
  - By default all members share a transition graph; the paper explains how to lift this.
  - Computing the risk depends only on the number of verification samples, not on model size.
- **Evaluation**:
  - Six benchmarks: UAV motion planning, aircraft collision avoidance, a chain problem, a betting game, a semi-autonomous vehicle, Firewire.
  - 600 sampled environments split evenly (Firewire uses 150 for verification); up to 10⁶ trajectories per environment.
  - UAV example: the true worst-case value over the hidden sampled MDPs is 0.711. The certified guarantee is 0.710 with a risk bound of 0.027. On 1000 fresh environments the policy fell below the guarantee in 0.3% of cases.
  - Across benchmarks the risk bounds are 0.027–0.055 with no samples discarded, against empirical risks of 0.2–0.5%.
  - 0.3–15 s per 10⁴ trajectories.
- **Claimed vs demonstrated**: the bounds are shown to be tight on these benchmarks. The two policy learners are deliberately not compared; their statistics are in the appendix.
- **Limitations**:
  - Authors: uncertain specifications are future work.
  - Observed: the guarantees are distributional (i.i.d. environments), not per-instance proofs; members share a state space.
- **Flags**: statistical guarantee under an i.i.d. assumption.

#### F.5 Abstract-level entries

| Paper | Venue / tier | Depth | What the abstract says |
|---|---|---|---|
| Raskin, Sankur. Multiple-Environment Markov Decision Processes. arXiv:1405.4733 | FSTTCS 2014 · Medium | Snippet and abstract | Defines multi-environment MDPs: a finite set of MDPs over one state space with different transitions; the controller does not know which applies. Several problems that are undecidable for POMDPs become decidable. |
| Chatterjee, Chmelík, Karkhanis, Novotný, Royer. Multiple-Environment Markov Decision Processes: Efficient Analysis and Applications | ICAPS 2020 · High | Snippet | Efficient algorithms for discounted-sum objectives in multi-environment MDPs, with applications. |
| van der Vegt, Jansen, Junges. Robust Almost-Sure Reachability in Multi-Environment MDPs. arXiv:2301.11296 | TACAS 2023 · High | Abstract | One policy that reaches the goal almost surely in every environment. Deciding this is PSPACE-complete (EXPTIME for POMDPs), and policies may need exponential memory. Implemented and benchmarked. |
| Bovy, Probine, Suilen, Topcu, Jansen. Multi-Environment POMDPs: Discrete Model Uncertainty Under Partial Observability. arXiv:2510.23744 | NeurIPS 2025 · High | Abstract | Policies optimal in the worst case over a finite set of POMDPs, via a reduction to adversarial-belief POMDPs; exact and point-based algorithms; benchmarks extended to the multi-environment setting. |
| Suilen, van der Vegt, Junges. (Exact title not recorded.) | CONCUR 2024 · Medium | Snippet | Almost-sure Rabin objectives in multi-environment MDPs; PSPACE-complete. |
| Chatterjee, Doyen, Raskin, Sankur. (Exact title not recorded.) | ICALP 2025 · Medium | Snippet | Parity and value-1 problems for multi-environment MDPs. |
| Bordais, Raskin. Multi-Environment MDPs with Prior and Universal Semantics. arXiv:2602.10938 | Preprint · Low | Snippet | Compares a prior (Bayesian) semantics with a universal (robust) semantics for multi-environment MDPs. |
| Lutz, Vos, Spaan, Lukina. Optimizing Minimax Regret in Uncertain MDPs with Small Sets of Policies (KAPS). arXiv:2608.02509 | Preprint · Low | Abstract | Synthesises k policies for a finite set of MDPs sharing states and actions, minimising worst-case regret. NP-hard; exact nested branch-and-bound; the largest gain comes from going from 1 to 2 policies. Cites Policies Grow on Trees. |
| Badings, Cubuktepe, Jansen, Junges, Katoen, Topcu. Scenario-based verification of uncertain parametric MDPs | STTT 24(5), 2022 · Medium | Title only (cited by F.1 and F.4) | Per F.4: scenario-based PAC guarantees that some policy reaches a given value, with sampled parameter values fully known. Not read. |

The CONCUR 2024 and ICALP 2025 rows are described from search snippets; their exact titles were not recorded.

### G. Synthesis and learning with probabilistic model checking in the loop (no LLM)

A search procedure or learner proposes candidates, and a probabilistic model checker evaluates them or prunes the search.

#### G.1 PAYNT: A Tool for Inductive Synthesis of Probabilistic Programs

- **Citation**: R. Andriushchenko, M. Češka, S. Junges, J.-P. Katoen, Š. Stupinský. CAV 2021, LNCS 12759, pp. 856–869.
- **Link**: https://doi.org/10.1007/978-3-030-81685-8_40 (code: github.com/randriu/synthesis; artifact: doi 10.5281/zenodo.4726056)
- **Venue / tier**: CAV, tool paper. **High**.
- **Depth**: Full. Seed paper.
- **Generated**: no LLM. A hole assignment ("realisation") of a user-written PRISM or JANI sketch: holes in guards and updates, each with a finite set of options, plus restrictions. This defines a finite family of Markov chains. Examples: power-manager thresholds and profiles, a maze controller with 1 bit of memory, coin biases in Herman's protocol.
- **Verifier**: Storm, exact. It checks single Markov chains, and checks a quotient MDP for a set of realisations to get lower and upper bounds. Z3 represents the unexplored part of the design space. Specifications are conjunctions of reachability-probability and expected-reward constraints, with an optional objective (feasibility, maximality, or maximality within ε).
- **Loop**: oracle-guided inductive synthesis, with two oracles and a hybrid:
  - **CEGIS**: check one realisation. If it fails, compute a counterexample (a critical sub-Markov chain, by MaxSat or greedy state expansion) and prune every realisation that agrees on the holes involved.
  - **Abstraction refinement**: check the quotient MDP of a subfamily. All satisfying, all failing, or inconclusive; if inconclusive, split.
  - **Hybrid**: switches between the two, sharing bounds.
- **Guarantee**: complete. It proves existence or non-existence, and optimality, over the finite family; every returned program is exactly model checked.
- **Evaluation**: five case studies (DPM with 43M realisations, Maze 9.4M, Herman 3.1M, Pole 1.3M, Grid 65k), compared with one-by-one enumeration (estimated for the large ones). For example, hard DPM takes 9.3 h against an estimated 35 days; Herman takes 17 minutes against an estimated 1.5 days.
- **Claimed vs demonstrated**: a tool paper; the speedups over enumeration are partly estimated, and marked as such.
- **Limitations**: observed:
  - A human writes the sketch and the option domains.
  - This paper handles families of Markov chains; MDP sketches came later (F.1, G.3).
  - The result is one hole assignment; nothing generalises outside the family.
- **Flags**: none.

#### G.2 Inductive Synthesis of Finite-State Controllers for POMDPs

- **Citation**: R. Andriushchenko, M. Češka, S. Junges, J.-P. Katoen. UAI 2022, PMLR 180:85–95.
- **Link**: https://proceedings.mlr.press/v180/andriushchenko22a.html (artifact: doi 10.5281/zenodo.6637489)
- **Venue / tier**: UAI (outside the list). **Medium**.
- **Depth**: Full (11 pages including references). Seed paper.
- **Generated**: no LLM. Deterministic finite-state controllers (Mealy machines over observations) for POMDPs. The design space is the family of controllers with k memory nodes, or "reduced" controllers with memory per observation. The synthesiser enumerates and splits candidates.
- **Verifier**: Storm, exact. It checks the Markov chain induced by one controller (an exact value), and an MDP abstraction of the whole controller family (observation-free policies, which give upper and lower bounds). An SMT solver (CVC5) holds the remaining design space. Specifications: indefinite-horizon reachability and expected-reward constraints, plus one objective; multiple objectives are supported.
- **Loop**: nested, two-level oracle-guided inductive synthesis.
  - **Inner**: find the best controller in the current design space. Either by abstraction refinement, splitting on the most significant inconsistent parameter (weighted by expected visits and value variance); or by counterexample pruning, where a critical sub-Markov chain becomes a conflict that removes every controller sharing the relevant parameters.
  - **Outer**: add memory at the observation where the optimal observation-free policy is most inconsistent. Symmetry reduction prunes equivalent controllers.
  - The search is anytime: each admissible controller tightens the optimisation constraint.
- **Guarantee**: every reported controller's value is exactly model checked, and the abstraction gives sound bounds on the whole family. Complete only with full refinement; the default strategy is incomplete by design.
- **Evaluation**:
  - POMDP benchmarks from Bork et al. (grid-av, grid, maze, crypt, nrp, hallway, drone, refuel, netw, rocks), up to about 3·10⁴ states.
  - Compared with PRISM's POMDP solver and Storm's belief-based under-approximation; compared qualitatively, through reported numbers, with MILP controller synthesis.
  - Competitive or better on models with moderate numbers of observations and actions. It finds small controllers (0–5 extra memory nodes).
  - Fails on Rocks and on a larger Netw. The counterexample and hybrid variants did not help on most models.
- **Claimed vs demonstrated**:
  - "Competitive with belief-based methods" holds for moderate observation counts.
  - The MILP comparison is qualitative: Hallway was re-modelled and reported numbers were used.
  - The authors note that the experiments cannot separate the heuristics' individual effects.
- **Limitations**:
  - Authors: comparisons across algorithms need more structure; symmetry reduction can discard optima; the default strategy is incomplete; scaling suffers with many observations or actions.
  - Observed: the structure (memory model, design space) is enumerated rather than informed by a prior; controllers are tables over observations and memory; one fixed POMDP.
- **Flags**: none.

#### G.3 An Oracle-Guided Approach to Constrained Policy Synthesis Under Uncertainty

- **Citation**: R. Andriushchenko, M. Češka, F. Macák, S. Junges, J.-P. Katoen. JAIR 82 (2025), pp. 433–469.
- **Link**: https://doi.org/10.1613/jair.1.16593
- **Venue / tier**: JAIR (a top AI journal, outside the list). **Medium**.
- **Depth**: Partial. Sections 1 and 6–8 and Appendix A were read. Sections 2.2–5 (the coloured-MDP formalism, the splitting score, counterexamples modulo bounds) were skimmed by heading only, for time.
- **Generated**: no LLM. A policy from a finite design space given as a "coloured MDP", covering:
  - hole assignments of probabilistic program sketches (programmatic policies, e.g., the DPM power manager);
  - deterministic finite-state controllers for POMDPs (k memory nodes);
  - pairs of controllers for Dec-POMDPs;
  - controllers for constrained POMDPs.
  A human writes the sketch or chooses k.
- **Verifier**: Storm, exact. It checks an MDP abstraction of a part of the design space (bounds) and single candidates as Markov chains. Counterexamples are critical sub-Markov chains, computed greedily and "modulo bounds".
- **Loop**: oracle-guided inductive synthesis, anytime, with an ε-optimal option:
  - **Abstraction refinement** splits sub-spaces using a score of how much the inconsistent choices matter.
  - **Counterexample generalisation** prunes every candidate that shares the counterexample's relevant parameters.
  - A manager switches between the two adaptively.
- **Guarantee**: complete exploration of the finite design space, which yields feasibility and optimality proofs within that space. Every returned policy is exactly model checked, and the constraints are guaranteed; the CNLP and PGA baselines violate the cost bound on tiger-grid.
- **Evaluation**:
  - DPM: 16,200 programs in 3 minutes; 43M programs in 6.5 h, against more than a month by enumeration; within 0.2% of optimal in 7 minutes.
  - Constrained maze POMDP: no controller with 1 or 2 memory nodes exists; the best 3-node controller among 4.1·10¹¹ is found in 42 s.
  - Meeting-grid Dec-POMDP: pairs of 2-node controllers, 10¹⁸ candidates, 15 s to 17 minutes.
  - Constrained POMDPs: better discounted reward than CNLP and PGA, while satisfying the cost bound.
  - Dec-POMDPs: better than InfJESP on Recycling and Box-pushing; struggles on DecTiger.
  - Engines compared (hybrid, abstraction refinement, counterexample generalisation): up to 3000× faster than PAYNT 2021.
- **Claimed vs demonstrated**: consistent. The CNLP and PGA numbers are taken from Wray and Czuprynski (2022), not rerun.
- **Limitations**:
  - Authors: probabilistic observations must be unfolded; domains that need much memory (DecTiger) blow up the abstraction; counterexample generalisation alone is weak where no small conflicts exist; it does not support reward-maximising specifications.
  - Observed: the design space (sketch, memory size, observation model) is human-specified; policies are tables over observations and memory; one model per run.
- **Flags**: none.

#### G.4 Small Decision Trees for MDPs with Deductive Synthesis (dtPaynt)

- **Citation**: R. Andriushchenko, M. Češka, S. Junges, F. Macák. CAV 2025. arXiv:2501.10126.
- **Link**: https://arxiv.org/abs/2501.10126
- **Venue / tier**: CAV, per the arXiv comment. **High**.
- **Depth**: Most. The introduction (which describes the whole loop), problem statement, experiments, related work and conclusion were read. The SMT encoding (Section 3) and the refinement details (Section 4) were skimmed.
- **Generated**: no LLM. A decision tree of bounded depth that maximises reachability probability or expected reward among all trees up to that depth. Predicates compare state variables with constants; leaves hold actions, including an action that picks uniformly at random.
- **Verifier**: Storm (through PAYNT) computes optimal policies on sub-MDPs. Z3 decides, through an SMT encoding in quantifier-free linear integer arithmetic, whether a policy can be represented by a tree of a given template.
- **Loop**: abstraction refinement.
  - Compute the optimal policy over a set of tree-representable policies; this over-approximates the set.
  - If it does not beat the best tree found so far, prune the set.
  - If the SMT check shows it is representable as a tree, keep it.
  - Otherwise it is spurious, and an unsatisfiable core guides a split into smaller sets ("harmonisation").
  - The search is anytime, and the best tree at depth k can seed depth k + 1.
- **Guarantee**: values of returned trees are exact (model checked). The tree is optimal within the depth bound only if the search completes; most runs hit the 20-minute timeout.
- **Evaluation**:
  - 21 models: 11 from OMDT (discounted rewards), 8 from QComp, 2 maze variants. Depths 1–8, 20-minute timeout.
  - **Against OMDT (MILP)**: similar on small models. On larger ones (up to 10k states and 175k choices) dtPaynt is much better; on some it gets within 1% of optimal while OMDT barely improves on the trivial tree.
  - **Against dtControl** (which maps a given optimal policy to a tree): trees 3–5× smaller on average at normalised values above 0.9, and up to 38× smaller on consensus (value above 0.93). Sometimes larger.
  - **Large MDPs** (up to about 1M states): it finds small trees when they exist, e.g., an optimal tree with no decision nodes on a 1M-state model.
  - **As a reducer of dtControl trees**: 46% fewer inner nodes on consensus and 90% fewer on csma (1.5M states), at normalised value 0.99.
- **Claimed vs demonstrated**: consistent. To let OMDT run, models are modified so that all actions are available everywhere; the authors note this makes the task harder.
- **Limitations**:
  - Authors: scalability suffers when deep trees are needed; counterexamples and a combination with dtControl are future work.
  - Observed: one fixed MDP per run; predicates are fixed comparisons of variables with constants.
- **Flags**: none.

#### G.5 Search-Based Synthesis of Probabilistic Models for Quality-of-Service Software Engineering (EvoChecker)

- **Citation**: S. Gerasimou, G. Tamburrelli, R. Calinescu. ASE 2015. Extended as "Synthesis of probabilistic models for quality-of-service software engineering", Automated Software Engineering 25(4):785–831, 2018.
- **Link**: https://eprints.whiterose.ac.uk/93144/1/ase.pdf (ASE 2015 accepted version)
- **Venue / tier**: ASE (a top SE venue, outside the list). **Medium**.
- **Depth**: Partial. The introduction, approach overview, evaluation design and result headings were read; formal definitions and detailed tables were skimmed.
- **Generated**: no LLM. Candidate probabilistic models (architectures plus parameter values) from a "probabilistic model template", a PRISM extension with evolvable parameters and modules. Candidates come from multi-objective genetic algorithms (NSGA-II, SPEA2, MOCell).
- **Verifier**: PRISM evaluates each candidate against QoS requirements (PCTL, CSL and reward properties; constraints and objectives), and the values are the fitness.
- **Loop**: a generational genetic algorithm. The feedback is numeric QoS values and constraint violations, used for selection; no counterexamples. 10,000 evaluations per run, 30 runs.
- **Guarantee**: each returned model's QoS values are exactly verified; the Pareto set is approximate.
- **Evaluation**: DPM and foreign-exchange trading, six variants, design spaces of 2·10¹⁴ to 7.22·10⁸⁶ candidates. The genetic algorithms significantly beat random search on the ε, hypervolume and IGD indicators. The approach also supports reconfiguring self-adaptive systems at runtime; the journal version has three case studies.
- **Claimed vs demonstrated**: consistent.
- **Limitations**: observed: the template, and so the design space, is human-written.
- **Flags**: none.

#### G.6 Counterexample-Guided Strategy Improvement for POMDPs Using Recurrent Neural Networks

- **Citation**: S. Carr, N. Jansen, R. Wimmer, A. Serban, B. Becker, U. Topcu. IJCAI 2019, pp. 5532–5539. arXiv:1903.08428. (Follow-up: JAIR 2021, "Task-aware verifiable RNN-based policies for POMDPs".)
- **Link**: https://www.ijcai.org/proceedings/2019/768
- **Venue / tier**: IJCAI main track. **High**.
- **Depth**: Partial. Sections 1–3 and the experimental setup were read; the result tables were skimmed.
- **Generated**: no LLM. An LSTM strategy trained on trajectories from the optimal strategy of the underlying fully observable MDP. It is then turned into a finite-memory, observation-based controller (with a predefined memory update).
- **Verifier**: PRISM (LTL) or Storm (expected rewards) checks the Markov chain induced by the extracted controller, exactly.
- **Loop**:
  - Verify the extracted controller.
  - Extract a counterexample: the "critical decisions", actions that may lead to critical states given the threshold.
  - Redistribute those choices per observation class with an LP-like step.
  - Generate new paths from the critical states and retrain the RNN.
  - 10 iterations in the experiments, or stop when progress is small.
- **Guarantee**: each extracted controller is verified exactly; the network itself is not. Incomplete (the problem is undecidable).
- **Evaluation**: gridworld navigation with moving obstacles (up to about 475k states), delivery and slippery delivery, and Maze, Grid and RockSample, against PRISM's POMDP solver and pomdpSolve. Up to three orders of magnitude faster, or on larger models. For large environments, training data is sampled from smaller environments with the same observation and action spaces, and the extracted controller is verified on the large instance.
- **Claimed vs demonstrated**: consistent with the tables, as far as skimmed.
- **Limitations**: observed: the controller is a table over observations and memory; the memory update is fixed in advance.
- **Flags**: none.

#### G.7 Abstract-level entries

| Paper | Venue / tier | Depth | What the abstract says |
|---|---|---|---|
| Batz, Biskup, Katoen, Winkler. Programmatic Strategy Synthesis: Resolving Nondeterminism in Probabilistic Programs. arXiv:2311.06889 | POPL 2024 (PACMPL 8, per dtPaynt's references) · Medium (outside the list) | Abstract | Resolves nondeterminism in probabilistic programs with strategies written as programs, deductively. Per dtPaynt's related work, this links program-level strategy construction to policies of possibly infinite MDPs. |
| Chakraborty, Majumdar, Mathew, Mukherjee, Raskin. Synthesizing POMDP Policies: Sampling Meets Model-checking via Learning. arXiv:2605.14440 | CAV 2026 · High | Abstract | L* automaton learning with sampling as the membership oracle and model checking as the equivalence oracle. Gives finite-state controllers with guarantees if the sampled policy is regular; relatively complete; threshold safety problems. |
| Bork, Chakraborty, Grover, Křetínský, Mohr. Learning Explainable and Better Performing Representations of POMDP Strategies. arXiv:2401.07656 | TACAS 2024 · High | Abstract | Learns small automata from an existing POMDP strategy with L*. The automata are smaller and more interpretable, can perform better, and scale better than solving the POMDP for automata directly. |
| Andriushchenko, Češka, Chakraborty, Junges, Křetínský, Macák. Symbiotic local search for small decision tree policies in MDPs | UAI 2025 · Medium | Title only (from dtPaynt's references) | The follow-up that dtPaynt points to for reducing large trees. Not read. |
| Zhu, Xiong, Magill, Jagannathan. An Inductive Synthesis Framework for Verifiable Reinforcement Learning. arXiv:1907.07273 | PLDI 2019 · Medium (outside the list) | Abstract | CEGIS synthesises deterministic programs, with inductive invariants, that approximate a neural policy; they act as a shield at deployment. Cyber-physical benchmarks. |
| Bastani, Pu, Solar-Lezama. Verifiable Reinforcement Learning via Policy Extraction (VIPER). arXiv:1805.08328 | NeurIPS 2018 · High | Abstract | Distils decision-tree policies from a neural network and its Q-function; the trees are verifiable (Pong variants, cart-pole). |
| Inala, Bastani, Tavares, Solar-Lezama. Synthesizing Programmatic Policies that Inductively Generalize | ICLR 2020 · High | Snippet and abstract | State-machine programmatic policies learned by adaptive teaching. They generalise to instances that need arbitrarily many repetitions, where neural policies fail. Empirical. |
| Bacci, Parker. Verified Probabilistic Policies for Deep Reinforcement Learning. arXiv:2201.03698 | NFM 2022 · Medium | Abstract | Interval-MDP abstractions of probabilistic deep-RL policies, using abstract interpretation, MILP, refinement and probabilistic model checking. |
| Gross, Jansen, Junges, Pérez. COOL-MC: A Comprehensive Tool for Reinforcement Learning and Model Checking. arXiv:2209.07133 | SETTA 2022 · Medium | Abstract | Connects Gym-style RL with Storm, builds policy-induced models incrementally, and bounds permissive policies. The base of A.1 and A.2. |
| Češka, Dehnert, Jansen, Junges, Katoen. Model Repair Revamped: On the Automated Synthesis of Markov Chains. arXiv:2105.13411 | No venue listed on arXiv · Low | Abstract | Outlines CEGAR (partition the design space and refine it from verification results) and CEGIS (critical subsystems as counterexamples that prune every program behaving the same way) for synthesising probabilistic models and programs. Applications: sketching, POMDP controllers, software product lines. |
| Evangelidis, Vázquez, Gerasimou. Accelerating Policy Synthesis in Large-Scale MDPs via Hierarchical Adaptive Refinement. arXiv:2506.17792 | FSE 2026 (PACMSE, per arXiv) · High | Abstract | Refines an MDP region by region, picking the most fragile regions first; the composed policy is near-optimal with a bounded error; MDPs up to 1M states, up to 2× faster than PRISM. |
| Gangopadhyay, Dasgupta. Counterexample Guided RL Policy Refinement Using Bayesian Optimization | NeurIPS 2021 · High | Snippet; also described in I.4 | Searches for counterexample trajectories of a trained RL policy by Bayesian optimisation, then repairs the policy at the failure points with gradient updates. |
| Zhang, Wu, Lin. Counterexample-guided Abstraction Refinement for POMDPs. arXiv:1701.06209 | Preprint · Low | Abstract | Abstracts POMDPs with a simulation relation that preserves a PCTL fragment, and refines the abstraction iteratively from verification counterexamples. |

### H. Permissive strategies, compact strategy representations, shields

A permissive strategy allows a set of actions per state; a shield blocks unsafe actions at runtime. This cluster also covers compact representations of a strategy.

#### H.1 Permissive Controller Synthesis for Probabilistic Systems

- **Citation**: K. Dräger, V. Forejt, M. Kwiatkowska, D. Parker, M. Ujma. Logical Methods in Computer Science 11(2:16), 2015 (extended version of a TACAS 2014 paper). arXiv:1504.04662.
- **Link**: https://arxiv.org/abs/1504.04662
- **Venue / tier**: journal, extending a TACAS paper. **Medium**.
- **Depth**: Full main text (Sections 1–6); the proofs in the appendix were skimmed. Seed paper (the two copies supplied are identical).
- **Generated**: no LLM and no loop. A memoryless multi-strategy for a turn-based stochastic two-player game: a set of allowed actions per controller state, deterministic or randomised over sets. It is maximally permissive with respect to a penalty scheme:
  - **static**: the sum of the penalties of the disallowed actions;
  - **dynamic**: the worst-case expected penalty.
- **Verifier / solver**: MILP encodings (CPLEX, Gurobi) inside PRISM-games. Exact for deterministic multi-strategies; an approximation scheme with a granularity parameter for randomised ones.
- **Guarantee**: soundness. Every memoryless controller strategy that complies with the multi-strategy satisfies the property against every environment strategy. Properties are bounds on expected total reward; probabilistic reachability reduces to these. Complexity:
  - NP-hard in every variant; in NP for deterministic multi-strategies and in PSPACE for randomised ones, with square-root-sum hardness for the randomised case;
  - randomised multi-strategies can be strictly more permissive, but an optimal randomised one may not exist under static penalties.
- **Evaluation**: six case studies (cloud, android, mdsm, investor, team-form, cdmsn), up to about 97k states and 9k controller states. Synthesis takes seconds to about 250 s. Randomised approximations run within a 5-minute timeout; some are not optimal.
- **Claimed vs demonstrated**: consistent. Scalability is limited by the MILP, which needs an integer variable per controller state-action pair.
- **Limitations**:
  - Authors: only expected total reward and reachability (richer temporal logics are future work); memoryless multi-strategies only; MILP cost grows with the number of actions per state.
  - Observed: the output is a per-state table of action sets for one fixed model.
- **Note** (from Marsha's email): once the model tracks which goals are done, goal ordering becomes a reachability property, so ordering requirements fall within this method's formal scope in such a model.
- **Flags**: none.

#### H.2 dtControl 2.0: Explainable Strategy Representation via Decision Tree Learning Steered by Experts

- **Citation**: P. Ashok, M. Jackermeier, J. Křetínský, C. Weinhuber, M. Weininger, M. Yadav. TACAS 2021, LNCS 12652, pp. 326–345. arXiv:2101.07202.
- **Link**: https://arxiv.org/abs/2101.07202 (artifact: doi 10.5281/zenodo.4437169)
- **Venue / tier**: TACAS. **High**.
- **Depth**: Full main text. The appendices referenced in the text were not in the arXiv version I had. Seed paper.
- **Generated**: no LLM. A decision tree representing an existing memoryless controller, which may be permissive. Predicates are axis-aligned, linear, categorical, or algebraic templates supplied by a domain expert, with fitted coefficients. An interactive interface lets the user choose predicates and retrain subtrees.
- **Verifier**: none in the loop; correctness is by construction. The tree represents the controller exactly, or a safe determinised subset of it. Safety is preserved; for reachability, determinisation can break the guarantee (the paper's Remark 1). Input controllers come from SCOTS, Uppaal Stratego, PRISM or Storm.
- **Loop**: no automatic loop; greedy top-down tree induction, with a human suggesting predicates and retraining.
- **Guarantee**: an exact representation of the input controller, or a determinised subset that preserves safety but may not preserve reachability.
- **Evaluation**:
  - Seven cyber-physical controllers from SCOTS and Uppaal (up to 16.6M states), compared with BDDs and dtControl 1.0. For example, cruise control (295k states) shrinks to 3 nodes when determinised; traffic (16.6M states) to 97 nodes.
  - 19 strategies for QComp and PRISM benchmark MDPs, compared with BDDs: the tree is smallest on 13 of 19, the BDD on 3, and neither compresses on 2 (e.g., pnueli-zuck, 303k states: 150k tree nodes against 5.7k BDD nodes).
  - Motivating example: expert algebraic predicates give an 11-node optimal permissive cruise-control tree, against 987 nodes without them.
- **Claimed vs demonstrated**:
  - The sizes are demonstrated.
  - Explainability is argued by example; there is no user study.
  - The gain from algebraic predicates is shown on the motivating example only.
- **Limitations**:
  - Authors: predicate synthesis from domain knowledge is manual or semi-automatic; Storm exports deterministic strategies, so determinisation gains nothing there; trees blow up when there are many distinct actions.
  - Observed: post-hoc compression of a strategy that must first be synthesised on the full, single model; nothing transfers across instances.
- **Flags**: none.

#### H.3 Abstract-level entries

| Paper | Venue / tier | Depth | What the abstract says |
|---|---|---|---|
| Vos, Verwer. Optimal Decision Tree Policies for Markov Decision Processes (OMDT). arXiv:2301.13185 | IJCAI 2023 · High | Abstract | An MILP maximises expected discounted return under a tree-size limit. Small trees learned by imitation are suboptimal; depth-3 optimal trees are often near-optimal. |
| Alshiekh, Bloem, Ehlers, Könighofer, Niekum, Topcu. Safe Reinforcement Learning via Shielding. arXiv:1708.08611 | AAAI 2018 · High | Snippet | A shield synthesised from an LTL safety specification and an abstraction of the environment overrides unsafe actions during learning and execution. |
| Jansen, Könighofer, Junges, Serban, Bloem. Safe Reinforcement Learning via Probabilistic Shields. arXiv:1807.06096 | CONCUR 2020 (venue from memory, not confirmed on the arXiv page) · Medium if so | Abstract | Builds a shield by model checking the probabilities of critical decisions in a safety-relevant fragment of an MDP; RL optimises within the shield. PAC-MAN and a service robot. |
| Andriushchenko, Češka, Junges. Shields to Guarantee Probabilistic Safety in MDPs. arXiv:2605.10888 | CAV 2026 · High | Abstract | Extends shields to probabilistic safety requirements. Shows that strong safety and full permissiveness cannot both hold in that setting, proposes natural shields with weaker guarantees, and gives offline and online constructions with strong safety; experiments suggest they are practical. |
| Yang, Marra, Rens, De Raedt. Safe Reinforcement Learning via Probabilistic Logic Shields (PLPG). arXiv:2303.03226 | IJCAI 2023 (venue from memory, not confirmed) · High if so | Abstract | Models safety constraints as differentiable functions with probabilistic logic programming, usable with any policy-gradient algorithm while keeping its convergence guarantees. |
| Vo Huynh, Parker, Feng. Optimization-Based Robust Permissive Synthesis for Interval MDPs. arXiv:2510.03481 | Preprint · Low | Abstract | Robust permissive multi-strategies for interval MDPs via MILP; scales to hundreds of thousands of states. |
| Wu, Zhang, Lin. Permissive Supervisor Synthesis for Markov Decision Processes through Learning. arXiv:1703.07351 | Preprint · Low | Abstract | Learns permissive local supervisors for MDPs iteratively from counterexamples of compositional (assume-guarantee) probabilistic model checking; terminates in finitely many steps and is correct. |
| Kinská, Křetínský, Meggendorfer, Rieder, Weininger. dtControl2+ε: Trading Optimality for Explainability in MDPs via Decision Trees. arXiv:2607.25925 | FMCAD 2026 (per arXiv) · Medium | Abstract (cites I.3) | Given an allowed imprecision ε, builds a much smaller decision tree for an MDP controller that is still guaranteed ε-optimal. |
| Rieder, Pranger, Chakraborty, Křetínský, Könighofer. Explainably Safe Reinforcement Learning. arXiv:2606.04634 | NeurIPS (per Semantic Scholar) · High if so | Abstract (cites I.3) | Represents a shield as a hierarchy of decision trees: a design-time tree classifies states into risk categories, and runtime trees explain which actions are allowed. |

### I. Generalising policies to held-out and larger instances

How a policy is carried from small instances to larger ones, and how to measure, with statistical guarantees, whether it generalises.

#### I.1 Per-Domain Generalizing Policies: On Validation Instances and Scaling Behavior

- **Citation**: T. P. Gros, N. J. Müller, D. Fišer, I. Valera, V. Wolf, J. Hoffmann. ICAPS 2025, 35(1):198–203 (short paper). arXiv:2505.00439.
- **Link**: https://ojs.aaai.org/index.php/ICAPS/article/view/36118
- **Venue / tier**: ICAPS, short paper. **High**.
- **Depth**: Full, including appendices.
- **Generated**: GNN value-function policies (the Ståhlberg et al. architecture); no LLM. The contribution is evaluation methodology.
- **"Verifier"**: statistical.
  - **Dynamic validation** selects policies: 10 instances are generated per size, increasing beyond the training sizes until coverage drops below 30%.
  - **Scaling evaluation**: for each instance size, instances are sampled uniformly over the generator's inputs (sizes encoded as a constraint problem). Sampling continues until a sequential Student-t (Chow–Robbins) interval bounds the estimated coverage within ε = 0.05 of the true value with confidence 1 − κ = 0.9. Larger sizes are tested until coverage stays below a threshold for several consecutive sizes.
- **Loop**: none (policy selection and evaluation).
- **Guarantee**: per-size statistical coverage estimates with confidence bounds; not per-instance proofs; i.i.d. over the generator's distribution.
- **Evaluation**: 9 IPC 2023 domains, 3 seeds. Dynamic validation gives the best scaling and summed coverage in all 9.
- **Claimed vs demonstrated**: consistent. The paper also shows that fixed IPC test sets hide scaling behaviour.
- **Limitations**: observed: deterministic domains; coverage, not quality.
- **Flags**: statistical guarantee.

#### I.2 Toward a General Framework for Evaluating Per-Domain Generalization Using LLMs, Theorem Provers, and Statistical Model Checking

- **Citation**: Müller, Rudolph, Taitler, Gros. ICAPS 2026 Workshop LM4Plan (8 pages).
- **Link**: https://icaps26.icaps-conference.org/files/workshops/lm4plan/12_Toward_a_General_Framework_-1.pdf
- **Venue / tier**: workshop at a listed venue. **Low–Medium**.
- **Depth**: Partial. The abstract, introduction and method overview were read; the experiments were skimmed by search within the PDF, given the paper's preliminary status.
- **Generated**: the LLM modifies an existing PDDL instance generator so that it produces only instances satisfying user-written first-order constraints, giving targeted instance distributions.
- **Verifier**: a theorem prover (SMT over uninterpreted predicates) checks the generated instances. Failures (unsatisfiable cores) go back to the LLM. The resulting generators plug into a Gymnasium-style framework that uses statistical model checking (PyDSMC) to estimate how a generalised policy performs per instance size, with confidence intervals.
- **Loop**: generate the generator, check instances, feed unsat cores back, repeat.
- **Guarantee**: the instances meet the constraints (SMT-checked); policy performance comes with statistical confidence intervals.
- **Evaluation**: preliminary. Transport and two other domains; GNN policies and Stein et al.'s programs are evaluated on the constrained instances.
- **Claimed vs demonstrated**: framed as a framework with preliminary experiments.
- **Limitations**: observed: deterministic domains; early-stage results.
- **Flags**: preliminary.

#### I.3 1–2–3–Go! Policy Synthesis for Parameterized Markov Decision Processes via Decision-Tree Learning and Generalization

- **Citation**: M. Azeem, D. Chakraborty, S. Kanav, J. Křetínský, M. Mohagheghi, S. Mohr, M. Weininger. VMCAI 2025, LNCS (Springer), pp. 97–120, doi 10.1007/978-3-031-82703-7_5. arXiv:2410.18293.
- **Link**: https://arxiv.org/abs/2410.18293 (artifact: github.com/muqsit-azeem/dtstrat-123go-artifact)
- **Venue / tier**: VMCAI, a peer-reviewed conference co-located with POPL, outside the list. **Medium**.
- **Depth**: Most. The abstract, introduction, related work, method (Section 3), evaluation setup, discussion and conclusion were read, and the results tables in part. Preliminaries were skimmed; the appendices (base instances, parameter values, timings) were not read.
- **Generated**: no LLM. A decision tree over the PRISM model's state variables.
  - Predicates are axis-aligned (`x > c`), chosen by Gini index, and the tree is grown until no split is possible.
  - It is learned from the optimal actions that Storm computes on a few small "base" instances of a parameterized MDP; one or two instances often suffice.
  - Actions are identified by PRISM command labels, made unique so that they mean the same thing in every instance.
  - States may carry several optimal actions (permissive data); each leaf takes a majority vote.
  - The tree is then applied to instances of any size.
- **Verifier**:
  - Storm computes the optimal policies of the base instances.
  - On a large instance, the value of the Markov chain the tree induces is computed exactly when that chain can still be built. It is then a guaranteed lower bound on the optimum (upper bound when minimising).
  - Otherwise statistical model checking estimates it (4 cases).
- **Loop**: none. Learn once, then apply to every larger instance. A follow-up workshop paper (I.4) proposes counterexample-guided refinement.
- **Guarantee**: on each instance whose induced chain can be analysed, the policy's value is certified, exactly or with statistical confidence. Nothing guarantees closeness to the optimum on instances too large to solve. The authors call the generalisation step unreliable reasoning, and the resulting policies provably good enough. If the tree picks an action that is not enabled, one is chosen uniformly at random; this never happened in the experiments.
- **Evaluation**:
  - Benchmarks: parameterized MDPs with reachability objectives from the Quantitative Verification Benchmark Set, plus Mars Exploration Rovers; 21 model–property combinations. Parameters are scaled until Storm can no longer solve the instance within an hour.
  - Baselines: Storm's optimum where solvable; statistical model checking with MODES using smart lightweight scheduler sampling, the uniform policy, and 1000 random deterministic policies.
  - Near-optimal in 13 of 21 cases; better than smart scheduler sampling in 2 of the other 8; generally better than random and uniform policies. Random policies do better in 2 instances.
  - Policy values often stay stable as parameters grow far beyond what Storm can solve.
- **Claimed vs demonstrated**: near-optimality beyond solvable sizes is extrapolated from smaller instances and from how stable the values are; the per-instance values themselves are computed.
- **Limitations**:
  - Authors: it fails when the optimal choice depends on state variables absent from the base instances (csma's property about all stations, where added modules bring new variables), on horizon-like parameters (Pac-Man's step bound), or on parameters that change only probabilities (Mars Exploration Rovers). Base instances are chosen heuristically. No optimality guarantee.
  - Observed: one objective (reachability). Predicates compare raw state variables with constants, so the tree transfers only where decisions depend on thresholds that do not move with the parameter.
- **Flags**: generalisation beyond solvable sizes empirical (per-instance values computed).

#### I.4 Counterexample-Guided Policy Improvement for Parameterized Markov Decision Processes

- **Citation**: M. Azeem, D. Chakraborty, K. Grover, S. Kanav, J. Křetínský. AAAI 2025 Workshop on Generalized Planning (GenPlan).
- **Link**: https://aair-lab.github.io/genplan25/papers/30.pdf
- **Venue / tier**: workshop at a listed venue. **Low–Medium**.
- **Depth**: Full, a short paper.
- **Generated**: no LLM. Starts from the 1–2–3–Go! tree (I.3) as the initial policy for a large instance of a parameterized MDP.
- **Verifier**: qualitative properties, such as almost-sure reachability. A counterexample is a simulated trace that violates the property (a loop, no progress, a bad state); for qualitative properties one trace is enough.
- **Loop** (proposed): add data from failed traces and local simulations, update the tree while keeping earlier behaviour, re-test on larger instances, and repeat.
- **Guarantee**: aims at the qualitative property on the instance under consideration; nothing is demonstrated.
- **Evaluation**: none reported; ongoing work.
- **Claimed vs demonstrated**: a proposal only.
- **Flags**: no results.

#### I.5 Abstract-level entries: generalized policies for probabilistic planning (no LLM)

| Paper | Venue / tier | Depth | What the abstract says |
|---|---|---|---|
| Yoon, Fern, Givan. Inductive Policy Selection for First-Order MDPs. arXiv:1301.0614 | UAI 2002 · Medium | Abstract | Learns first-order policies, as ensembles of decision lists in a taxonomic concept language, from solutions of small instances computed by PGraphplan. The policies generalise as the number of objects grows. Extends Martín and Geffner's generalized policies to stochastic domains. |
| Fern, Yoon, Givan. Approximate Policy Iteration with a Policy Language Bias: Solving Relational Markov Decision Processes. arXiv:1109.2156 | JAIR 25 (2006); conference version NIPS 2003 · Medium (High for the NIPS paper) | Abstract | Approximate policy iteration that learns in policy space, using a relational policy language, bootstrapped by random walks because goal rewards are sparse. Finds good policies for several classical planning domains and their stochastic variants. |
| Toyer, Trevizan, Thiébaux, Xie. ASNets: Deep Learning for Generalised Planning. arXiv:1908.01362 | JAIR 68 (2020), extending AAAI 2018 · Medium (High for the AAAI paper) | Abstract | Neural generalised policies for probabilistic and classical (P)PDDL planning. Weights are shared across a domain and trained by imitating a planner on a handful of small problems, then applied to much larger ones; seven domains. Sparsity-inducing regularisation makes some networks readable. |
| Garg, Bajpai, Mausam. Size Independent Neural Transfer for RDDL Planning. arXiv:1902.03081 | ICAPS 2019 · High | Abstract | Transfers neural policies across RDDL problems of different sizes (SysAdmin, Game of Life). |
| Garg, Bajpai, Mausam. Symbolic Network: Generalized Neural Policies for Relational MDPs (SymNet). arXiv:2002.07375 | ICML 2020 · High | Abstract | Neural generalized policies for RDDL domains, applied to new instances without retraining; nine IPPC domains. Better than random, and sometimes better than training a deep reactive policy from scratch. Notes that early first-order generalized policies had limited success. |

None of these verifies the generalized policy formally; they are evaluated by simulation on test instances.

### J. Self-adaptive systems (SEAMS)

Context for a SEAMS submission: LLMs as adaptation managers, and earlier PRISM-in-the-loop self-adaptation.

#### J.1 Vibe-Coding: Feedback-Based Automated Verification with no Human Code Inspection, a Feasibility Study

- **Citation**: M. Töpfer, F. Plášil, T. Bureš, P. Hnětynka. arXiv:2604.14867, Apr 2026 (companion: arXiv:2602.18607).
- **Link**: https://arxiv.org/abs/2604.14867
- **Venue / tier**: preprint, short. **Low**.
- **Depth**: Full.
- **Generated**: an LLM writes an adaptation manager in Python for a collective adaptive system: it resolves ensembles at each step. The prompt template gives the interface contract, domain rules, strategy intent and a summary of the constraints. The model is not named in the text I read.
- **Verifier**: runtime monitoring of test executions, with varied initial states and seeds. Generic architectural constraints and functional constraints are written in FCL, a first-order temporal logic over finite traces with windowed counting operators. Deterministic per trace, empirical over the test suite.
- **Loop**: a report of violated constraints (formula, step window, witness agents) is appended to the prompt; up to 10 iterations; 10 independent attempts per feedback variant.
- **Guarantee**: none formal. The authors present feasibility evidence; results are only as good as the constraints and the coverage of the runs.
- **Evaluation**: the feedback granularity is varied: metrics only (win/loss, hit points), generic constraints only, or full constraint feedback. Full feedback converges within a few iterations; metrics-only feedback often stalls and oscillates, overfitting to the metric.
- **Claimed vs demonstrated**: one synthetic case study (Dragon Hunt), 10 attempts per variant.
- **Limitations**: observed: no resampling baseline at equal budget; one case study.
- **Flags**: guarantee empirical only; no equal-budget resampling baseline.

#### J.2 Abstract-level entries and background

| Paper | Venue / tier | Depth | What is known |
|---|---|---|---|
| Li, Zhang, Li, Weyns, Jin, Tei. Exploring the Potential of Large Language Models in Self-adaptive Systems. arXiv:2401.07534 | SEAMS 2024 · Medium | Abstract | A literature classification; at the time, LLM work at SEAMS and TAAS was scarce. |
| Li, Zhang, Li, Weyns, Jin, Tei. Generative AI for Self-Adaptive Systems: State of the Art and Research Roadmap. arXiv:2512.04680 | ACM TAAS · Medium | Abstract | Benefits of generative AI for the MAPE-K functions and for keeping humans on the loop. The abstract does not mention formal or probabilistic verification of generated plans. |
| Grammar-Constrained Refinement of Safety Operational Rules Using Language in the Loop: What Could Go Wrong. arXiv:2604.23523 | SEAMS 2026 short paper · Medium | Abstract | An LLM refines safety operating rules under a grammar, using counterfactual simulation; the models tend to over-tighten bounds. |
| Maia, Vieira, Barros De Oliveira et al. MAPER: Extending MAPE-K with LLM-Based Reasoning to Manage Unanticipated Situations in Self-Adaptive Systems | SEAMS 2026 full paper · Medium | Title and authors only (from the programme) | No abstract or text found (see [Roadblocks](#6-roadblocks)). |
| Benecchi, Cardone, Camilli, Lestingi, Mirandola. Verify, Augment, Improve: Self-Adaptation Repair via Automated Knowledge Augmentation from Mistakes. https://publikationen.bibliothek.kit.edu/1000196013 | SEAMS 2026 full paper, ACM SIGSOFT Distinguished Paper · Medium | Abstract (open access on KITopen) | No LLM in the abstract. Extends MAPE-K with an asynchronous loop: each ineffective adaptation is checked against a ground truth (e.g., a high-fidelity simulator); when the predictive surrogate drifts, the failure is explained, training data is augmented near the drift, and the surrogate is retrained. A human–machine teaming benchmark and two subjects; average gains of 6.89% and 10.88%. |
| Vázquez, Evangelidis, Shahbeigi, Calinescu, Gerasimou. Mind the Prompt: Self-adaptive Generation of Task Plan Explanations via LLMs (COMPASS). arXiv:2604.21092 | SEAMS 2026 full paper · Medium | Abstract | An LLM writes explanations of automated task plans. A POMDP over the user's hidden cognitive states (attention, comprehension) and observable cues yields a synthesised policy that decides when to adapt explanations and refine prompts. Two cyber-physical case studies. The probabilistic model drives prompt adaptation; it does not verify LLM output. |
| Other SEAMS 2026 research-track papers with LLMs or agents (titles from the programme, not read): Leveraging Low-Parameter LLMs for Self-Healing in Kubernetes-Based Container Orchestration (best student paper runner-up); Dynamic Agent Generation for Self-Adaptive Root Cause Analysis; Software Self-Extension with SelfEvolve (short); CALM, QoS-aware routing in small-language-model systems; ASTL, an adaptive serving stack for LLMs (short) | SEAMS 2026 · Medium | Title only | Listed for completeness; none of the titles mentions formal verification. |
| Kara et al. LLM-in-the-loop MAPE-K versus agentic end-to-end adaptation (KIT, 2026) | Report · Low | Snippet | Compares the LLM inside a MAPE-K loop with an end-to-end agentic design. |
| Nascimento, Alencar, Cowan. Self-Adaptive Large Language Model (LLM)-Based Multiagent Systems. arXiv:2307.06187 | ACSOS 2023 workshop (per search) · Low | Snippet | LLM-based multi-agent systems organised around MAPE-K. |
| POLARIS: Is Multi-Agentic Reasoning the Next Wave in Engineering Self-Adaptive Systems? arXiv:2512.04702 | SEAMS 2026 short paper (per the programme) · Medium | Title only | Not read. |
| Vázquez, Evangelidis, Shahbeigi, Gerasimou. Adaptive Human-Robot Collaborative Missions using Hybrid Task Planning. arXiv:2504.06746 | Preprint · Low | Abstract | No LLM. First finds a feasible plan, then adds uncertainty and verifies it, yielding Pareto-optimal plans; adaptation tactics handle changing requirements and capabilities; an industrial vineyard case study. |
| Background, pre-LLM PRISM-in-the-loop adaptation: Calinescu et al., "Self-adaptive software needs quantitative verification at runtime" (CACM 2012); Casimiro et al., "A Probabilistic Model Checking Approach to Self-adapting Machine Learning Systems" (2021; venue not confirmed); Pandey et al., hybrid planning (ACSOS 2020) | Various | Not re-read | Runtime quantitative verification with PRISM drives adaptation decisions; EvoChecker ([G.5](#g5-search-based-synthesis-of-probabilistic-models-for-quality-of-service-software-engineering-evochecker)) also supports runtime reconfiguration. |

The SEAMS 2026 research-track programme was checked title by title; it has nine papers involving LLMs or agents. None of them, and no SEAMS 2025 paper found by search, combines LLM-generated adaptation policies with PRISM or PCTL verification. SEAMS 2025 was covered by searches only, not by its programme.

### K. Reusing solved instances or experience in context

Reusing solved instances, or past experience, as in-context examples. All entries were read at abstract, snippet or title level only.

| Paper | Venue / tier | Depth | What is known |
|---|---|---|---|
| Kumar, Cohen. Localizing and Correcting Errors for LLM-based Planners (L-ICL). arXiv:2602.00276 | Preprint · Low | Abstract | Iteratively adds, for the first constraint violation in a plan, a minimal input-output demonstration of the correct step. On an 8×8 gridworld, 89% valid plans with 60 training examples against 59% for the best baseline; also mazes, Sokoban and Blocksworld, across several LLMs. Violations are detected against the domain constraints; the abstract does not describe any formal verification. |
| Sohrabi et al. HCL-GP (see [B.8](#b8-abstract-level-entries)) | Preprint · Low | Abstract | A library of reusable policy components, mined from successful runs and retrieved by semantic search. |
| Brooks, Walls, Lewis, Singh. Large Language Models can Implement Policy Iteration (ICPI) | NeurIPS 2023 · High | Snippet | Policy iteration carried out in the prompt rather than in the weights, using in-context rollouts; six small RL tasks. |
| Song et al. LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models. arXiv:2212.04088 | ICCV 2023 (venue from memory) · Medium | Snippet | Retrieves in-context examples by embedding similarity to the new instruction (top K from a store). |
| Zhao et al. ExpeL: LLM Agents Are Experiential Learners | AAAI 2024 · High | Title only | Not read. |
| LifeMem: Enabling Lifelong Experience Reuse for LLM Agents. arXiv:2609.12655 | Preprint · Low | Snippet | Per a search summary, retrieving raw past trajectories surfaced misleading examples when the task wording shifted. |
| Sarukkai, Xie, Fatahalian. Self-Generated In-Context Examples Improve LLM Agents for Sequential Decision-Making Tasks. arXiv:2505.00234 | Preprint · Low | Abstract | Builds a database of the agent's own successful trajectories and uses them as in-context examples for later tasks. Simply accumulating successes lifts ALFWorld from 73% to 89%, and curation reaches 93%. Success is judged by task outcome, not by formal verification. |

Several entries above put worked examples in the prompt, though not solved instances of the same family: a Gripper program ([B.6](#b6-language-models-for-generalised-pddl-planning-synthesising-sound-and-programmatic-policies-lmplan)), three example domains ([D.2](#d2-large-language-models-as-planning-domain-generators)), and truncated training tasks ([B.1](#b1-generalized-planning-in-pddl-domains-with-pretrained-large-language-models)). None of the reuse work found attaches formal certificates to the reused solutions.

### L. LLM-guided controller synthesis in control

Covers the control-theory side of the brief ("controllers with guarantees"). All entries were read at abstract level, and none has a confirmed peer-reviewed venue.

| Paper | Venue / tier | Depth | What the abstract says |
|---|---|---|---|
| Bayat, Abate, Ozay, Jungers. LLM-Enhanced Symbolic Control for Safety-Critical Applications. arXiv:2505.11077 | Preprint · Low | Abstract | A code agent translates a natural-language reach-avoid problem into the input of abstraction-based controller-design tools, and a checker agent verifies the code and flags mismatches with the specification. Guarantees come from the symbolic-control tool. |
| Li, Li, Xu, Chen. From Natural Language to Certified H-infinity Controllers: Integrating LLM Agents with LMI-Based Synthesis (S2C). arXiv:2511.07894 | Preprint · Low | Abstract | Five agent roles go from natural-language requirements to H∞ state-feedback controllers, certified through LMI synthesis, Monte Carlo tests and frequency-domain checks, with specification refinement. 14 COMPleib problems: 100% synthesis success, convergence within six iterations; robust across five LLM backbones. |
| Bosio, Mueller. Synthesizing Interpretable Control Policies through Large Language Model Guided Search. arXiv:2410.05406 | Conference paper, venue not confirmed · Low | Abstract | Evolves Python control policies with an LLM, evaluated in simulation (pendulum swing-up, ball in cup). |
| Bosio, Guarrera, Sangiovanni-Vincentelli, Mueller. Combining Large Language Models and Gradient-Free Optimization for Automatic Control Policy Synthesis. arXiv:2510.00373 | Preprint · Low | Abstract | The LLM searches over program structure while numerical optimisation tunes the parameters; higher returns and sample efficiency than LLM-only search. |
| Wang, Li, Zhang, Ren. AI Control Scientist: LLM-driven Agentic System for Automated Control Design. arXiv:2608.26780 | Preprint · Low | Abstract | Agents for task modelling, controller structure and parameter tuning under closed-loop criteria; better design success and efficiency than automated baselines. |

### M. Domain resources and the project's own prior work

- **Hoekstra, Ellerbroek. BlueSky ATC Simulator Project: an Open Data and Open Source Approach.** ICRAT 2016. Seed PDF supplied by the user.
  - Depth: Full. Tier: Medium (domain conference).
  - Not a generate-and-verify method. It is an open-source Python air-traffic simulator with a scenario language (TrafScript), BADA 3 compatibility and open performance data. It has no verification, probabilistic model or controller synthesis. It is listed only as a possible domain; Marsha's email sets ATC aside for now.
- **UUV case-study source** (arXiv:2308.14663, the pipeline-inspection AUV used by this repository). Not read for this survey.
- **Predecessor: "LLM-Based Grid-World Path Planning With Probabilistic Model Checking."** LLMTrust workshop at FSE 2026 (FSE Companion), doi 10.1145/3803437.3806714.
  - Tier: Low–Medium (workshop at a listed venue).
  - Depth: not re-read; described from the project notes.
  - The LLM writes a per-state gridworld policy for each goal, and the policies are combined. PRISM checks the induced Markov chain against reachability, ordering and avoidance requirements. Feedback is a hard-coded natural-language rendering of each failed requirement. Per the Sept 24 notes, blind retries did almost as well as feedback.
  - Semantic Scholar lists it among the papers citing VeriPlan.

## 3. Coverage map

### 3.1 The fully or partly read papers at a glance

Abbreviations:
- **Prob.**: whether the verifier has probabilistic semantics, and how it is used.
- **CE**: counterexample.
- **PMC**: probabilistic model checking.
- **DL**: description logic.

| Entry | Generator | Artifact | Verifier | Prob. | Loop and feedback | Scope of guarantee | Tier · depth |
|---|---|---|---|---|---|---|---|
| A.1 Gross et al. 2025 | LLM, queried per state | the LLM's per-state choices | COOL-MC + Storm, PCTL reachability | exact | none | one seeded run, one model | Low · Full |
| A.2 Gross & Spieker 2024 | LLM | overrides for a DQN policy | COOL-MC + Storm | exact | one round; violating state-action pairs | value of the repaired policy, one model | Medium · Full |
| A.3 VeriPlan | LLM | linear plan, LTL rules, PRISM translation | PRISM, Stormpy | qualitative; soft rules sampled | 3 rounds; violated rules | translated plan against translated rules | Medium · Most |
| A.4 Saccon et al. | LLM writes a Prolog model | MDP (Storm synthesises the policy) | Storm, as synthesiser | exact, on the generated model | none; experts fix errors | policy optimal for the generated MDP | Medium · Most |
| A.5 Pro2Guard | LLM agent at runtime | actions and revised plans | PRISM on a learned Markov chain | PAC, on the learned model | every step; risk alert with probability | learned model, Markov assumption | Low · Full |
| B.1 Silver et al. | LLM | Python generalized plan | execution, VAL | no | up to 4 rounds; errors, VAL hints | training tasks; larger tasks tested | High · Full |
| B.2 Stein et al. (ICAPS) | 4 LLMs | pseudocode and Python | VAL on 6 tasks | no | strategy and code debugging; several initial programs | empirical | High · Most |
| B.3 Stein et al. (Lean) | LLM | Lean program and proofs | Lean kernel | no | plan and proof repair | all instances meeting a written specification | Low · Most |
| B.4 Drexler et al. | LLM agent | DL features and rules | exhaustive proof on training tasks | no | CE (open state, cycle); up to 329 rounds | training tasks proven; test empirical | Low–Medium · Full |
| B.5 GenePlan | LLM | Python planners | VAL; mean plan length | no | evolutionary; parents' scores and errors | empirical | High · Full |
| B.6 LMPlan | 3 LLMs | Python policies and heuristics | none (execution is sound by construction) | no | none; best of 10 samples | returned plans are valid | Low · Full |
| B.7 Pereira et al. | LLM | Python heuristic | DFS check of "directness" | no | up to 10 rounds; CE state with values | property on training tasks, time-capped | Low · Most |
| C.1 Stechly et al. | LLM | answers and plans | VAL, SymPy, edge check, or the LLM itself | no | up to 15 back-prompts, against resampling | from the verifier | High · Full |
| C.2 Olausson et al. | 3 LLMs | Python code | unit tests | no | one repair level, against i.i.d. sampling | tests | High · Full |
| C.3 LLM-Modulo | LLM | plans | hard and soft critics | no | back-prompting | from the hard critics | High (position) · Most |
| D.1 DPO-AF | LLM, fine-tuned | automaton controllers | NuSMV, LTL | no | rule counts used for DPO training | abstract model only | Medium · Partial |
| D.2 Oswald et al. | 7 LLMs | PDDL actions | VAL cross-validation of plans | no | none | heuristic equivalence | High · Partial |
| E.1 KnowNo | LLM | next-step choices | conformal calibration | statistical | none; asks a human | success ≥ 1 − ε, i.i.d. | Medium · Partial |
| F.1 Policies Grow on Trees | abstraction refinement | policy tree, tables at the leaves | Storm: games, quotient MDPs | exact | split on inconsistent indices | finite family; sound and complete | Medium · Full |
| F.2 SMPMC | CDCL(T) search | constrained, robust policies (e.g., trees) | Z3 with Storm as a theory | exact (to 1e-4) | conflicts and bounds | finite parameter space; complete | High · Most |
| F.3 rfPG | subgradient ascent | finite-state controller | PAYNT robust evaluation | exact | worst-case member, then gradient steps | exact robust value of the returned controller | High · Full |
| F.4 Schnitzer et al. | interval-MDP learning or meta-RL | memoryless policy | PRISM: learned interval MDPs + scenario theorem | PAC | none | unseen environments, i.i.d. | High · Most |
| G.1 PAYNT | CEGIS, abstraction refinement | hole assignment of a sketch | Storm + Z3 | exact | CEs; splitting | finite family; complete | High · Full |
| G.2 FSC synthesis (UAI) | inductive synthesis | finite-state controller | Storm + CVC5 | exact | splitting, CEs, adding memory | one POMDP; bounds on the family | Medium · Full |
| G.3 Oracle-guided synthesis (JAIR) | abstraction refinement and CEs | sketch, controller, Dec-POMDP controllers | Storm | exact | splitting score; CEs | design space; complete | Medium · Partial |
| G.4 dtPaynt | abstraction refinement + SMT | bounded-depth decision tree | Storm + Z3 | exact | splits guided by unsatisfiable cores | best tree within the depth, if the search completes | High · Most |
| G.5 EvoChecker | genetic algorithms | instance of a probabilistic model | PRISM | exact | fitness (QoS values) | each returned model | Medium · Partial |
| G.6 Carr et al. | RNN | extracted finite-state controller | PRISM, Storm | exact | critical decisions, then new data | per extracted controller; small-to-large transfer | High · Partial |
| H.1 Dräger et al. | MILP | permissive multi-strategy | PRISM-games | exact | none | all compliant strategies, one model | Medium · Full |
| H.2 dtControl 2.0 | decision-tree learning | tree for a given strategy | none (by construction) | inherited | human in the loop | exact representation | High · Full |
| I.1 Gros et al. | GNN | value-function policy | sequential confidence intervals | statistical | none | coverage per instance size | High · Full |
| I.2 Müller et al. | LLM edits instance generators | instance generators | SMT check; statistical model checking | statistical (evaluation) | unsatisfiable cores back to the LLM | constraints met; performance intervals | Low–Medium · Partial |
| I.3 1–2–3–Go! | decision-tree learning from the optimal policies of small instances | decision tree over state variables | Storm on the small instances; exact value of the induced chain on each large instance, or statistical model checking | exact or statistical | none | value certified per instance; near-optimality extrapolated | Medium · Most |
| I.4 Counterexample-guided improvement (GenPlan workshop) | decision tree refined from counterexamples (proposed) | decision tree | simulated traces that violate a qualitative property | qualitative | proposed loop; no results | — | Low–Medium · Full |
| J.1 Töpfer et al. | LLM | adaptation manager (Python) | FCL runtime monitoring | no | up to 10 rounds; violated constraints | empirical over runs | Low · Full |

### 3.2 Verifier against generator

| Verifier | LLM in an iterative loop | LLM, one shot or sample-and-select | No LLM: search or synthesis | No LLM: learned policy |
|---|---|---|---|---|
| Exact PMC on a given model | A.2 (one round), A.3 (qualitative) | A.1 (the LLM is the policy), A.4 (Storm solves an LLM-written model) | F.1, F.2, G.1–G.5, H.1, multi-environment algorithms (F.5), shields (H.3), model repair and hierarchical refinement (G.7) | F.3, G.6, I.3 (per large instance), COOL-MC and Bacci & Parker (G.7) |
| PMC on a learned or abstracted model | A.5, AgentGuard (A.6) | TraceToChain, SafetyDrift (A.6; Markov chains fitted to agent traces, analysed in closed form) | — | F.4 (learned interval MDPs) |
| Statistical: conformal, scenario, sequential intervals | I.2 (unsat-core loop; statistical evaluation) | E.1, E.2 | — | F.4, I.1, I.3 (largest instances) |
| Deterministic exhaustive check: model checking, SMT, proof over a finite set of tasks | B.4, B.7, Jha et al. (C.4), GLM2FSA and ESBMC-AI (D.3), Cui et al. (B.8); D.1 uses it to train | Ramani et al., joint verification and refinement (D.3) | — | — |
| Deductive proof for all instances | B.3 | — | Batz et al., Zhu et al. (G.7) | — |
| Execution, tests, plan validation, simulation | B.1, B.2, B.5, C.1–C.3, J.1, L-ICL (K), AutoToS (B.8), Eureka, FunSearch (D.3), control-policy search (L) | B.6, Corrêa et al. (B.8), D.2, Thought of Search (B.8) | — | generalized policies for probabilistic planning (I.5); I.4 (simulated counterexamples, proposed) |
| None; correct by construction | — | B.6's execution wrapper | H.2 | — |

### 3.3 Scope of the guarantee

| Scope | Entries |
|---|---|
| One instance, exact | A.1 (one seeded run), A.2, A.4 (on the generated model), G.1–G.5, H.1, H.2. A.3 is qualitative. |
| A finite family sharing a state space, exact | F.1, F.2, F.3 (value of the returned controller), multi-environment MDPs (F.5), KAPS (F.5) |
| Small to large instances, verified exactly on each | G.6 (trained on small environments, verified on the large one); I.3 (decision trees learned on small instances; value computed exactly on each large instance where the induced chain can be built, statistically otherwise) |
| Small to large instances, simulation only | I.5 (relational decision lists, ASNets, SymNet) |
| A distribution over parametric environments, PAC | F.4; Badings et al. 2022 (F.5; guarantees that some policy exists) |
| A distribution over tasks, conformal | E.1, E.2 |
| Instance sizes, statistical per size | I.1, I.2 |
| Training tasks proven, larger tasks tested | B.4, B.7 |
| All instances meeting a written specification, deductive (deterministic domains) | B.3 |
| Training tasks only, larger tasks tested | B.1, B.2, B.5; B.6 (valid plans only) |
| A learned model, PAC under a Markov assumption | A.5 |

### 3.4 What goes back to the generator

| Feedback content | LLM generators | Non-LLM generators |
|---|---|---|
| Counterexample states, paths or cycles | B.4 (open states, cycles), B.7 (state with heuristic values), Jha et al., GLM2FSA, CESS, ESBMC-AI | G.1, G.3 (critical sub-Markov chains), G.6 (critical decisions), I.4 (violating traces; proposed) |
| Violated rules or properties | A.2 (violating state-action pairs), A.3 (violated rules), J.1 (formula, step window, witness agents) | — |
| Probabilities or other quantitative values | A.5 (risk alert with probability) | G.5 (QoS values as fitness), F.3 (worst-case member and value) |
| Scores: fitness, coverage | B.5 (plan length, failures), Eureka (training statistics), B.7's ablation (coverage) | G.5 |
| Execution errors and validator hints | B.1, B.2, B.5, C.1, C.3, AutoToS | — |
| Guidance on where to split the search | — | F.1 (inconsistent indices), G.2 and G.3 (splitting scores weighted by expected visits), G.4 (unsatisfiable cores) |
| Nothing (sample and select) | B.6, Corrêa et al. | — |

### 3.5 Comparisons with resampling or random search at a matched budget

| Entry | What was matched | Outcome |
|---|---|---|
| C.1 | Rounds: 15 back-prompts against 15 or 25 fresh samples | Resampling does about as well; the detail of the feedback barely matters. |
| C.2 | Number of programs; tokens in an appendix | Repair is marginal or worse at small budgets; diverse samples beat deeper repair. |
| B.2 | Maximum number of programs: 3 × 6 against 5 × 3 | No consistent winner. |
| B.5 | Generations and LLM calls, in the no-evaluator variant at a similar dollar cost | Without fitness, the loop performs like one-shot chain-of-thought. |
| B.7 | Not matched: 3.4 candidates against 25; coverage-only feedback in the same loop | Counterexample feedback beats sample-and-select and coverage-only feedback. |
| G.5 | Number of evaluations (10,000) | Genetic algorithms beat random search. |
| J.1 | Iterations, across feedback granularities | Full constraint feedback converges; metrics-only feedback stalls. No resampling arm. |
| B.1, B.4, A.2, A.3 | None | — |

### 3.6 Where the literature is dense and where it is thin

| Cluster | Full, most or partial reads (tiers) | Abstract, snippet or title-only entries |
|---|---|---|
| A. LLM output checked by PMC | 5 (Medium 3, Low 2) | 7 |
| B. LLM generalized policies and programs | 7 (High 3, Low–Medium 1, Low 3) | 6 |
| C. Feedback against resampling; LLM-Modulo | 3 (High 3) | 6 |
| D. LLMs with non-probabilistic verification | 2 (High 1, Medium 1) | 10 |
| E. Statistical and conformal guarantees | 1 (Medium 1) | 2 |
| F. Families, multi-environment, robust | 4 (High 3, Medium 1) | 9 |
| G. Synthesis with PMC in the loop | 6 (High 3, Medium 3) | 13 |
| H. Permissive strategies, compact strategies, shields | 2 (High 1, Medium 1) | 9 |
| I. Generalising policies to larger instances | 4 (High 1, Medium 1, Low–Medium 2) | 5 |
| J. Self-adaptive systems | 1 (Low 1) | 11 |
| K. In-context reuse | 0 | 7 |
| L. LLM controller synthesis in control | 0 | 5 |

Dense:
- LLMs with deterministic verifiers in classical planning (B, C), where most work is High tier.
- Exact synthesis over finite families and POMDP controllers without LLMs (F, G).
- Compact and permissive strategy representations (H).
- Feedback against resampling for code and short reasoning tasks (C).

Thin:
- LLM output checked by probabilistic model checking (A): no High-tier entry; of the five full entries, two are Low and one is a short paper.
- LLM work in self-adaptive systems with formal verification (J).
- In-context reuse with verification (K).
- Readable policies carried from small to larger instances of a probabilistic model: one non-LLM paper computes their value per instance (I.3); older relational and neural generalized policies are evaluated by simulation (I.5).

## 4. Limitations across approaches

1. **Guarantees stop at the instances that were checked.**
   - LLM generalized-planning work proves or validates on training tasks and tests larger tasks empirically (B.1, B.2, B.4, B.5, B.7). B.3 is the exception: its proof is deductive, but relative to a hand-written specification.
   - Exact family methods (F.1–F.3) and the multi-environment algorithms cover only finite families that share a state space.
   - Distribution-level methods (F.4, I.1, E.1) give statistical guarantees that assume i.i.d. sampling.
   - I.3 computes a policy's value on each larger instance, but its near-optimality there is extrapolated. The relational and neural generalized policies (I.5) are evaluated by simulation only.
2. **The probabilistic verifier often does not certify the LLM's artifact.**
   - A.3 uses it qualitatively; in A.4 it synthesises the policy itself; A.5 checks a learned model; A.1 certifies one seeded run.
   - Only A.2 feeds exact model-checking results back into an LLM, and only for one round.
3. **The faithfulness of LLM-written models and specifications goes unmeasured.** Where the LLM writes the model or the specification (A.3, A.4, D.2, Guan et al., Ramani et al.), correctness is judged by users or experts, or by cross-validating plans against a reference model.
4. **Much of the structure is supplied by humans.**
   - The non-LLM synthesisers need a human-written design space: sketches and option domains (G.1, F.1), memory sizes (G.2, G.3), predicate templates (H.2, G.4).
   - The LLM loops also lean on human input: strategies and constraint specifications (B.3), abstraction predicates (A.5), rule templates (A.3), curated task ranges (B.4).
5. **Objectives are narrow.**
   - Family and permissive synthesis mostly handle reachability or expected total reward against one threshold (F.1, F.2, H.1). Multi-objective constraints appear in G.1–G.3.
   - LLM work uses LTL only qualitatively (A.3, D.1).
   - LLM planning work handles goal reachability in deterministic domains only.
6. **Policies are often tables.**
   - Formal synthesis returns per-state tables, finite-state controllers or multi-strategies (F.1, F.3, G.2, H.1). The exceptions are decision trees (F.2, G.4, H.2, I.3) and the decision lists of I.5.
   - LLM-written programs are readable, but their guarantees are weaker (item 1).
7. **Evaluation practice.**
   - Single runs and single models are common (A.1, A.2, A.3, B.4, and B.5 with one run per domain); seeds and variance are often missing.
   - Matched-budget comparisons with resampling are rare (table 3.5).
   - Costs are reported in different units or not at all: tokens (B.2, A.5), dollars (B.5), nothing (B.4).
8. **Abstracts sometimes claim more than the results show.**
   - A.5: the headline numbers are not in the results table.
   - B.6: "provably sound" means only that execution is sound.
   - A.1: "exact guarantees" cover one seeded run.
   - A.3: reliability is shown only through perceived ratings.
9. **Contamination.** Most LLM planning work uses IPC domains. The mitigations are new domains (B.5, D.2) or anonymised names. Anonymisation has inconsistent effects: results collapse in B.1 and B.5, and are mixed in B.6.
10. **The scalability bottleneck differs by approach:**
    - one LLM call per reachable state (A.1);
    - the size of the MILP (H.1, OMDT);
    - pessimism of the abstraction (F.1 on Rocks-6-4);
    - SMT cost as tree depth grows (G.4);
    - exhaustive policy graphs and costly features (B.4);
    - verifier time caps that count timeouts as passes (B.7).

## 5. Gaps

These are observations about the literature found. Each also depends on the limits of the search (item 10).

1. **LLM policies for MDPs, revised over several rounds with exact probabilistic verification.**
   - Found:
     - an LLM acting as a per-state policy, verified once (A.1, preprint);
     - one LLM repair round on a DQN policy (A.2, short paper);
     - LLM plans checked qualitatively by PRISM (A.3);
     - LLM-written models solved by Storm (A.4);
     - runtime reflection triggered by model checking a learned model (A.5).
   - Not found: a peer-reviewed work in which an LLM revises a policy for a given MDP over several rounds, using quantitative model-checking results (values against thresholds, counterexamples, or where probability mass is lost).
2. **Probabilistic semantics in LLM generalized planning.**
   - Every LLM generalized-planning paper found (B) uses deterministic, fully observable PDDL domains and goal reachability; none handles stochastic transitions, or thresholds on probabilities or rewards.
   - LLMs do plan per task in stochastic environments (PlanU, B.8), without verification.
   - Without LLMs, generalized policies for probabilistic planning exist (relational decision lists, ASNets, SymNet; I.5), evaluated by simulation.
3. **Policies carried across instances of different sizes in probabilistic models, with a certificate on each instance.**
   - Exact family methods (F, G) need finite families with a shared state space.
   - Distribution-level methods (F.4) give PAC bounds over i.i.d. environments that share a state space.
   - Instances that vary in size are covered by deductive proofs for deterministic domains (B.3) or by statistical estimates per size (I.1).
   - Two non-LLM works do this with per-instance verification:
     - G.6 extracts finite-state controllers from an RNN over a shared observation space and verifies them exactly on the large instance.
     - I.3 learns decision trees over state variables from small instances and computes their value on each larger instance (exactly where the induced chain can be built, statistically otherwise). It has no refinement loop; I.4 proposes a counterexample-guided one for qualitative properties, without results.
   - Not found:
     - LLM-generated policies verified this way (see gap 2);
     - a published refinement loop driven by quantitative verification results on the larger instances.
4. **Partial or permissive policies from LLMs.** Without LLMs, permissive multi-strategies (H.1), robust permissive synthesis (H.3) and probabilistic shields (H.3) exist. Every LLM work found outputs a complete policy, a plan or overrides. Not found: an LLM work that outputs a partial or permissive policy and evaluates it by its best and worst case over completions.
5. **Feedback against resampling with quantitative verifiers.** Matched-budget comparisons exist only with deterministic verifiers on short answers or code (C.1, C.2), plus partial comparisons in B.2, B.5 and G.5 (G.5 has no LLM). Not found: such a comparison where the verifier returns probabilities or expected rewards.
6. **What a probabilistic verifier should feed back to an LLM.**
   - Non-LLM synthesisers use critical sub-Markov chains, splitting heuristics weighted by expected visits, worst-case members and unsatisfiable cores (3.4).
   - LLM loops return errors, violated rules, scalar scores or a probability alert.
   - Not found: a study comparing kinds of quantitative feedback for an LLM generator.
   - Ablations of feedback content with deterministic checkers disagree. In C.1 the detail barely matters; in J.1 and B.7 detailed feedback beats outcome-only feedback.
7. **Reporting practice.**
   - Single runs and single models are common in the LLM-loop papers.
   - Token budgets are rarely reported in comparable units.
   - Contamination is handled with new or anonymised domains, with inconsistent effects.
8. **Self-adaptive systems.**
   - Found: PRISM-in-the-loop adaptation without LLMs (J.2 background, G.5), and emerging work with LLM adaptation managers checked by runtime monitoring or simulation (J.1, J.2).
   - Not found: a SEAMS 2025 or 2026 paper that verifies LLM-generated adaptation policies with probabilistic model checking.
9. **In-context reuse with certificates.** Reuse of past solutions, components or corrections in context exists for agents and deterministic planning (K). There, success is judged by task outcome, as in Sarukkai et al.'s self-generated examples, or by constraint checks. Not found: work in which the reused examples are formally verified solutions of probabilistic models.
10. **Limits of this search.**
    - Sources: web search, arXiv, Semantic Scholar and reference lists. Paywalled proceedings were not searched systematically.
    - Preprints from September and October 2026 may be missing.
    - Several adjacent areas were covered only at abstract level: multi-environment theory, shields, probabilistic programs, control synthesis.

## 6. Roadblocks

**No central roadblocks.** Every paper I judged central was readable in full or in its main parts.

**Could not access** (can be supplied manually):

| Title | Link | Blocker | Central? |
|---|---|---|---|
| MAPER: Extending MAPE-K with LLM-Based Reasoning to Manage Unanticipated Situations in Self-Adaptive Systems (Maia, Vieira, Barros De Oliveira et al.; SEAMS 2026 full paper) | Programme: https://conf.researchr.org/track/seams-2026/seams-2026-research-track. The proceedings are in the ACM DL; "Mind the Prompt" from the same proceedings has doi 10.1145/3788550.3794879 | No abstract, preprint or DOI found by title search | No (relevant only for SEAMS positioning) |

**Extraction cut short.** The full texts are available:

| Paper | What was not read |
|---|---|
| D.1 DPO-AF (MLSys 2024) | The rest of Section 5.3 and the appendix (specifications, world models) |
| E.1 KnowNo (CoRL 2023) | The hardware results after the cut-off, and the appendices |
| D.2 Oswald et al. (ICAPS 2024) | Everything from the action-reconstruction-error analysis on, including the conclusion |
| C.3 LLM-Modulo (ICML 2024) | The final related-work paragraph |

**Read partly by choice.** The rest is available:

| Paper | Not read in full |
|---|---|
| G.3 JAIR 2025 | Sections 2.2–5 (formalism, splitting score, counterexample generalisation), skimmed by heading |
| G.6 Carr et al. | Result tables, skimmed |
| G.5 EvoChecker | Formal definitions and detailed tables, skimmed |
| G.4 dtPaynt | SMT encoding and refinement details, skimmed |
| F.4 Schnitzer et al. | Sections 3.1 and 3.4, proofs, benchmark appendices |
| F.2 SMPMC | Section 4 and appendices, skimmed |
| B.2, B.3, B.5, B.6, B.7 | Appendices; B.3's PDDL-to-Lean formalisation skimmed |
| A.4 Saccon et al. | Graph-construction details and discussion, skimmed |
| I.2 Müller et al. | Experiments, skimmed |
| I.3 1–2–3–Go! | Preliminaries skimmed; results tables in part; appendices not read |

**Abstract or snippet only, available.** These are the tables in A.6, B.8, C.4, D.3, E.2, F.5, G.7, H.3, I.5, J.2, K and L. They were left at abstract level for time, because they are peripheral or background, or because their facts were already covered by a full entry. If phase 2 needs more depth, the most useful next reads are probably:
1. Corrêa et al., NeurIPS 2025 (B.8)
2. Badings et al., STTT 2022 (F.5)
3. van der Vegt et al., TACAS 2023 (F.5)
4. Bovy et al., NeurIPS 2025 (F.5)
5. Andriushchenko et al., CAV 2026, probabilistic shields (H.3)
6. Cui et al. 2026 (B.8)
7. Li, Parsert and Polgreen, CAV 2024 (C.4)
8. GLM2FSA (D.3)
9. Batz et al., POPL 2024 (G.7)
10. Jansen et al., probabilistic shields (H.3)
11. Fern, Yoon and Givan, JAIR 2006, and Yoon, Fern and Givan, UAI 2002 (I.5): rule-list (decision-list) policies for relational MDPs, learned on small instances

**Venues not confirmed:** Jansen et al., probabilistic shields (CONCUR 2020?); PLPG (IJCAI 2023?); LLM-Planner (ICCV 2023?). These are from memory and are marked in the tables.

## 7. Search log

**When and how.** Searches ran on 2026-10-08, using:
- 50 web searches;
- about 75 arXiv abstract pages and 27 full texts (arXiv HTML, or PDFs converted to text);
- the Semantic Scholar Graph API, for forward citations of 6 papers;
- the reference lists of the seed papers and of most full entries, for backward citations;
- the SEAMS 2026 research-track programme, read title by title (https://conf.researchr.org/track/seams-2026/seams-2026-research-track).

**Searches**

| # | Query (shortened) | Notable results | Yield |
|---|---|---|---|
| 1 | LLM generates policy verified with probabilistic model checking (PRISM), feedback loop, refinement | A.1; A.3; LLMCHECKER and PMC of autoregressive models (A.6); a survey of safe LLM agents (2608.14590); access-control policy repair with LLMs (off topic) | New |
| 2 | LLM, MDP controller synthesis, Storm, counterexample feedback | A.4; G.2; Wu et al. (H.3); Bayat et al. and S2C (L); GLM2FSA (D.3) | New |
| 3 | "Prompt, Prove, Patch", Drexler | B.4's venue (RIPL at ICAPS 2026); CESS (D.3); Assured Automatic Programming via LLMs (2410.18494) | Venue |
| 4 | Title of the project's predecessor paper | Not indexed; LLM path-planning papers without formal verification (GridRoute, LLM-A*) | — |
| 5 | Gross & Spieker, ICTSS 2024 | A.2 | New |
| 6 | VeriPlan, CHI 2025 | A.3 | — |
| 7 | LLM writes a policy program for an MDP, PMC, iterative refinement, counterexample | A.4; AutoCedar (off topic); also, per the notes, G.3, A-LAMP and VerMCTS | Partly new |
| 8 | Silver et al., AAAI 2024 | B.1; also surfaced B.6 | New |
| 9 | Citations of Policies Grow on Trees, families of MDPs | F.3 | New |
| 10 | Oswald et al., ICAPS 2024 | D.2 | — |
| 11 | Olausson et al., self-repair | C.2 | — |
| 12 | Stechly et al., self-verification | C.1 | — |
| 13 | LLM writes a PRISM model from natural language, counterexample repair | Nothing direct; TraceFix (TLA+ counterexample repair, 2605.07935), ExVerus, unrelated papers named "PRISM" | — |
| 14 | LLM, self-adaptive system, MAPE-K, PRISM, SEAMS | J.2 entries (Li et al. ×2, Nascimento et al., the KIT report) | New |
| 15 | Pro2Guard | A.5 | — |
| 16 | DPO-AF | D.1; RLSF (2405.16661, fine-tuning from symbolic feedback; off topic) | — |
| 17 | Joint verification and refinement, Yang and Topcu | D.3 entry | — |
| 18 | LLM-Modulo, ICML 2024 | C.3 | — |
| 19 | Cui et al. 2026, abstraction generation | B.8 entry; B.3 | New (B.3) |
| 20 | LLM policy for stochastic planning (PPDDL, RDDL, MDP) with probabilistic guarantees and refinement | A.4; agentic PDDL simulation (C.4); VeriGuard; S2C (L). No stochastic generalized policies | None new in scope |
| 21 | LM generalized policy, probabilistic planning, families, larger instances | SayCanPay (heuristic planning, no verification); B.2; B.6; B.5's ICAPS page | B.5's venue |
| 22 | LLM and shield synthesis, probabilistic shields | Alshiekh et al. and PLPG (H.3); Easy-to-Use Shielding (tool, 2606.03804). No LLM-generated shields | New |
| 23 | Gros et al., ICAPS 2025 | I.1 | — |
| 24 | Multi-environment MDPs (Raskin and Sankur; Chatterjee et al.; van der Vegt et al.) | F.5 entries; F.4 (Schnitzer et al., TACAS 2025) | New |
| 25 | Conformal prediction for LLM planners, temporal logic, Kantaros | E.2 entries; E.1 | New |
| 26 | Carr et al., IJCAI 2019 | G.6; JAIR 2021 follow-up | — |
| 27 | EvoChecker | G.5 (ASE 2015 accepted version, ASEJ 2018) | — |
| 28 | Predecessor, LLMTrust workshop | Workshop page only; a paper citing A.1 (2608.30581, testing LLM explainers); a survey of safe agents | — |
| 29 | LLM proposes finite-state controllers for POMDPs, Storm, PAYNT, sketch | G.2; F.2; LLM-guided control-policy search (L) | New (L) |
| 30 | LLM adaptation policy with PMC, SEAMS 2025 and 2026 | GenAI-for-SAS roadmap; POLARIS; Casimiro et al. (J.2) | — |
| 31 | SEAMS 2026 safety operational rules; MAPER | J.2 entries; SEAMS 2026 track page; Plug in the Safety Chip (2309.09919) and AgentSpec (2503.18666): LTL and rule enforcement for LLM agents, deterministic | New |
| 32 | Jha et al., CEGIS with LLMs and SAT | C.4 entry | — |
| 33 | Inala et al., ICLR 2020 | G.7 entry; Program Machine Policy (NeurIPS 2023) | — |
| 34 | LLMs reusing solved instances as in-context examples across a problem family | K entries; in-context RL papers (2506.06303, 2602.04089); A-LAMP; PlanBench evaluation of o1 | New (K) |
| 35 | LLM policy synthesis for families of MDPs, multi-environment robust verification | Only papers already in the survey (F.1, A.1, A.4) | Saturation |
| 36 | LLM generates an MDP policy, verified with PRISM or Storm, refined from model-checking feedback (extended search) | Only known papers in scope (A.1–A.3). Also counterexample-guided RL policy refinement (NeurIPS 2021), CEGAR for POMDPs, and a GenPlan 2025 paper on counterexample-guided policy improvement for parameterized MDPs, which led to I.3 and I.4 | New (I.3, I.4, G.7) |
| 37 | Natural language to PCTL with LLMs, robot missions | As You Wish (ICRA 2026; LTL feedback loops); HERACLEs. No natural-language-to-PCTL work found | New (D.3) |
| 38 | LLM generalized policies for stochastic planning (PPDDL, RDDL) on larger instances | Only deterministic LLM work (B); SymNet; size-independent RDDL transfer; PlanU | New (I.5, B.8) |
| 39 | LLM adaptation strategies verified with PRISM, 2026 | POLARIS; the safe-agents survey; pre-LLM PRISM work on self-adaptation (e.g., Moreno et al., FSE 2015; Päßler et al. 2025, per a search summary). Nothing combining an LLM with PRISM for adaptation | — |
| 40 | LLM in counterexample-guided synthesis of MDP controllers; partial or permissive strategies | Known papers only (H.1, H.3, A.1, A.4) | Saturation |
| 41 | LLM agent verification with Markov chains learned from traces, 2026 | TraceToChain, SafetyDrift, the safe-agents survey (A.6); AgentLTL (2607.02599) and a paper on automata from agent traces (2608.23670), both deterministic and title-level only | New (A.6) |
| 42 | Fern, Yoon and Givan: approximate policy iteration with a policy language bias | I.5 (UAI 2002, JAIR 2006); SymNet | New (I.5) |
| 43 | ASNets | I.5 | New (I.5) |
| 44 | PlanU | B.8 | New (B.8) |
| 45 | 1–2–3–Go! by title | VMCAI 2025 venue confirmed; arXiv:2410.18293 | Venue |
| 46 | Generalising MDP policies from small to larger parameterized instances with decision trees and model checking | Only I.3 | Saturation |
| 47 | MAPER, SEAMS 2026, LLM reasoning in MAPE-K | SEAMS 2026 pages; the KIT report (J.2); no MAPER text | — |
| 48 | "Verify, Augment, Improve", SEAMS 2026 | KITopen record with abstract and open full text; SEAMS 2026 awards page | New (J.2) |
| 49 | "Mind the Prompt", task-plan explanations via LLMs | arXiv:2604.21092; SEAMS 2026 details page | New (J.2) |
| 50 | MAPER by exact title | Nothing (see [Roadblocks](#6-roadblocks)) | — |

**Citation trails**
- Forward, via Semantic Scholar:

  | Paper | Citing papers |
  |---|---|
  | Policies Grow on Trees | 3: SMPMC, rfPG, KAPS |
  | A.1 (Gross et al. 2025) | 1: a testing paper (2608.30581) |
  | VeriPlan | 54, including the predecessor, AgentGuard, Ramani et al., a multi-agent verification preprint (2605.25233) and HyPlan (driving; off topic) |
  | PAYNT | 25, none involving LLMs |
  | Dräger et al. | 22 since 2023, none involving LLMs |
  | 1–2–3–Go! (I.3) | 8, none involving LLMs: dtControl2+ε and Explainably Safe RL (H.3); two COOL-MC applications (platelet inventory, sepsis treatment); Parameterized Infinite-State Reactive Synthesis (PACMPL); Bellman-certified rounding for sparse policies (preprint); an AAAI 2025 abstract on explainable policy representations; a survey of consensus-protocol verification |

- Backward, from the reference lists of Policies Grow on Trees, PAYNT, Prompt Prove Patch, A.1, SMPMC, B.3, dtPaynt, LMPlan, GenePlan and Schnitzer et al. These gave the multi-environment cluster, Badings et al., OMDT, COOL-MC, Carr et al., Corrêa et al., Batz et al. and the UAI 2025 decision-tree follow-up.

**Venue checks:** arXiv comment fields for B.5 (ICAPS 2026), G.4 (CAV 2025), H.3's CAV 2026 shields paper and E.2 (ACM TCPS); proceedings pages for B.1, B.2, B.5, D.2, I.1 and C.3; F.4's own reference to its TACAS 2025 version; a university publication record for I.3 (VMCAI 2025); the SEAMS 2026 programme for the J.2 entries.

**Saturation.** Several signals suggest the search is close to saturated:
- Searches 35, 40 and 46 returned only papers already in the survey.
- Forward citations of PAYNT, Dräger et al. and 1–2–3–Go! contained no LLM work.
- The in-scope finds of searches 29–50 were in adjacent areas: parameterized-MDP policy generalisation (I.3, I.4), older generalized policies for probabilistic planning (I.5), control synthesis (L) and parametric robustness (F.4).

**Hits judged out of scope:**
- LLM path-planning benchmarks without verification (GridRoute, LLM-A*).
- Access-control policy synthesis with LLMs (AutoCedar, CloudFix, Prose2Policy).
- Event-B Agent.
- Name collisions with "PRISM": a proof-carrying artifact generator, a belief-space planner, an activation probe, a topic-clustering method.
- Self-Adapting Language Models (2506.10943), which concerns model weights, not software adaptation.
- mGPT (a probabilistic planner without LLMs).
- HyPlan (driving).
- Satisfiability-solving evaluations of LLMs (2605.28602).
