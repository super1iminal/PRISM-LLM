# Rule-set semantics

> **Keep this current.** It describes the inputs as they are today: a user-written PRISM MDP and LLM-written rules over declared variables. Update it whenever the inputs change (e.g. the LLM builds the model), the rule language changes, or the REFINE/EXTEND branch changes. Items marked *(planned)* are not implemented yet; see `docs/plan.md`.

## Inputs (per instance)
- **MDP** `M = (S, s0, Act, P)`, given as a PRISM `mdp` model. `A ⊆ Act` are the *policy actions*: the action labels the policy controls. Other labels (e.g. UUV's `[step]`) are never restricted.
- **Policy-visible variables** `X`, a subset of the state variables, with finite domains. `obs(s)` is the projection of a state `s` onto `X`.
- **Requirements** `φ_i ⋈_i θ_i`, where `φ_i` is a PRISM path formula, `⋈_i ∈ {≥, ≤}` and `θ_i` is a threshold. *(Planned: expected-reward requirements `R{r} ≤ c [F goal]`, for UUV energy.)*

## Rules
A rule set is an ordered list `R = ⟨(c_1, a_1), …, (c_n, a_n)⟩`, where each `c_j` is a boolean expression over `X` (`= != < <= > >= + - & | ! =>`, integer/boolean constants) and `a_j ∈ A`.

1. **Deciding rule:** `first(v) = min{ j : v ⊨ c_j }`, or `⊥` if no rule matches. First match wins, so overlapping rules never produce nondeterminism.
2. **Allowed actions:** `Allow(s) = {a_first(obs(s))}` if `first(obs(s)) ≠ ⊥`, else all of `A`.
3. **Induced MDP** `M_R`: same states and transitions as `M`, but at each `s` only the choices whose label is in `Allow(s)` or not in `A`.
4. **Decision states:** `s` is a decision state if its choices in `M` differ (forced states, where every choice has the same successor distribution, are excluded). It is **uncovered** if it is a decision state and `first(obs(s)) = ⊥`. Coverage counts distinct `obs` values over the reachable decision states of `M_R`.

Domains must keep every policy action enabled in every decision state (gridworld and UUV do), so restricting `Allow(s)` never deadlocks.

## Verification
- **Best case** `b_i` = the optimum over all schedulers of `M_R` (`Pmax` if `⋈_i` is `≥`, `Pmin` if `≤`). **Worst case** `w_i` is the opposite optimum.
- **Soundness.** PRISM's schedulers are history-dependent and randomized, and they see the full state. So `[w_i, b_i]` contains the value of *every* deployment consistent with `R`: any fallback at uncovered states, with or without memory or extra observations. **If `w_i ⋈_i θ_i` for all `i`, every such deployment satisfies every requirement.** This is the success criterion. A passing rule set is a *permissive strategy* (Dräger et al.).
- **Full coverage** (no reachable uncovered state) gives `b_i = w_i`: the value of the single policy `R` defines.
- **Joint best case** *(planned)*: does a single scheduler `σ` of `M_R` exist with `P^σ(φ_i) ⋈_i θ_i` for all `i` at once? This is PRISM's `multi(…)` query (sparse engine). Per-requirement best cases passing doesn't imply this. A "no" is exact. A "yes" is still optimistic for memoryless, observation-based completions.
- **Observation:** when `X` omits state variables (gridworld's `obs_idx`), the schedulers see more than the rules, so `w_i` is conservative and `b_i` optimistic. *(Planned: per-domain switch making them visible; on for both legacy and symbolic.)*

## Loop branch
Each round: verify `R`, then
- **done** if every `w_i` passes;
- **REFINE** (the LLM rewrites the whole list) if no completion can pass, judged per requirement today *(planned: joint best case)*;
- **EXTEND** (the LLM returns new rules, appended to the end) otherwise.

Appending only removes choices at previously uncovered states, so EXTEND never lowers `w_i` and never raises `b_i`. The best rule set so far is kept, scored lexicographically by (worst-case failures, best-case failures, worst shortfall, best shortfall).

## PRISM encoding
The planner adds one variable-free module that synchronizes on every label in `A`:

```
module policy
  [a] (e_j for rules j with a_j = a) | !(c_1 | … | c_n) -> true;   // one command per a ∈ A
endmodule
```

Here `e_j = c_j ∧ ¬c_k` for each earlier rule `k` that can also match (found by enumerating the `X` domain when it has ≤ 200k valuations, otherwise all earlier rules). A label whose guard is false is blocked by synchronization. Code: `src/core/rules.py` (`to_prism_module`), `src/core/verifier.py`.
