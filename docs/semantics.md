# Rule-set semantics

> **Keep this current.** It describes the inputs as they are today: a user-written PRISM MDP and LLM-written rules over declared variables. Update it whenever the inputs change (e.g. the LLM builds the model), the rule language changes, or the REFINE/EXTEND branch changes. Items marked *(planned)* are not implemented yet; see `docs/plan.md`. Last updated for Phase A (branch `phase-a`).

## Inputs (per instance)
- **MDP** `M = (S, s0, Act, P)`, given as a PRISM `mdp` model. `A ⊆ Act` are the *policy actions*: the action labels the policy controls. Other labels (e.g. UUV's `[step]`) are never restricted.
- **Policy-visible variables** `X`, a subset of the state variables, with finite domains. `obs(s)` is the projection of a state `s` onto `X`.
- **Requirements** `φ_i ⋈_i θ_i`, where `⋈_i ∈ {≥, ≤}` and `θ_i` is a threshold, on either a probability `P[φ_i]` (`φ_i` a PRISM path formula) or an expected reward `R{r}[φ_i]` (e.g. UUV's expected energy until `F "done"`). Below, "value" means either. Best/worst for a reward bound `≤` are `Rmin`/`Rmax`. Shortfalls and blame stakes of reward requirements are taken relative to `θ_i`, so they are summed on the same scale as probabilities.

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
- **Joint best case** `J`: does a single scheduler `σ` of `M_R` exist with `P^σ(φ_i) ⋈_i θ_i` for all `i` at once? This is PRISM's `multi(…)` query (sparse engine, exact LP). Per-requirement best cases passing doesn't imply this. A "no" is exact. A "yes" is still optimistic for memoryless, observation-based completions. Where LP doesn't apply (step-bounded requirements, e.g. UUV's deadline), `J` is *undecided*.
- **Observation:** `X` = the spec's variables plus the config's `domain.visible_extra` (gridworld default: `obs_idx`, for legacy too). Any state variable outside `X` is seen by the schedulers but not the rules, which makes `w_i` conservative and `b_i` optimistic. The runs before Phase A hid `obs_idx` (condition `pre_phase_a`).
- **Numerics:** values are computed with Gauss-Seidel (`prism.method`). Rules that read `obs_idx` can make `M_R` periodic, and plain value iteration then does not converge.

## Loop branch
Each round: verify `R`, then
- **done** if every `w_i` passes;
- **REFINE** (the LLM rewrites the whole list) if no completion can pass: some `b_i` fails, or else `J` is "no" (`planner.branch: joint`; the old runs used `per_requirement`, which never asks `J`);
- **EXTEND** (the LLM returns new rules, appended to the end) otherwise, including when `J` is undecided;
- unless the retry policy (`planner.retry`) says to start over from the initial prompt instead. That check comes first.

Appending only removes choices at previously uncovered states, so EXTEND never lowers `w_i` and never raises `b_i`. The best rule set so far is kept, scored lexicographically by (worst-case failures, best-case failures, worst shortfall, best shortfall).

The loop is per instance and sees the LLM only through tasks and results (`src/core/tasks.py`). How instances are interleaved (`run.scheduler`: threads, or lockstep batches) does not change any instance's rounds: given the same answers, both give identical prompts, rule sets and values.

## PRISM encoding
The planner adds one variable-free module that synchronizes on every label in `A`:

```
module policy
  [a] (e_j for rules j with a_j = a) | !(c_1 | … | c_n) -> true;   // one command per a ∈ A
endmodule
```

Here `e_j = c_j ∧ ¬c_k` for each earlier rule `k` that can also match (found by enumerating the `X` domain when it has ≤ 200k valuations, otherwise all earlier rules). A label whose guard is false is blocked by synchronization. Code: `src/core/rules.py` (`to_prism_module`), `src/core/verifier.py`.
