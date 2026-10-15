# 02 · Pac-Man baselines: 1-2-3-Go! reimplemented, PPO, and the simple ones (generic over domains)

Read `plans/00-overview.md` first.

## Goal

Pac-Man gets real baselines, all PRISM-certified on the same horizons as our rules:
- 1-2-3-Go! reimplemented: trees over the raw state variables and trees over our features;
- PPO;
- the cheap baselines of the 1-2-3-Go! paper (uniform policy, best of 1000 random deterministic policies);
- the strategy synthesized on one instance and applied to the others (Marsha's baseline).

They are built on a **generic explicit-model layer**, so plan 08's new domains get the same baselines for free.

## Why

On Pac-Man we compare only with hand-written rules. The gridworld baselines run on a numpy model of the gridworld (`src/baselines/gridworld_mdp.py`) and are gridworld-only. 1-2-3-Go! itself reports Pac-Man: near-optimal at horizon 5 (its training horizon), 0.869 at 25, 0.99998 at 200 and 300, against an optimum of 0.5511 at every horizon. Our frozen Haiku rules hold the optimum up to 18 and reach 0.749 at 25.

## Before starting

Read `src/baselines/` (`base.py`, `decision_tree.py`, `ppo.py`), `src/run_baseline.py`, `configs/baselines/README.md`, `src/transfer.py`, `src/core/prism.py` (`PrismRunner.run(..., export_transitions=True)` already returns every state's valuation and its choices with action labels and successors), `src/core/verifier.py` (`policy_valuation`, decision states, forced states), `domains/pacman/domain.py` and `model.prism.j2` (the `arbiter` module's unlabelled step), `tests/test_pacman.py` and `tests/test_baselines.py`.

## Steps

1. **Size check (PRISM only).** For horizons 5, 25, 50, 100 and 200: states, reachable decision states and time to the optimum. Only the bare model's optimum, never a policy's value on 50, 100 or 200, since those are sealed. This decides whether per-state rule lists are feasible (step 4) and whether 200 is practical. If PRISM cannot solve 200 within the timeout, drop it and tell Asher.
2. **Dataset rows** (Asher's decision 1). Append to `domains/pacman/data/pacman_transfer.csv`, which keeps the ids that runs refer to:
   - `steps_05` (1-2-3-Go!'s base instance, for training);
   - `steps_30` (validation);
   - `steps_50`, `steps_100`, `steps_200` (the sealed test set).

   Add the three sealed ids to `sealed` in `configs/transfer/default.yaml` **in the same commit**, and check that `transfer.py` and `run_baseline.py` refuse them without `--unseal`. Every baseline below is certified on the validation horizons (5, 8–25 except its training ones, 30) only.
3. **Explicit model** (`src/baselines/explicit_mdp.py`, domain-agnostic, no Pac-Man code):
   - `ExplicitMDP(domain, instance)`: the domain's model as PRISM exports it. Policy-visible valuations, choices with labels and successors, state labels, the initial state.
   - Decision states exactly as the verifier defines them. Reuse its code, do not redefine.
   - Value iteration in numpy for reachability and until objectives over labels, enough for Pac-Man's `F "Crash"` and the QVBS properties. Anything else raises `NotImplementedError`.
   - The optimal action set per decision state (every action within 1e-9 of the optimum).
   - The optimum is over all schedulers, arbiter included, as in Storm and in 1-2-3-Go!.
   - Test: on horizons 8 and 25 the numpy optimum equals PRISM's 0.5511.
   - Gridworld keeps `GridMDP`; its baseline numbers must not change. A test on one small grid that the explicit optimum equals `GridMDP`'s is enough.
4. **Certifying a per-state policy.** Compact rules where they exist (trees over raw variables, step 5). Otherwise either:
   - per-state rules over the visible variables, as `decision_tree.policy_rules` does for gridworld, if step 1 shows the decision states are few enough for the rule compiler; or
   - a restricted explicit model written as PRISM explicit files and checked with `-importmodel`, so it is still PRISM that certifies.

   Whichever is used, the output is `transfer.py`'s CSV, and both are tested against each other on horizon 8.
5. **1-2-3-Go! reimplemented, generic.** A `model: explicit` path in `decision_tree.py`, or a sibling class; leave the gridworld path byte-identical.
   - Training data: the reachable decision states of the training instances, with *every* optimal action (one sample per optimal action, weight 1/k: the paper's majority vote per leaf).
   - CART with Gini, grown until pure.
   - Inputs `raw`: the policy-visible state variables (as the paper: `xP, yP, dP, xG0, …, steps`).
   - Inputs `features`: the rule-language features of all four actions plus `moves_left`, evaluated with the core rule evaluator on each state (no copied expressions).
   - A tree over raw variables compiles exactly into a decision list: one rule per leaf, the path's `var <= t` / `var >= t+1`, the leaf's action. This needs a baseline-only condition with `rules.extended: true, general: false`, so numbers are allowed (`configs/run/conditions/pacman_baseline.yaml`). Report the rule count.
   - A tree over features goes through step 4: a per-action feature can only name the rule's own action.
   - Training sets: horizon 5 ("as in the paper") and horizons 14, 16 and 18 (ids 3–5, our training set). Each is a source.
6. **PPO, generic** (`ppo.py` `model: explicit`, gridworld path unchanged).
   - The environment samples the explicit model. The policy acts at decision states. Other steps sample successors. Remaining nondeterminism, such as the arbiter, is resolved uniformly in training; say so in the docstring.
   - Reward: −1 on entering a `Crash` state, which ends the episode. The episode also ends at the horizon.
   - Observations: raw (normalized) and features.
   - Trained on ids 3–5, 3 seeds: three sources. Frozen as argmax, certified through step 4.
7. **Simple baselines** (`src/baselines/simple.py`):
   - uniform randomized policy (a DTMC, certified through step 4's explicit path);
   - best of 1000 random deterministic memoryless policies, ranked on the training instances by the numpy value, then certified on the targets;
   - "strategy of one instance applied to the others": the optimal strategy of horizon 18 as a table over the visible variables, aligned to other horizons by `steps` or by `moves_left` (two sources). Observations never seen in training are left uncovered (worst case).
8. **Configs and runs.** One `configs/baselines/pacman_<method>.yaml` per method; `src/run_baseline.py` for each; results in `out/results/baselines/<stem>/transfer_pacman_transfer.csv`. Add the entries to `rq1_comparison.yaml` under plan 01's `method:` grouping. Then `src/run_experiment.py configs/experiments/rq1_comparison.yaml --baselines`.
9. **Sanity check against the paper, not tuning to it.** A tree trained on horizon 5 should be near-optimal at 5 and far above 0.56 at 25 (published 0.869). Report the number. If it differs a lot, say why (tie-breaking, sklearn against dtControl, the label set) rather than adjusting the method to match.
10. **Optional:** run the generic trees and PPO on one random grid as a cross-check against the gridworld implementations (same order of magnitude); no new gridworld numbers.

## Done when

- The RQ1 Pac-Man table has PRISM-certified rows for trees (raw: horizon-5 and 14–18 training; features), PPO (raw, features), uniform, random-best-of-1000, and both one-instance strategies.
- Plan 01's horizon figure has their lines.
- Tests cover the explicit optimum, the tree-to-rules compilation (the rules pick the tree's action in every decision state) and the explicit certification path.
- `docs/experiments.md` ("Methods") and `configs/baselines/README.md` are updated.

## Files

Owns `src/baselines/` (except plan 12's new module), `configs/baselines/pacman_*`, `configs/run/conditions/pacman_baseline.yaml`, and the appended rows of `domains/pacman/data/pacman_transfer.csv`. Adds entries to `rq1_comparison.yaml`.

## Report back

The step-1 sizes, the per-method numbers per horizon, rule counts of the trees, and the 1-2-3-Go! sanity check.
