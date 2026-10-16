# 08 · More benchmark models where 1-2-3-Go! reports results

Read `plans/00-overview.md` first. The baselines need plan 02's generic explicit-model layer. Scouting starts now; the domains are for ICAPS (Dec 7), unless one is ready by Oct 20.

## Goal

Two to four parameterized MDPs from 1-2-3-Go!'s evaluation become domains. Each has:
- a test that reproduces a published optimum;
- a rule vocabulary and per-instance thresholds;
- training, validation and sealed test instances;
- every generic baseline from plan 02.

Pick a mix: some where 1-2-3-Go! fails (where we must win) and one where it does well (where we must at least match).

## The candidates (1-2-3-Go!, VMCAI 2025, arXiv:2410.18293, Tables 1, 2 and 4)

| Model, property | Parameter | 1-2-3-Go! | Source of the model |
|---|---|---|---|
| consensus (N=2), `c2` (min) and `disagree` (max) | K | fails at small K (0.469 against 0.383 at K=2) and on `disagree` at every K | generated: `third_party/123go/scripts/generate-models.py` (`generate_models_consensus`) |
| csma (N=2), `some_before` | K | near-optimal | generated (`gen_csma`) |
| firewire_dl, `deadline` | deadline or delay | near-optimal | QVBS |
| zeroconf_dl, `deadline_max` / `deadline_min` | deadline, K | max near-optimal, min fails when N scales | QVBS |
| mer, `p1` | x (a probability) | fails for x ≥ 0.08 (0.19999 against 0.214–0.671) | Feng et al. PRISM case study, as in the artifact. **Done:** `domains/mer/` (steps 2 and 4–7), translated from the artifact's `test-data/models/mer.jani`, so nothing was downloaded; baselines, Haiku runs and RQ1 entry still to do |

**Rule for now:** the instances of one family must share the policy-visible variables (`docs/semantics.md`, "Training sets"). Parameters that add modules (N) are out until the rule language handles indexed variables; note that as an ICAPS question, not a task. That keeps consensus and csma at N=2 (K varies), firewire, zeroconf (deadline, K) and mer (x).

**Framing caveat:** in consensus, csma, firewire and zeroconf the nondeterminism is a protocol's scheduler, so the "policy" is a scheduler, as 1-2-3-Go! treats it. Say so in each domain's description and in the paper.

## Steps

1. **Ask Asher before downloading** the QVBS models (firewire_dl, zeroconf_dl) or the mer model: name the files, the source and their size. The consensus and csma generators are already in `third_party/123go` (untrusted data). Read them and write our own Jinja templates; never run the artifact's scripts.
2. **Scout each candidate (PRISM only, no LLM).**
   - Build 3–4 instances along the family's parameter.
   - Record states, decision states and time to the optimum.
   - Reproduce two of 1-2-3-Go!'s published Storm optima per family with PRISM.
   - Check that the actions can be labelled uniquely across instances (what the artifact's `give_action_names_to_models.py` does: one label per command).
   - Report a table: model, sizes, times, reproduced values, verdict. Drop a candidate that PRISM cannot solve at validation sizes within minutes.
3. **Pick 2–4 with Asher**, at least one where 1-2-3-Go! fails and one where it succeeds. Recommended start: consensus N=2 over K (fails; generated, no download needed) and firewire over deadline (succeeds).
4. **Domain per pick**, following `domains/README.md` (`domains/gridworld/` is the reference, `domains/pacman/` the closest):
   - `model.prism.j2` with labelled actions;
   - `spec.yaml.j2`: policy-visible variables (the state variables, as 1-2-3-Go! uses), actions and the requirement;
   - `description.md.j2`, `examples.md.j2` and `examples_extended.md.j2`, `visual.txt.j2` if one makes sense;
   - the rule vocabulary: the instance constants (K, deadline, …) and a few generic features. Features are written in the domain and computed from the instance, never from PRISM;
   - `domain.py`.
5. **Thresholds per instance** (Asher's decision 4).
   - The optimum plus 5% of it (minimizing) or minus 5% (maximizing), from `src/calibrate.py` per instance, so "certified" reads like 1-2-3-Go!'s "near-optimal".
   - Where the optimum is 0 or 1, a relative slack does nothing: report those instances and ask Asher.
   - Thresholds are data in the dataset CSV, like Pac-Man's `crash_threshold`.
6. **Datasets.**
   - Training: 1-2-3-Go!'s base instances (Table 4: consensus N=2 K=8, firewire deadline=200 delay=3, …) plus one or two more small ones.
   - Validation: middle sizes.
   - Test: the largest sizes PRISM solves, **sealed** in `configs/transfer/default.yaml` from the first commit.
7. **Tests** (`tests/test_<domain>.py`): the reproduced optima, as `tests/test_pacman.py` does, and one rule list per domain that the verifier certifies.
8. **Baselines** from plan 02's generic layer:
   - trees over raw variables trained on 1-2-3-Go!'s base instances (sanity check against its published values; report, do not tune);
   - trees over features, PPO, uniform, random-best-of-1000, and the one-instance strategy.
9. **Haiku.** First a smoke run (`--set run.limit=1`), then the core configuration: multi-instance training on the small instances, `rules.any_order: all` (or `rotations` when there are many actions), 5 rounds, 3 repeats, transfer to validation.
10. **RQ1.** Add each domain as a benchmark in `rq1_comparison.yaml` (plan 01's structure), once Asher has seen the scouting table and the first results.

## Done when

- The scouting table is reported.
- At least two domains are merged with tests and sealed test sizes, their baselines certified, and Haiku runs done.
- They appear in RQ1; committed.

## Files

Owns `domains/<new>/`, `tests/test_<new>.py`, `configs/run/conditions/haiku_<new>*.yaml` and `configs/baselines/<new>_*.yaml`. Adds benchmark entries to `rq1_comparison.yaml`. No changes to `src/core/`. If a domain truly needs one, make it generic and log it in `DECISIONS.md`.

## Report back

The scouting table, the picks, the reproduced optima, and per-domain numbers for ours against 1-2-3-Go! reimplemented and published.
