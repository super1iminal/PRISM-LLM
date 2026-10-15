# 12 · Baseline: LLM-written policy programs with simulation feedback (no model checker)

Read `plans/00-overview.md` first. Needs plan 02's explicit model (simulation and per-state certification).

## Goal

The closest LLM baseline from the related work, implemented and PRISM-certified on random grids, families and Pac-Man. It is generalized planning by LLM-written programs (Silver et al., AAAI 2024; Stein et al.; Prompt, Prove, Patch in `docs/related_work_survey.md`, section B):
- the same LLM writes a Python policy over the same observation and vocabulary;
- it is debugged from simulated episodes on the training instances;
- the result is frozen and certified like every other method.

It isolates the story's claim: the same model and the same training sets, but the feedback comes from simulation instead of the model checker, and the policy is a program instead of a rule list.

## Method

- **Prompt:**
  - the domain description and the training instances, as our prompt shows them;
  - the policy-visible variables, constants and features, given as precomputed booleans per action;
  - the requirements;
  - the ask: write `def policy(state, consts, features) -> action` that works on every instance, including larger unseen ones.

  Reuse the domain's templates where possible; any new template lives in the baseline module, not in `core/`.
- **Feedback (Silver et al.'s automated debugging):**
  - per training instance, the estimated value of each requirement over N simulated episodes;
  - one failing episode as a trace;
  - exceptions and timeouts with their tracebacks.

  The same round budget as ours (5) and the same keep-best idea (by estimated values). Report the tokens, to compare cost.
- **Safety:** LLM-written code runs in a subprocess (`python -I`), with a whitelist of the AST nodes and names it may use, no imports, no file or network access, and a per-call timeout. A program that breaks the rules counts as invalid and is re-asked with the error, like an invalid rule answer.
- **Certification:**
  - the frozen program is evaluated on every decision state of each target;
  - an exception or timeout at a state leaves it uncovered (worst case);
  - the per-state policy is certified through plan 02's path. The transfer CSV is the same.
- **Repeats:** 3 unseeded Haiku runs per benchmark, the same training sets as the core. No Sonnet (Haiku only until the final method).

## Steps

1. A design note in the module docstring (`src/baselines/llm_program.py`): prompt, feedback, sandbox, certification. It uses the LLM through `core/backends/` directly, since it is a baseline, not the planner.
2. Implement it, with tests:
   - the sandbox refuses imports, file access and long loops;
   - a hand-written program certifies the same as the equivalent rule list on the tiny grid;
   - with a scripted backend (`tests/fakes.py`), the loop's prompts contain the simulation feedback.
3. Smoke on one grid training set, then 3 runs per benchmark from the main checkout.
4. Add it to `rq1_comparison.yaml` as the method "LLM-written programs (simulation feedback)", under plan 01's `method:` grouping.

## Done when

Certified numbers for this baseline are in RQ1 on all three benchmarks, with token costs, tests pass, and it is committed.

## Files

Owns `src/baselines/llm_program.py`, `configs/baselines/llm_program_*.yaml` and its tests. Registers itself in `src/baselines/__init__.py` (one line).

## Report back

The numbers against ours at equal tokens, and three example programs (one good, one bad, one that crashed).
