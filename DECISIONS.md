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
