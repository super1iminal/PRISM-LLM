# Work plan (after the Sept 24 meeting)

Target: **SEAMS 2027** (research track Oct 23; check the official site), with ICAPS (Dec 7 abstract) as the fallback. Write first, then finish. Rajan helps with implementation.

**Rule: finish every code change in Phase A before any LLM run**, so a small tweak never forces a re-run. Runs need an explicit go (see `CLAUDE.md`).

## Phase A: code (no GPU)
| # | Change | Notes / decisions needed before coding |
|---|---|---|
| A1 | **Observation switch:** a per-domain `visible` option; gridworld exposes `obs_idx` to rules **and** to legacy. | **Legacy cost:** one action per (cell, phase) means 8×8 × 6 = 384 actions per goal, which is over the 8192-token output cap. Proposal: legacy makes one call per goal *per obstacle phase* (64 actions each). Regression and `legacy_translate` gain `obs_idx`; the hidden setting stays available for the old runs. |
| A2 | **Joint best case** in the REFINE/EXTEND branch (PRISM `multi(…)`, `-sparse` engine, about 1.3 s per check). | Log per-requirement vs joint disagreements per round, so we can report how often the old branch was wrong. |
| A3 | **Reward requirements** in core: `R{r} ≤ c [F goal]`, best = `Rmin`, worst = `Rmax`; UUV gets an energy budget. | Calibrate the threshold with `domains/uuv/data/calibrate.py` (as for the others). Mass analysis uses the reward vectors (the stake is a cost-to-go gap). PRISM's multi-objective queries accept mixed P/R objectives. |
| A4 | **Retry policies** in `PlannerConfig`: `stall:k`, `never`, `every:k`, `gain<ε`, `always`. | R3: k = 3. R4: ε = 0.05 total shortfall. |
| A5 | **Seeds:** `--seed` in both runners, passed to Ollama's `seed` option. | |
| A6 | **Ablation runner** `src/run_ablation.py` with a condition registry, and the `out/results/ablations/…` layout. | See `docs/ablations.md`. |
| A7 | **Mass horizon H** per domain (default 100; UUV = its deadline). | Decide now; it changes feedback, so changing it later means re-runs. |
| A8 | **Analysis:** ceilings script, renormalized metrics, F1 budget curves, revised write-up statistics (medians and spread, input tokens, PRISM time, coverage, no "every metric"). | |
| A9 | Tests for A1–A4, plus a `--limit 1` smoke run of each changed path (asks first). | |

## Phase B: deliverables without GPU (can go to Marsha before any run)
- **Per-instance ceilings:** for each grid and UUV scenario, PRISM's best achievable value per requirement (bare MDP, full state) and whether all thresholds are achievable *at once* (`multi` query). This separates "the LLM failed" from "the instance is impossible". Shortfall is renormalized against the achievable value.
- `docs/semantics.md` (drafted; update after A1–A3).
- F1 budget curves from the existing runs (illustrative only, since they predate A1–A2).

## Phase C: runs (after A, with a go)
1. Smoke tests (`--limit 1`) for each changed path.
2. Main comparison B1 + B2: gridworld, 3 seeds. Legacy is the expensive one, an estimated ~4.5 h per seed with the obstacle visible.
3. UUV symbolic with the energy requirement: both scenarios, 3 seeds (minutes).
4. Retry sweep R1–R5: 3 seeds each, about 7.5 h total.
5. Deferred ablations only after the story is settled.

## "Why an LLM?" (Asher to write; inputs)
The meeting asked for a reason the LLM goes *beyond* PRISM: ~3 paragraphs plus early evidence. Candidates:
- **Expressiveness gaps.** Requirements or structure PRISM can't express, or only through awkward or approximate encodings (Aren). Example: the UUV paper's three-phase mission structure, of which we model one phase.
- **Model engineering.** Start from the problem description. The LLM builds the PRISM model, PRISM's counterexamples drive revisions, and the revisions double as a justification of the model (Marsha).
- **Restricted observation.** Memoryless observation-based policies are hard to synthesize, and PRISM's POMDP engine rejected our obstacle requirements outright.

Note: an **energy budget is expressible in PRISM** (the paper's Table 2 uses rewards). A3 closes a gap in *our core*, not in PRISM, so it doesn't support the expressiveness argument by itself.
