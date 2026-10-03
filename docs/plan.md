# Work plan (after the Sept 24 meeting)

Target: **SEAMS 2027** (research track Oct 23; check the official site), with ICAPS (Dec 7 abstract) as the fallback. Write first, then finish. Rajan helps with implementation.

**Rule: finish every code change in Phase A before any LLM run**, so a small tweak never forces a re-run. Runs need an explicit go (see `CLAUDE.md`).

## Phase A: code (no GPU). Done on branch `phase-a` (A3 on `energy`)
| # | Change | Status |
|---|---|---|
| A0 | **One config:** `configs/default.yaml` + `conditions.yaml`, `src/config.py`, resolved `config.json` per run. | Done. See `docs/config.md`. |
| A1 | **Obstacle phase visible** (`domain.visible_extra: [obs_idx]`) to rules **and** legacy. | Done. Legacy makes one call per (goal, phase). Equivalence tests cover phase-observing policies. `pre_phase_a` reproduces the hidden setting. |
| A2 | **Joint best case** in the REFINE/EXTEND branch (PRISM `multi(…)`, sparse engine, exact LP). | Done. Each round logs `kept_joint_feasible` and `branch_disagreement`. If LP can't decide (step-bounded requirements, i.e. UUV), the result is *undecided* and the loop uses the per-requirement branch; PRISM's value-iteration variant wrongly said "no" on UUV. |
| A3 | **Reward requirements** (UUV energy). | Done on branch `energy` (worktree `../PRISM-LLM-energy`), to merge into `phase-a` after the gridworld batch. UUV thresholds ≤ 26.5 / ≤ 62.5, calibrated like the others. |
| A4 | **Retry policies**: `stall:k`, `never`, `every:k`, `gain:ε`, `always`. | Done (`src/core/retry.py`). R3 every:3, R4 gain:0.05. |
| A5 | **Seeds** (`llm.seed`, passed to Ollama). | Done. |
| A6 | **Ablation runner** `src/run_ablation.py` (conditions × seeds, resumable, `--dry-run`). | Done. |
| A7 | **Mass horizon H** configurable (`feedback.horizon`, `horizon_by_domain`). | Done. Gridworld 100. **Open:** UUV value (decide before the next UUV run). |
| A8 | **Analysis.** Ceilings (`src/ceilings.py`); ceiling-aware metrics, medians and coverage in `src/compare.py`; F1 budget curves (`viz/plot_budget.py`, pools `seed_*` dirs). | Done. Still to do once runs exist: pool seeds in `viz/plot_runs.py`, paired tests. |
| A9 | Tests (41 passing). Dry runs of the loop with a fake LLM, B2/R3/R4/R5. | Done. **Smoke runs with the real LLM need a go.** |

**Found while implementing:** with `obs_idx` visible, rules can make the induced chain *periodic*, and PRISM's default value iteration then fails to converge (an instance would crash mid-run). The default solver is now Gauss-Seidel (`prism.method`), and joint queries use exact LP. Checks that reproduce old numbers (the UUV paper, the regression) pin PRISM's defaults.

## Phase B: deliverables without GPU (ready)
- **Ceilings:** `out/results/ceilings/gridworld_grid_20_balanced.md` and `uuv_uuv_paper.md`.
  - Gridworld: 19/20 grids jointly solvable, and almost every requirement's optimum is 1.0. Grid 17 is not solvable: `complete_sequence` can reach at most 0.70 against a 0.80 threshold.
  - Re-scored existing runs (`src/compare.py`): solved of solvable 0/19 legacy vs 1/19 symbolic.
  - UUV: joint undecided (step-bounded), but the `stay` reference policy is a witness that both scenarios are solvable.
- **Budget curves** from the existing runs: `viz/figures/budget_grid20.png`. Symbolic starts behind (3.25 vs 4.00 requirements met after 1 round) but improves every round to 5.25; legacy plateaus at 4.30 after round 3.
- `docs/semantics.md` updated for A1–A2.

## Phase C: runs (with a go)
2 seeds per condition, every seed's outputs kept (`docs/ablations.md`). **All symbolic runs first, then legacy.**
1. Smoke tests: `python src/run_ablation.py B2 R5 B1 --seeds 1 --set run.limit=1` (then delete those dirs).
2. B2 symbolic, 2 seeds (~1 h): `python src/run_ablation.py B2`.
3. Retry sweep, 2 seeds each (~5 h): `python src/run_ablation.py R1 R2 R3 R4 R5`.
4. B1 legacy with the obstacle visible, 2 seeds (estimated ~9 h): `python src/run_ablation.py B1`.
5. Later: UUV with the energy requirement (after A3), and the deferred ablations once the story is settled.

## "Why an LLM?" (Asher to write; inputs)
The meeting asked for a reason the LLM goes *beyond* PRISM: ~3 paragraphs plus early evidence. Candidates:
- **Expressiveness gaps.** Requirements or structure PRISM can't express, or only through awkward or approximate encodings (Aren). Example: the UUV paper's three-phase mission structure, of which we model one phase.
- **Model engineering.** Start from the problem description. The LLM builds the PRISM model, PRISM's counterexamples drive revisions, and the revisions double as a justification of the model (Marsha).
- **Restricted observation.** Memoryless observation-based policies are hard to synthesize, and PRISM's POMDP engine rejected our obstacle requirements outright.

Notes:
- An **energy budget is expressible in PRISM** (the paper's Table 2 uses rewards). A3 closes a gap in *our core*, not in PRISM, so it doesn't support the expressiveness argument by itself.
- The ceilings make the synthesis question sharper: PRISM's optimum reaches 1.0 on almost every gridworld requirement.
