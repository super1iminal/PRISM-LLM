# Work plan

Target: **SEAMS 2027** (research track Oct 23; check the official site), with ICAPS (Dec 7 abstract) as the fallback. Write first, then finish. Rajan helps with implementation.

**Rule: every code change in Phase A lands before any LLM run**, so a small tweak never forces a re-run. Runs need an explicit go (see `CLAUDE.md`).

## Phase A: code (no GPU). Done
| # | Item | Status |
|---|---|---|
| A0 | **One config:** `configs/run/default.yaml` + one file per condition in `configs/run/conditions/`, `src/config.py`, resolved `config.json` per run. | Done. See `docs/config.md`. |
| A1 | **Obstacle phase visible** (`domain.visible_extra: [obs_idx]`) to rules **and** legacy. | Done. Legacy makes one call per (goal, phase). Equivalence tests cover phase-observing policies. `pre_phase_a` hides the phase. |
| A2 | **Joint best case** in the REFINE/EXTEND branch (PRISM `multi(…)`, sparse engine, exact LP). | Done. Each round logs `kept_joint_feasible` and `branch_disagreement`. If LP can't decide (step-bounded requirements, i.e. UUV), the result is *undecided* and the loop uses the per-requirement branch (PRISM's value-iteration variant wrongly answers "no" there). |
| A3 | **Reward requirements** (UUV energy). | Done. UUV thresholds ≤ 26.5 / ≤ 62.5, calibrated like the others. |
| A4 | **Retry policies**: `stall:k`, `never`, `every:k`, `gain:ε`, `always`. | Done (`src/core/retry.py`). R3 every:3, R4 gain:0.05. |
| A5 | **Seeds** (`llm.seed`, passed to Ollama). | Done. |
| A6 | **Ablation runner** `src/run_ablation.py` (conditions × seeds, resumable, `--dry-run`). | Done. |
| A7 | **Mass horizon H** configurable (`feedback.horizon`: steps, or `domain` for the domain's own). | Done. Gridworld 100; UUV the mission deadline (U1 sets `domain`, see `DECISIONS.md`). |
| A8 | **Analysis.** Ceilings (`src/ceilings.py`); ceiling-aware metrics, medians and coverage in `src/compare.py`; F1 budget curves (`viz/plot_budget.py`, pools `seed_*` dirs); paired tests in `viz/ablation_summary.py`. | Done. Open: pool seeds in `viz/plot_runs.py`. |
| A9 | Tests, including the loop with a scripted LLM (`tests/fakes.py`). | Done. |

Periodic chains need specific solver and LP settings: see `DECISIONS.md`, "Phase A".

## Phase B: deliverables without GPU. Done
- **Ceilings:** `out/results/ceilings/gridworld_grid_20_balanced.md` and `uuv_uuv_paper.md`.
  - Gridworld: 19/20 grids jointly solvable, and almost every requirement's optimum is 1.0. Grid 17 is not solvable: `complete_sequence` can reach at most 0.70 against a 0.80 threshold.
  - The single-seed runs scored against the ceilings (`src/compare.py`): solved of solvable 0/19 legacy vs 1/19 symbolic.
  - UUV: joint undecided (step-bounded), but the `stay` reference policy is a witness that both scenarios are solvable, energy budget included.
- **Budget curves** from the single-seed runs: `viz/figures/budget_grid20.png`. Symbolic starts behind (3.25 vs 4.00 requirements met after 1 round) but improves every round to 5.25; legacy plateaus at 4.30 after round 3.
- `docs/semantics.md` covers A1–A2.

## Phase C: runs (with a go)
2 seeds per condition, every seed's outputs kept (`docs/ablations.md`). **All symbolic runs first, then legacy.**
- Run: B2, the retry sweep R1–R5, S1, S4, S5, V1, V2, D7, and U1 (UUV with the energy requirement). Results: `out/results/ablations/summary/SUMMARY.md`.
- Not run: the legacy conditions B1, L1, L2 (lower priority than the symbolic results).
- Command: `python src/run_ablation.py <conditions>`. Smoke-test a batch first with `--seeds 1 --set run.limit=1 --out-root out/results/smoke` (then delete that directory).
- Open: further ablations (e.g. a stronger model) once the story is settled.

## "Why an LLM?" (Asher to write; inputs)
Marsha and Aren asked for a reason the LLM goes *beyond* PRISM: ~3 paragraphs plus early evidence. Candidates:
- **Expressiveness gaps.** Requirements or structure PRISM can't express, or only through awkward or approximate encodings (Aren). Example: the UUV paper's three-phase mission structure, of which we model one phase.
- **Model engineering.** Start from the problem description. The LLM builds the PRISM model, PRISM's counterexamples drive revisions, and the revisions double as a justification of the model (Marsha).
- **Restricted observation.** Memoryless observation-based policies are hard to synthesize, and PRISM's POMDP engine rejected our obstacle requirements outright.

Notes:
- An **energy budget is expressible in PRISM** (the paper's Table 2 uses rewards). A3 closes a gap in *our core*, not in PRISM, so it doesn't support the expressiveness argument by itself.
- The ceilings make the synthesis question sharper: PRISM's optimum reaches 1.0 on almost every gridworld requirement.
