# Results notes for the SEAMS evaluation section

Built on 2026-10-19 by `viz/paper_results.py` (from `PRISM-Guided-Learning/`: `python viz/paper_results.py`). The script only reads existing runs and makes no LLM or PRISM calls. Two items needed new PRISM checks of existing lists, made beforehand (no LLM calls): F3's budget curves and T4's Pac-Man NoRepair row. Their entries say how. Tables are in `out/paper/tables/`, figures in `out/paper/figures/`. File names use `_` where the LaTeX labels use `:` (`tab_rq1-coverage.tex` is `tab:rq1-coverage`).

`sections/evaluation.tex` is not in this repository, so each `.tex` file follows the columns the brief lists. Each file is a tabular body: header row and booktabs rules, with no `\begin{tabular}`.

**Split.** Every number below is a **validation** result. No sealed test target was read: grids 12-19 of `grid_20_large`, the 10x10-14x14 family members, Pac-Man horizons 8-30 and MER validation ids 0-8, as in `configs/experiments/rq1_comparison.yaml`. The sequential study uses validation family members and the 20 grids of `grid_20_large`. Each CSV has a `split` column.

**Loop version.** "Core loop" means the rebuilt loop of Oct 16 (the `port_*` conditions) as configured in the RQ1 cells:
- random grids and families with the Oct 17 gridworld feedback fix (`feedback.history`, `feedback.words`, `prompt.compact`, strategy `prompt.hint`): `port_ts_history`, `port_fam_history`;
- Pac-Man with verifier narrowing: `port_pacman_narrow_core`;
- MER: `port_mer`, `port_mer_n` (the fix's keys are gridworld-only).

Every table holds only core-loop runs except where a line says otherwise.

## Configuration mapping

| Paper name | Repository |
|---|---|
| `Full` | RQ1 cell "Ours, Haiku 5.5": `core/port_ts_history`, `port_fam_history`, `port_pacman_narrow_core`, `port_mer`, `port_mer_n` (3 runs each). The as-in-the-paper MER variants `port_mer_one` and `port_mer_n_one` are in the CSVs only. |
| `DT-raw` | "1-2-3-Go! (reimplemented: trees over raw variables)", trained on our training sets: Pac-Man horizons 14-18, MER x-train / n = 1-3. This is not the paper's single instance; that variant is in the RQ1 report only. |
| `DT-voc` | "1-2-3-Go! with our features": the rule-language features of all actions plus the goal flags. **It has no instance constants**, while the brief says "constants + features". |
| `RL` | "PPO with our features" (features plus goal flags as the observation). |
| `Hand` | Random grids and families: "Hand-written template", variant `cautious`. Pac-Man: "Hand-written maze rules" (`flee_both_any`, `flee_maze_any`, so n = 2). **These read the maze-distance features `closer_g0`/`closer_g1`, which Full's Pac-Man runs do not see**, so on Pac-Man Hand does not use the same vocabulary. MER: no hand-written list exists. |
| `Restart` | `ablations/port_mer_resample`: `port_mer` with `planner.retry: always`, the only setting that differs. |
| `NoRepair` | The same Full runs, with the pick before repair (`norepair.json`; transfer `*_norepair_determinized.csv`). MER's was back-filled after the runs (`backfill_norepair.py`), which records only whether every training member passes. |
| `NoSP` (Hand) | The template. It never reads `closer`, so it is identical to Hand: same CSV, `template_grid_large`. |
| `NoSP` (Full) | `core/port_ts_noclose`. It is **not** the core loop: it predates the Oct 17 fix and has no strategy hint. Used only in T4c. |
| `PassFail`, `Margins` | Never run on the core loop. |
| Extended / Frozen library | `src/sequential.py` modes `growing` / `frozen` (`out/results/sequential/shift_v0/`). |

## Produced

- **T1 `tab:rq1-coverage`** (`tables/tab_rq1-coverage.{tex,csv}`). Validation, core loop.
  - Coverage is the share of a controller's held-out targets that it certifies, averaged over controllers. Families: the list's own family (20 targets). Grids 8, Pac-Man 6, MER 6 targets.
  - n per cell (lists = training sets × runs):

    | | Full | DT-raw | DT-voc | RL | Hand |
    |---|---|---|---|---|---|
    | grids | 12 | 4 | 4 | 4 | 1 |
    | families | 9 | 3 | 3 | 3 | 3 (one template counted on each family) |
    | Pac-Man | 3 | 1 | 1 | 3 seeds | 2 |
    | MER (both) | 3 | 1 | 1 | 3 seeds | – |

  - Permissive = the list as kept, `any` open: worst case over every completion (`transfer_*.csv`). Frozen = `any` resolved by the order the pick chose on the training members (`transfer_*_determinized.csv`). For Pac-Man, Full's frozen controller is the narrowed lists as kept, which are already resolved.
  - For baselines, the permissive column uses their ties-open CSV where one exists (only the maze rules). Otherwise the controller has one action per state and both columns coincide.
  - Bold marks the best mean per block and column. Ties are all bolded: Pac-Man Full = DT-raw at 4/6; MER Full = DT-raw at 6/6.
  - The timeout column is the share of (controller, target) checks that hit PRISM's 900 s timeout: **0 everywhere**. No check errored either.
  - Inconclusive is shown as "--": see Missing. MER across x and across n are two blocks; the as-in-the-paper single-instance variants are in the CSV only.
- **T1 `tab:rq1-coverage-llm`**. Only Claude Haiku 5.5 has core-loop runs, so the table has one row per domain, equal to T1's Full rows. Sonnet is under Missing.
- **F1 `fig:rq1-size`** (`figures/fig_rq1-size.{pdf,csv}`).
  - Panels: random grids (side length), families (side length), Pac-Man (game length = `MAX_STEPS`), MER across n (number of requests).
  - The dashed line marks the largest training size: 11, 8, 18, 3.
  - Full uses its **permissive** lists, except on Pac-Man, which has no permissive form, so it uses the frozen narrowed lists. Baselines use their permissive controller as in T1. Put this in the caption.
  - Lines are dodged horizontally by about 1% of the axis so coinciding lines stay visible. On Pac-Man, Full and DT-raw coincide; on MER across n, Full and DT-raw coincide at 1, and DT-voc and RL at 0.
  - MER across x is not plotted: x is a probability, not a size.
- **T1b** (`tables/tab_rq1-misses.csv`). For every held-out target a frozen Full list does not certify: the smallest margin over the requirements (negative = shortfall) and which requirement it is.
  - Families: 23 misses. 13 are within 0.05 of a threshold (median −0.046), mostly an `avoid_moving` requirement, so near misses.
  - Random grids: 7 misses, none within 0.05 (median −0.099, mostly `complete_sequence`).
  - Pac-Man: 6 misses, crash 0.13-0.175 over the threshold (horizons 25 and 30).
  - The `mer_one` misses are near (median −0.020); the `mer_n_one` misses are not.
- **T2 `tab:rq2-decisions`** (`tables/tab_rq2-decisions.{tex,csv}`). Grid worlds only. Sequences s1-s3 of `configs/sequential/shift_v0.yaml`: 12 episodes each, 36 per setting. The sequences were chosen by hand, not drawn at random:
  - s1: families first, then a shift to random grids;
  - s2: interleaved;
  - s3: mixed.

  They use validation family members (sizes 10-14) and the 20 random grids of `grid_20_large` (`sequential_v0.csv` ids 180-199, sizes 9-13). The seed library was never trained on these grids. Ids 180-191 are the RQ1 grid runs' training grids, ids 192-199 its validation grids.

  The seed library holds the 3 frozen lists of `core/port_fam_history/seed_none_1` (Claude Haiku 5.5, `any` fixed by `determinize.py`). Generation runs the core families condition, also with Haiku 5.5, on the closest list's 3 smallest training instances plus the instance met.

  | | Retain | Switch | Generate | Defer | Final library |
  |---|---|---|---|---|---|
  | Extended | 26 | 5 | 3 | 2 | 5, 3, 4 |
  | Frozen | 26 | 5 | 0 | 5 | 3, 3, 3 |

  - Extended made 5 generations: 3 were certified, and 2 failed on s3's 9x9 grids 180 and 183, which were then deferred.
  - The CSV also has generations attempted per sequence.
- **T3 `tab:rq2-cost`** (`tables/tab_rq2-cost.{tex,csv}`). Median (min–max) per decision type, pooling both settings and all sequences; the decision's episode count is in parentheses. Times are wall clock on the machine below.
  - (a) Model: PRISM's reported model-construction time per certification check. Feature computation is not timed separately: see Missing.
  - (b) Certify one: wall time of one exact certification of one rule list, interval iteration or its fallback.
  - (c) Library search: the episode's certifications before any generation (the current list plus the stored ones). Not applicable for retain.
  - (d) Generation: `run_ablation.py` plus `determinize.py` wall time. It includes the loop's verification, the pick and repair, and LLM latency.
  - LLM calls and tokens (prompt + output) per generation.
  - The **Defer** row of the extended setting contains the two failed generations (generation columns). Its long certification (192 s) is a Gauss–Seidel fallback check on grid 183.
  - Extended sequences ran two at a time (s2 and s3 in one process, overlapping s1), so times include contention.
- **`rq2-cert-time-by-size.csv`**. Certification time per instance size (side lengths 9-13 occur in the sequences), with number of checks, model-construction time, states, timeouts, inconclusive outcomes and Gauss–Seidel fallbacks. Over all 118 checks: **0 timeouts and 0 inconclusive**. 2 checks fell back to Gauss–Seidel: the same list on grid 183, once per setting, failing by 0.54 both times, so the verdict stands. Median certification time grows from 3.2 s (9x9) to 4.1 s (13x13).
- **F2 `fig:rq2-extension`** (`figures/fig_rq2-extension.{pdf,csv}`). Cumulative generations summed over the 3 sequences, against position (1-12); extended vs frozen; one panel (grid worlds). Per-sequence curves are in the CSV.
- **T4 `tab:rq3-ablation`** (`tables/tab_rq3-ablation.{tex,csv}`). Validation, core loop.
  - Coverage: frozen lists, as in T1.
  - Training pass rate: the share of training members the final list passes in the worst case as the loop ends (`outputs/sample_*.json`, `members[].success`; NoRepair: `pick.norepair.members`, or the back-filled set-level flag on MER).
  - Blocks: random grids (Full, NoRepair, NoSP-Hand), families (Full, NoRepair), Pac-Man (Full, NoRepair), MER across x (Full, Restart, NoRepair), MER across n (Full, NoRepair). n = 12 / 9 / 3 / 3 / 3 lists.
  - **Pac-Man NoRepair** (added 2026-10-19, PRISM only): `src/transfer.py <run> --rules norepair_determinized --targets 0-7,9` on the 3 `port_pacman_narrow_core` runs, i.e. the pick before repair with `any` resolved by its order (`transfer_pacman_transfer_norepair_determinized.csv`).
    - It is **not narrowed**. Narrowing comes after repair in the pick, so the unrepaired list has no narrowed form. Full's Pac-Man row is the narrowed list, as in T1.
    - Result: 3 of 6 held-out horizons in every run, against Full's 4 of 6. Both pass every training member.
    - The difference is horizon 20, where the worst-case crash probability is 0.662 without repair against 0.5585 with it (threshold 0.56). Both fail at 25 and 30.
    - The three runs' lists differ in text (they were picked from different chains) but give identical values on every target, in both arms.
  - LLM calls per training set are in the CSV. NoRepair equals Full by construction: same runs, and repair uses no LLM.
  - **Restart on MER**: 59.0 calls per set against Full's 63.7. Both have the same cap (8 chains × 7 rounds); restarts stop earlier when a chain reaches the margin. It is kept in the table because the budget (the cap) is the same and the gap goes against Restart. **Asher: drop the row if "same number of calls" must mean calls actually spent.**
- **T4b** (`tables/tab_rq3-selection.csv`). Held-out targets certified per frozen Full list, split by whether the list passed every training member, core loop.
  - Only random grids have lists that did not pass: 7.56 of 8 (n = 9) passed vs 7.00 of 8 (n = 3) did not.
  - Families (17.44 of 20, n = 9), Pac-Man (4 of 6, n = 3) and both MER benchmarks (6 of 6, n = 3 each) have no failing lists. The core loop solves every training set there, so the comparison does not exist.
- **T4c** (`tables/tab_rq3-nosp.csv`). The "6.00 vs 3.25" on random grids without `closer` (`grids_noclose`: grid_20_large targets 12-19, 8 held out):
  - Hand (template; n = 1 list): 6 of 8.
  - Full: `port_ts_noclose`, 1 run × 4 training sets, n = 4 lists: 3.25 ± 0.96 of 8. That run is the core loop **before** the Oct 17 gridworld feedback fix, without the strategy hint, so it is not comparable with T4's Full rows.
- **F3 `fig:rq3-budget`** (`figures/fig_rq3-budget.{pdf,csv}`; added 2026-10-19, PRISM only, no LLM calls). MER across x, validation targets 0-8 (6 held out), core loop. 3 runs each of Full (`core/port_mer`) and Restart (`ablations/port_mer_resample`), one training set per run.
  - **How each point is made.** For round budget R = 1-7, every one of the run's 8 chains is cut to its first R rounds. The run's own pick with repair is rerun on what is left (as `src/equal_spend.py` does at one budget). `any` is resolved by the pick's order, as in T4's frozen lists. The list is then certified by `transfer.py`'s check.
  - x is the LLM calls of those rounds summed over chains, re-asks included. Each point is the runs' mean calls and mean coverage at one R; the band is the range over runs.
  - **Tool change.** `src/budget_transfer.py` refused multi-chain runs; it now has a chain mode (`chain_picks`, `chain_budget_table`), which writes `<run>/budget_transfer_chains.csv`.
  - **Check.** At R = 7 the re-pick reproduces every run's stored `determinized.json` list, so its rows are the run's own transfer CSV. No check timed out or errored.
  - **Restart's spend** is 59.0 calls (its 3 runs' totals: 64, 53, 60), the same as T4.
    - At 59 calls (each run's last R within 59), Full certifies 6, 6 and 3 of 6, a mean of 0.83. Restart certifies 6 of 6 in every run.
    - Full needs its full 63.7 calls to reach 6 of 6 in all runs.
  - **The result goes against Full.** Restart's mean is at or above Full's at every R (R = 1: 0.50 vs 0.33; R = 7: 1.00 each). Every first answer certifies 3 of 6, except Full seed 3's at 0. Restart first reaches 6 of 6 at 18-57 calls, Full at 18-59.
  - **Not monotone.** Full seed 3 certifies 6 of 6 at R = 2-4, 3 at R = 5-6 and 6 again at R = 7. The pick ranks lists by their training score, not by held-out coverage, so a later list that scores better on training can certify fewer held-out targets. Say so in the caption.
  - With 3 runs per arm and one training set each, no difference here is significant. The planned feedback-vs-restart test (`rq2_restarts`) is on families, on the pre-fix loop.

## Missing

- **T1, share inconclusive.** RQ1's held-out checks (`src/transfer.py`) use the loop solver (Gauss–Seidel) and re-check with the exact solvers only within 0.03 of a threshold. They record the values, not the interval, so an interval containing the threshold cannot be counted. Smallest experiment: re-certify every RQ1 controller on its held-out targets with `sequential.certify`'s rule (interval iteration, inconclusive band of 1e-9). PRISM only, about 1.5k checks.
- **T1, Pac-Man permissive Full.** Narrowing (`rules.early_narrow`) is applied after `any` is resolved, so no permissive narrowed list exists. Smallest experiment: certify the pick's list before resolution and narrowing with `any` open (3 runs × 6 targets, PRISM only). This is a different controller from the frozen one.
- **T1, Hand on MER.** No hand-written MER rule list. Smallest experiment: write one over the MER vocabulary and certify it on the 6 + 6 validation targets.
- **T1 `tab:rq1-coverage-llm`, any LLM other than Haiku 5.5 on the core loop.** Sonnet 5.5 exists only on the previous loop. Smallest experiment: 3 Sonnet runs of each core condition (needs Asher's go-ahead).
- **T2/T3/F2, Pac-Man and MER.** Sequential studies exist only for grid worlds. Smallest experiment: a `configs/sequential/` study per domain seeded from its core run, frozen mode first (PRISM only) and growing after (LLM).
- **T3 (a), feature-computation time.** Features are computed in Python when the model is rendered, and that time is not recorded; only PRISM's model-construction time is. Smallest experiment: time `domain.model()` per check in `sequential.certify`.
- **T3b, synthesis time.** Not measured. What exists:
  - For grid worlds the repository runs PRISM's multi-objective query (`multi(...)`, exact LP, sparse engine) on the bare MDP, but only as a feasibility check (`src/ceilings.py`). It confirms one scheduler meets all thresholds jointly on all 20 `grid_20_large` grids. It records no time, and no strategy is exported.
  - Such a scheduler may be randomized and use memory, and it observes the full state, so it is not comparable with a rule list.
  - LP cannot handle step-bounded requirements (UUV), so there the query is undecided.
  - Pac-Man has one requirement, so synthesis is a single `Pmin` query. MER's requirements (one `err` bound) are likewise single-objective.

  Smallest experiment: time `multi(...)` with `-exportstrat` (or the single-objective query) on the bare MDP of each held-out instance, next to `rq2-cert-time-by-size.csv`. Pac-Man's bare model has 8.3M states at horizon 30 and cannot be built at 50, while a frozen list's model has 29k states at 30. That size gap is the cost argument and is already documented (`docs/experiments.md`).
- **T4, Restart on families, random grids, Pac-Man and MER across n (core loop).** Only MER across x has a core-loop restart arm. Families has `port_fam_resample`, but that is the restart arm of the pre-fix loop (below). Smallest experiment: `port_fam_history` with `planner.retry=always`, 3 Haiku runs; likewise `port_ts_history`.
- **T4, PassFail and Margins.** Not run on the core loop. Smallest experiment: two feedback-content switches on `port_fam_history` and `port_mer` (3 runs each). The previous loop's "results table only" arm is the closest to Margins (below). Nothing like PassFail was ever run.
- **T4, NoSP (Full) on the core loop.** `port_ts_noclose` predates the fix. Smallest experiment: `port_ts_history` with `closer` hidden, 3 runs. Families: none.
- **F3, families and other benchmarks.** Budget curves exist only for MER across x, the only benchmark with a core-loop restart arm (see Produced). Smallest experiment: the families restart arm above, then `src/budget_transfer.py` on both arms (PRISM only).

## Previous loop only

These numbers exist only for the previous loop, or the pre-fix version of the new one, and do not go in the tables. Values are certified per list, validation.

- **Selection evidence**: 14.2 vs 5.2 (families), 4.0 vs 1.8 and 6 vs 0 (DECISIONS.md, previous loop). For the core loop see T4b.
- **Feedback vs restarts on families** (`rq2_restarts`; `port_fam` vs `port_fam_resample`, 3 runs each). This is the new loop *before* the Oct 17 gridworld feedback fix and without the strategy hint:
  - feedback 17.44 of 20 vs restarts 13.67;
  - feedback at the restarts' dollar spend: 15.67 (`src/equal_spend.py`).
- **"5.00 at the restarts' spend" (MER across x), clarified.**
  - `src/equal_spend.py` cut every chain of each of the 3 core `port_mer` runs to its first R rounds, with R the most that keeps the training set's **dollar** spend within the restarts' mean ($0.283 per set, against $0.337 for feedback). R = 5 of 7 in seed 1.
  - It then re-ran that run's own pick with repair, without LLM calls.
  - The three resulting lists certify 3, 6 and 6 of 6, mean 5.00, range 3-6.
  - It is an equal-cost comparison, not an equal-calls one: restarts send shorter prompts for about as many calls. The 6 vs 6 tie in T4 is at the full budget.
- **Previous loop, Haiku 5.5** (`rq2_feedback`, `rq2_features`, `rq2_verifier_output`, `rq2_prompting`, against its reference of grids 6.25/8, families 9.67/20, Pac-Man 2.40/6):
  - restarts only: 5.50, 9.20, 2.20;
  - no `closer`: grids 3.50;
  - "results table only" feedback (closest to Margins): 6.08, 7.89, 2.33;
  - prompting variants: `rq2_prompting`.
- **Sonnet 5.5, previous loop** (`rq2_previous_loop`): grids 6.67/8, families 14.00/20, Pac-Man 2.33/6, MER across x 5.00/6, across n 6.00/6, grids without `closer` 2.92/8.

## Setup facts

- **PRISM**: 4.10.1, explicit engine by default (`PrismRunner`), 4 GB Java heap per process, at most 8 concurrent processes.
  - **Inside the loop** ("fast"): Gauss–Seidel value iteration (`prism.method: gaussseidel`, PRISM's default convergence criterion), falling back to modified policy iteration. This is *not* plain value iteration, which fails on periodic chains.
  - **Final check of each run's lists** on the training members: interval iteration (`-intervaliter -topological -epsilon 1e-9`), falling back to Gauss–Seidel at 1e-12.
  - **RQ1 held-out certification** (`transfer.py`): loop solver, with an exact re-check when a worst case is within 0.03 of its threshold.
  - **Sequential study**: interval iteration on every check. Certified only if the smallest margin is > 1e-9; inconclusive within ±1e-9 or after the Gauss–Seidel fallback with a margin ≥ −1e-9.
  - **Timeout**: 900 s per PRISM call (`prism.timeout_s`).
  - **Repair's "fast exact checker"** (`Domain.fast_checker`): a numpy model of the instance giving each requirement's worst-case value of a rule list. Gridworld: `baselines/gridworld_check.GridChecker`; MER and Pac-Man: the PRISM-exported explicit model in `baselines/explicit_mdp.RuleChecker`. Picked lists are re-checked with PRISM.
- **Loop** (core configs):

  | | K (chains) | Rounds per chain | Stopping margin | Candidates (`pick_top`) | Repair (steps × neighbours) |
  |---|---|---|---|---|---|
  | Families | 6 | 7 | 0.05 | 8 | 40 × 60 |
  | Random grids | 8 | 7 | 0.05 | 8 | 40 × 60 |
  | Pac-Man, MER | 8 | 7 | 0.002 | 6 | 25 × 40 |

  - A chain stops when every member passes with worst-case margin > the stopping margin.
  - Up to 2 re-asks per round for invalid answers; output capped at 64 rules and 200-character conditions.
  - The pick: the `pick_top` best distinct lists of every round of every chain, each repaired, then the one certified on most members with the largest margin.
  - Pac-Man narrowing: `rules.early_narrow`.
- **LLM**: `anthropic/claude-haiku-5.5` through OpenRouter, provider pinned to `anthropic`.
  - Thinking on; 16,384 context and output tokens; temperature = model default (`null`); no seed (unseeded repeats).
  - Structured output: JSON schema with the actions as an enum.
  - Priced at $0.10 / $0.50 per M prompt / output tokens.
- **Repetitions**: 3 unseeded runs per training set.
- **Training sets per domain**:
  - random grids: 4 sets, each a 9x9, 10x10 and 11x11 grid;
  - families: 3, one per family, members 6x6-8x8;
  - Pac-Man: 1 (horizons 14, 16, 18);
  - MER across x: 1 (x = 0.01, 0.05, 0.13 at n = 10);
  - MER across n: 1 (n = 1, 2, 3 at x = 0.89).
- **Held-out sizes**:
  - random grids 12x12-13x13 (8);
  - families 10x10-14x14 (20 per family);
  - Pac-Man horizons 8, 10, 12, 20, 25, 30;
  - MER across x: x = 0.03, 0.08, 0.21, 0.34 and n = 1, 20;
  - MER across n: n = 4, 5, 6, 8, 10, 15.
- **How held-out instances were generated and checked solvable**:
  - Random grids: `datagen.py` with a BFS check of the ordered goal path. PRISM's joint multi-objective query confirms all 20 `grid_20_large` grids are jointly feasible (`out/results/ceilings/`).
  - Families: `families.py`; each kept family is jointly solvable at every size checked (`docs/experiments.md`).
  - Pac-Man: optimum 0.5511 ≤ threshold 0.56 at every horizon up to 30.
  - MER: thresholds are the per-instance optimum −5%, so each is achievable.
- **Validation vs sealed test**:
  - Sealed (`configs/transfer/default.yaml`): `grid_test_large.csv`; family members 15x15-20x20 (ids 36-59, 96-119, 156-179); Pac-Man horizons 50 and 100; MER ids 9-11.
  - **Every number here is validation.**
- **Hardware**: AMD Ryzen 7 9800X3D (8 cores, 16 threads), 31 GB RAM, Windows 11.
  - Runs were parallel: several hosted-LLM jobs at once, each with up to `run.workers` PRISM processes.
  - Wall-clock times are therefore under load; tokens are the cost measure.
- **Baselines compiled for certification**:
  - Gridworld trees and PPO: the frozen controller's action in every state of the target, written as rules over x, y, the goal flags and `obs_idx`, composed with the model like any rule list (`decision_tree.policy_rules`).
  - On MER and Pac-Man, trees compile to rules exactly. Networks and tables are certified on the target's model restricted to the policy, exported as PRISM explicit files and checked with `-importmodel`.
- **RL**: PPO (`src/baselines/ppo.py`).
  - Network: MLP 2×64 tanh with policy and value heads; Adam 3e-4; rollouts 128 × 32 environments; GAE λ = 0.95; grad-norm clip 0.5.
  - γ = 0.99 on gridworld (shaped reward: +1 per goal in order, −1 and end on an early goal or collision, −0.01 per step); γ = 1 on the explicit models (undiscounted requirement).
  - Budget: 2M environment steps, seed 0, one policy per training set on gridworld; 1M steps × 3 seeds on Pac-Man and MER.
  - The frozen policy takes its most likely action.
- **Hand rule lists**:
  - The gridworld template (`domains/gridworld/data/template_policy.py`) was committed on 2026-10-08, before the first transfer runs (Oct 14), but after earlier per-grid LLM rule sets existed.
  - The Pac-Man maze rules were committed on 2026-10-14 with the transfer search.
  - Who wrote them and how long it took is not recorded in the repository. **Asher to fill in.**

## Blocked

- **E4 (third LLM).** `openai/gpt-6-luna` (OpenRouter, provider pinned to `openai`, $0.10 / $0.50 per M) and the fallback `openai/gpt-5-mini` both fail every call with HTTP 400: "Invalid schema for response_format 'answer': 'additionalProperties' is required to be supplied and to be false". OpenAI's strict structured output needs `additionalProperties: false` in the loop's JSON schema (`core/planner.py` `rule_schema`), which is a code change, so per the ground rules the model is dropped (two failed smoke tests). Smoke runs deleted; their error logs were kept outside the repository.
- **E5 (PassFail).** Not a config switch: `planner.feedback` is `blame | table | failures`, and `table` is the results table (closer to Margins). Skipped.

## In progress (2026-10-19, session cut by a usage limit)

- E1 `ablations/port_fam_history_resample`, E2 `ablations/port_ts_history_resample` and `port_pacman_narrow_core_resample`, E3 `ablations/port_*_sonnet` (5 core conditions × 3; the Sonnet smoke test on families set 0 passed, structured output and re-asks both work). OpenRouter lists Sonnet 5.5 at $2 / $10 per M; `rq1_comparison.yaml` says 3 / 15.
- P1: `src/recertify.py` (sequential.certify's rule on every check) writes `transfer_ii_*.csv` next to each transfer CSV, and `sizes.json` for refitted baselines. Its first checks (MER run 1, grid runs 1-3) give the same verdicts as the Gauss–Seidel CSVs. `viz/paper_results.py` reads these by default (`--checks gs` gives the old tables, unchanged).
- P6 frozen mode: `configs/sequential/shift_v0_lib2.yaml`, `shift_v0_lib3.yaml`. P8: `configs/baselines/ppo_features_*_seed{1,2}.yaml`.
- Done: P3 (`tab_rq2-episodes.csv`, `tab_rq2-example.csv`), P4 (`tab_rq2-generation-cost.csv`), P7 rule-list sizes (`tab_rq1-sizes.csv`; trees and PPO wait for P1's `sizes.json`). A3 was already done (F3, MER panel with Restart's spend marked).
- Not started: P2 (sealed; job list ready), P5 (`src/budget_ii.py` written, not yet run), A1/A2/A4 regeneration after P1, A5.
