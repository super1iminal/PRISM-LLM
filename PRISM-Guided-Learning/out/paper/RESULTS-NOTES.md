# Results notes for the SEAMS evaluation section

Built on 2026-10-19 by `viz/paper_results.py` (from `PRISM-Guided-Learning/`: `python viz/paper_results.py`). The script only reads existing results and makes no LLM or PRISM calls. Tables are in `out/paper/tables/`, figures in `out/paper/figures/`. File names use `_` where the LaTeX labels use `:` (`tab_rq1-coverage.tex` is `tab:rq1-coverage`).

`sections/evaluation.tex` is not in this repository, so each `.tex` file follows the columns the brief lists. Each file is a tabular body: header row and booktabs rules, with no `\begin{tabular}`.

**Split.** Every number below is a **validation** result: grids 12-19 of `grid_20_large`, the 10x10-14x14 family members, Pac-Man horizons 8-30 and MER validation ids 0-8, as in `configs/experiments/rq1_comparison.yaml`. The sequential study uses validation family members and the 20 grids of `grid_20_large`. Each CSV has a `split` column. The sealed test pass (P2) has its own tables, `tab:rq1-coverage-sealed` and `tab:rq3-ablation-sealed` (below).

**Certification rule.** Every held-out check in the tables uses one rule: interval iteration at 1e-9 (capped at 10^6 iterations), PRISM's linear programming where it stalls, then Gauss–Seidel at 1e-12 (no error bound). A check is certified only if a sound solver finished and the smallest margin is beyond its band (1e-9 for interval iteration, 1e-3 for LP); inconclusive within the band or when only Gauss–Seidel finished; "not certifiable within limits" when no solver finishes (PRISM's 900 s timeout) or the model cannot be built. The checks are `src/recertify.py`'s, merged per controller into `transfer_exact_*.csv` by `python src/recertify.py combine out/results` (interval iteration's verdict where it converged, else LP's). `python viz/paper_results.py --checks gs` gives the tables from the loop solver's checks of this morning. The only exception is T4c (below).

**Loop version.** "Core loop" means the rebuilt loop of Oct 16 (the `port_*` conditions) as configured in the RQ1 cells:
- random grids and families with the Oct 17 gridworld feedback fix (`feedback.history`, `feedback.words`, `prompt.compact`, strategy `prompt.hint`): `port_ts_history`, `port_fam_history`;
- Pac-Man with verifier narrowing: `port_pacman_narrow_core`;
- MER: `port_mer`, `port_mer_n` (the fix's keys are gridworld-only).

Every table holds only core-loop runs except where a line says otherwise.

## Restart and OneShot (validation, fixed rule)

Certified coverage of the held-out targets per list, mean ± sd over lists (`tab_rq3-ablation`), with LLM calls per training set.

| | Full | Restart | OneShot | NoRepair |
|---|---|---|---|---|
| Random grids | 92.7 ± 11.3 (55.8 calls) | 79.2 ± 9.7 (56.0) | 74.0 ± 3.6 (1.0) | 86.5 ± 6.4 |
| Families | 87.2 ± 12.0 (36.4) | 73.3 ± 26.9 (42.0) | 43.9 ± 35.8 (1.0) | 67.8 ± 39.5 |
| Pac-Man | 66.7 ± 0.0 (49.3) | 33.3 ± 28.9 (59.0) | 38.9 ± 9.6 (1.0) | 50.0 ± 0.0 |
| MER across x | 100 ± 0 (63.7) | 100 ± 0 (59.0) | 0 ± 0 (1.0) | 100 ± 0 |
| MER across n | 100 ± 0 (60.0) | -- (saturated, not run) | 33.3 ± 57.7 (1.0) | 100 ± 0 |

- **Restart** = the core condition with `planner.retry: always` (every answer a fresh sample of the initial prompt), same chains × rounds, stop rule and pick with repair: `ablations/port_{ts_history,fam_history,pacman_narrow_core}_resample` (3 Haiku 5.5 runs each, generated 2026-10-19) and `ablations/port_mer_resample`. Training pass rate: grids 75.0 %, families 100 %, Pac-Man 33.3 % (1 of 3 runs), MER 100 %.
- **OneShot** = each core run's first valid answer of its first chain (chain 0, round 1; no answer needed a re-ask), `any` resolved as the pick resolves it (`core.pick.resolved`: the best action order on the training members), no selection, no repair: `src/oneshot.py` (`<run>/oneshot.json`, `transfer_exact_<dataset>_oneshot.csv`). Same 3 runs per benchmark as Full. Its list was never checked on the training members, so it has no pass rate.

## Configuration mapping

| Paper name | Repository |
|---|---|
| `Full` | RQ1 cell "Ours, Haiku 5.5": `core/port_ts_history`, `port_fam_history`, `port_pacman_narrow_core`, `port_mer`, `port_mer_n` (3 runs each). The as-in-the-paper MER variants `port_mer_one` and `port_mer_n_one` are in the CSVs only. |
| `DT-raw` | "1-2-3-Go! (reimplemented: trees over raw variables)", trained on our training sets: Pac-Man horizons 14-18, MER x-train / n = 1-3. This is not the paper's single instance; that variant is in the RQ1 report only. |
| `DT-voc` | "1-2-3-Go! with our features": the rule-language features of all actions plus the goal flags. **It has no instance constants**, while the brief says "constants + features". |
| `RL` | "PPO with our features" (features plus goal flags as the observation). Seeds 0-2 on every benchmark (`configs/baselines/ppo_features_*_seed{1,2}.yaml` add seeds 1-2 on the grid worlds). |
| `Hand` | Random grids and families: "Hand-written template", variant `cautious`. Pac-Man: "Hand-written maze rules" (`flee_both_any`, `flee_maze_any`, so n = 2). **These read the maze-distance features `closer_g0`/`closer_g1`, which Full's Pac-Man runs do not see**, so on Pac-Man Hand does not use the same vocabulary. MER: no hand-written list exists. |
| `Restart` | `ablations/port_{ts_history,fam_history,pacman_narrow_core}_resample` and `ablations/port_mer_resample`: the core condition with `planner.retry: always`, the only setting that differs. |
| `OneShot` | The core runs' first valid answer of chain 0, `any` resolved by the pick's rule, no selection or repair (`src/oneshot.py`). |
| `NoRepair` | The same Full runs, with the pick before repair (`norepair.json`; `*_norepair_determinized.csv`). MER's was back-filled after the runs (`backfill_norepair.py`), which records only whether every training member passes. |
| `NoSP` (Hand) | The template. It never reads `closer`, so it is identical to Hand: same CSV, `template_grid_large`. |
| `NoSP` (Full) | `core/port_ts_noclose`. It is **not** the core loop: it predates the Oct 17 fix and has no strategy hint. Used only in T4c. |
| Extended / Frozen library | `src/sequential.py` modes `growing` / `frozen` (`out/results/sequential/shift_v0/`; frozen mode also from two more seed libraries, `shift_v0_lib2`, `shift_v0_lib3`). |

## Produced

- **T1 `tab:rq1-coverage`** (`tables/tab_rq1-coverage.{tex,csv}`). Validation, core loop.
  - Coverage is the share of a controller's held-out targets that it certifies, averaged over controllers. Families: the list's own family (20 targets). Grids 8, Pac-Man 6, MER 6 targets.
  - n per cell (lists = training sets × runs; RL = training sets × seeds):

    | | Full | DT-raw | DT-voc | RL | Hand |
    |---|---|---|---|---|---|
    | grids | 12 | 4 | 4 | 12 | 1 |
    | families | 9 | 3 | 3 | 9 | 3 (one template counted on each family) |
    | Pac-Man | 3 | 1 | 1 | 3 | 2 |
    | MER (both) | 3 | 1 | 1 | 3 | – |

  - Permissive = the list as kept, `any` open: worst case over every completion. Frozen = `any` resolved by the order the pick chose on the training members (`*_determinized`). For Pac-Man, Full's frozen controller is the narrowed lists as kept, which are already resolved.
  - For baselines, the permissive column uses their ties-open checks where they exist (only the maze rules). Otherwise the controller has one action per state and both columns coincide.
  - Bold marks the best mean per block and column. Ties are all bolded: Pac-Man Full = DT-raw at 4/6; MER Full = DT-raw at 6/6.
  - Inconclusive and timeout columns: shares of the frozen checks. Full, DT and Hand: 0 everywhere. RL: random grids 1.0 % inconclusive and 2.1 % not certifiable within limits; families as in the CSV.
  - **RL is PPO over seeds.** The rechecks retrain PPO, and training does not reproduce (the retrained networks' worst cases differ from earlier ones by up to 0.82), so RL is the mean over seeds × training sets, each network certified in one recheck. Random grids 66.7 ± 17.9 %, families 61.1 ± 44.8 %, Pac-Man 50.0 %, MER 0 %.
  - MER across x and across n are two blocks; the as-in-the-paper single-instance variants are in the CSV only.
- **T1 `tab:rq1-coverage-llm`**. Haiku 5.5 only (other LLMs dropped for now, below), equal to T1's Full rows.
- **T1, sealed test `tab:rq1-coverage-sealed`** (`tables/tab_rq1-coverage-sealed.{tex,csv}`; P2). T1's layout plus a column n (controllers per cell), on the sealed targets with the same rule: random grids `grid_test_large.csv` (42 grids, sizes 9-20), family members 15x15-20x20 (24 per family, each list on its own family), MER ids 9-11 (3 per benchmark). `split` = "sealed test". No check is inconclusive or timed out.
  - Frozen: random grids Full 91.1 ± 7.5 % (validation 92.7), DT-raw 1.2 %, DT-voc 57.7 %, Hand 52.4 %; families Full 89.4 ± 9.8 % (87.2), DT-raw 0 %, DT-voc 59.7 %, Hand 29.2 %; MER across x Full and DT-raw 100 %, DT-voc 33.3 %; across n Full and DT-raw 100 %, DT-voc 0 %.
  - Not checked: RL (PPO; the sealed checks need the validation networks, which are not stored) and OneShot. Pac-Man is left out (Missing).
- **T4, sealed test `tab:rq3-ablation-sealed`** (`tables/tab_rq3-ablation-sealed.{tex,csv}`; P2). Coverage per list of Full, Restart and NoRepair on the same sealed targets, with n; training pass rates and LLM calls are T4's. Random grids: Full 91.1 ± 7.5 %, Restart 77.0 ± 12.0 %, NoRepair 85.7 ± 5.9 %; families: 89.4 ± 9.8, 75.5 ± 28.9, 71.8 ± 36.0 %; MER across x 100 % each; across n Full and NoRepair 100 %. No inconclusive checks. Pac-Man is left out.
- **F1 `fig:rq1-size`** (`figures/fig_rq1-size.{pdf,csv}`).
  - Panels: random grids (side length), families (side length), Pac-Man (game length = `MAX_STEPS`), MER across n (number of requests).
  - The dashed line marks the largest training size: 11, 8, 18, 3.
  - Full uses its **permissive** lists, except on Pac-Man, which has no permissive form, so it uses the frozen narrowed lists. Baselines use their permissive controller as in T1. Put this in the caption.
  - Lines are dodged horizontally by about 1% of the axis so coinciding lines stay visible. On Pac-Man, Full and DT-raw coincide; on MER across n, Full and DT-raw coincide at 1, and DT-voc and RL at 0.
  - MER across x is not plotted: x is a probability, not a size.
- **T1b** (`tables/tab_rq1-misses.csv`). For every held-out target a frozen Full list does not certify: the smallest margin over the requirements (negative = shortfall), which requirement it is, and the outcome. Every miss is a failure, none inconclusive.
  - Families: 23 misses. 13 are within 0.05 of a threshold (median −0.046), mostly an `avoid_moving` requirement, so near misses.
  - Random grids: 7 misses, none within 0.05 (median −0.099, mostly `complete_sequence`).
  - Pac-Man: 6 misses, crash 0.13-0.175 over the threshold (horizons 25 and 30).
  - The `mer_one` misses are near (median −0.020); the `mer_n_one` misses are not.
- **T2 `tab:rq2-decisions`** (`tables/tab_rq2-decisions.{tex,csv}`). Grid worlds only. Sequences s1-s3 of `configs/sequential/shift_v0.yaml`: 12 episodes each, 36 per setting. The sequences were chosen by hand, not drawn at random:
  - s1: families first, then a shift to random grids;
  - s2: interleaved;
  - s3: mixed.

  They use validation family members (sizes 10-14) and the 20 random grids of `grid_20_large` (`sequential_v0.csv` ids 180-199, sizes 9-13). The seed library was never trained on these grids. Ids 180-191 are the RQ1 grid runs' training grids, ids 192-199 its validation grids.

  The seed library holds the 3 frozen lists of `core/port_fam_history/seed_none_1` (Claude Haiku 5.5, `any` fixed by the pick's order, `determinized.json`). Generation runs the core families condition, also with Haiku 5.5, on the closest list's 3 smallest training instances plus the instance met.

  | | Retain | Switch | Generate | Defer | Final library |
  |---|---|---|---|---|---|
  | Extended | 26 | 5 | 3 | 2 | 5, 3, 4 |
  | Frozen | 26 | 5 | 0 | 5 | 3, 3, 3 |
  | Frozen, seed library of core run 2 | 28 | 3 | 0 | 5 | 3, 3, 3 |
  | Frozen, seed library of core run 3 | 20 | 10 | 0 | 6 | 3, 3, 3 |

  - Extended made 5 generations: 3 were certified, and 2 failed on s3's 9x9 grids 180 and 183, which were then deferred.
  - Reuse alone serves 31, 31 and 30 of 36 episodes with the three seed libraries (P6: `configs/sequential/shift_v0_lib{2,3}.yaml`, frozen mode only, same sequences).
  - The CSV also has generations attempted per sequence.
- **`tab_rq2-episodes.csv`** (P3). One row per (setting, sequence, episode): the map and its size, the decision, the list that served it and where that list came from. In the extended setting 22 episodes are served by a seed list, 3 by a list generated in the episode, and 9 by a list generated earlier in the same sequence (s1's `gen_t4` 7 times, `gen_t2` once; s3's `gen_t10` once). Frozen: 31 by seed lists.
- **`tab_rq2-example.csv`** (P3, the Approach's running example, instance 142). See "For the paper".
- **T3 `tab:rq2-cost`** (`tables/tab_rq2-cost.{tex,csv}`). Median (min–max) per decision type, pooling both settings and all sequences; the decision's episode count is in parentheses. Times are wall clock on the machine below.
  - (a) Model: PRISM's reported model-construction time per certification check. Feature computation is not timed separately: see Missing.
  - (b) Certify one: wall time of one exact certification of one rule list, interval iteration or its fallback.
  - (c) Library search: the episode's certifications before any generation (the current list plus the stored ones). Not applicable for retain.
  - (d) Generation: `run_ablation.py` plus `determinize.py` wall time. It includes the loop's verification, the pick and repair, and LLM latency.
  - LLM calls and tokens (prompt + output) per generation.
  - The **Defer** row of the extended setting contains the two failed generations (generation columns). Its long certification (192 s) is a Gauss–Seidel fallback check on grid 183.
  - Extended sequences ran two at a time (s2 and s3 in one process, overlapping s1), so times include contention.
- **`rq2-cert-time-by-size.csv`**. Certification time per instance size (side lengths 9-13 occur in the sequences), with number of checks, model-construction time, states, timeouts, inconclusive outcomes and Gauss–Seidel fallbacks. Over all 118 checks: **0 timeouts and 0 inconclusive**. 2 checks fell back to Gauss–Seidel: the same list on grid 183, once per setting, failing by 0.54 both times, so the verdict stands. Median certification time grows from 3.2 s (9x9) to 4.1 s (13x13).
- **`tab_rq2-generation-cost.csv`** (P4; per training set: `-per-set`). Median and range per condition of wall-clock minutes, LLM and PRISM seconds, LLM calls, tokens and dollars. LLM and PRISM seconds are summed over every round of every chain; chains run concurrently, so they overlap each other and can exceed the wall clock. Median per training set: random grids 19.4 min, 56 calls, $0.093; families 11.5 min, 42 calls, $0.063; Pac-Man 21.1 min, 49 calls, $0.098; MER across x 56.1 min, 66 calls, $0.349; across n 53.9 min, 60 calls, $0.370. The sequential study's 5 generations cost $0.36 in total ($0.070, 10.9 min and 42 calls each, median).
- **F2 `fig:rq2-extension`** (`figures/fig_rq2-extension.{pdf,csv}`). Cumulative generations summed over the 3 sequences, against position (1-12); extended vs frozen; one panel (grid worlds). Per-sequence curves are in the CSV.
- **T4 `tab:rq3-ablation`** (`tables/tab_rq3-ablation.{tex,csv}`). Validation, core loop. The numbers are in "Restart and OneShot" above.
  - Coverage: frozen lists, as in T1.
  - Training pass rate: the share of training members the final list passes in the worst case as the loop ends (`outputs/sample_*.json`, `members[].success`; NoRepair: `pick.norepair.members`, or the back-filled set-level flag on MER). OneShot has none.
  - Blocks: random grids (Full, Restart, NoRepair, OneShot, NoSP-Hand), families, Pac-Man and MER across x (Full, Restart, NoRepair, OneShot), MER across n (Full, NoRepair, OneShot). n = 12 / 9 / 3 / 3 / 3 lists.
  - **Pac-Man NoRepair**: the pick before repair with `any` resolved by its order. It is **not narrowed**: narrowing comes after repair in the pick, so the unrepaired list has no narrowed form. Full's Pac-Man row is the narrowed list, as in T1. The difference is horizon 20, where the worst-case crash probability is 0.662 without repair against 0.5585 with it (threshold 0.56). The three runs' lists differ in text but give identical values on every target, in both arms.
  - LLM calls per training set are in the table. NoRepair equals Full by construction: same runs, and repair uses no LLM.
  - **Restart's calls**: Restart and Full have the same cap (chains × rounds); restarts stop earlier when a chain reaches the margin, or later (families 42.0 against 36.4). **Asher: drop a Restart row if "same number of calls" must mean calls actually spent.**
- **T4b** (`tables/tab_rq3-selection.csv`). Held-out targets certified per frozen Full list, split by whether the list passed every training member, core loop.
  - Only random grids have lists that did not pass: 7.56 of 8 (n = 9) passed vs 7.00 of 8 (n = 3) did not.
  - Families (17.44 of 20, n = 9), Pac-Man (4 of 6, n = 3) and both MER benchmarks (6 of 6, n = 3 each) have no failing lists. The core loop solves every training set there, so the comparison does not exist.
- **T4c** (`tables/tab_rq3-nosp.csv`). The "6.00 vs 3.25" on random grids without `closer` (`grids_noclose`: grid_20_large targets 12-19, 8 held out). **Loop solver's checks** (`checks` column): `port_ts_noclose` was not re-certified.
  - Hand (template; n = 1 list): 6 of 8.
  - Full: `port_ts_noclose`, 1 run × 4 training sets, n = 4 lists: 3.25 ± 0.96 of 8. That run is the core loop **before** the Oct 17 gridworld feedback fix, without the strategy hint, so it is not comparable with T4's Full rows.
- **F3 `fig:rq3-budget`** (`figures/fig_rq3-budget.{pdf,csv}`). One panel per benchmark, validation, core loop; the CSV has a `checks` column.
  - **MER across x** (left): 3 runs each of Full (`core/port_mer`) and Restart (`ablations/port_mer_resample`), one training set per run, held-out targets 0-8 (6). **Still the loop solver's checks** (its title says so); P5 redoes it under the fixed rule.
  - **Families** (right; P5, fixed rule): Full only, 3 runs × 3 training sets, each list on its own family's 20 held-out targets (`core/port_fam_history/seed_none_*/budget_ii_rounds.csv`, `src/budget_ii.py`). Mean coverage by round budget R = 1-7: 0.59, 0.83, 0.68, 0.87, 0.82, 0.86, 0.87 at 6.0-36.4 calls per training set. Not monotone in R (R = 3 dips), as on MER. Restart's replays are not done yet, so there is no Restart curve or spend marker here.
  - **OneShot** (diamond, both panels): the first answer of chain 0, frozen, no selection or repair, at 1 call: MER 0.00, families 0.44.
  - The K-chains replays (`budget_ii_chains.csv`, first k chains at full rounds) exist for families Full but are not plotted yet.
  - **How each point is made.** For round budget R = 1-7, every one of the run's 8 chains is cut to its first R rounds. The run's own pick with repair is rerun on what is left (as `src/equal_spend.py` does at one budget). `any` is resolved by the pick's order. The list is then certified (`src/budget_transfer.py` chain mode; under the fixed rule `src/budget_ii.py`, which also replays the first k chains, the K-chains axis).
  - x is the LLM calls of those rounds summed over chains, re-asks included. Each point is the runs' mean calls and mean coverage at one R; the band is the range over runs.
  - **Restart's spend** is 59.0 calls (its 3 runs' totals: 64, 53, 60). At 59 calls (each run's last R within 59), Full certifies 6, 6 and 3 of 6, a mean of 0.83; Restart certifies 6 of 6 in every run. Full needs its full 63.7 calls to reach 6 of 6 in all runs.
  - **The result goes against Full.** Restart's mean is at or above Full's at every R (R = 1: 0.50 vs 0.33; R = 7: 1.00 each).
  - **Not monotone.** Full seed 3 certifies 6 of 6 at R = 2-4, 3 at R = 5-6 and 6 again at R = 7. The pick ranks lists by their training score, not by held-out coverage, so a later list that scores better on training can certify fewer held-out targets. Say so in the caption.
  - With 3 runs per arm and one training set each, no difference here is significant.
- **`tab_rq1-sizes.csv`** (P7; per controller: `-per-controller`). See "For the paper".

## For the paper

- **Running example, instance 142** (11x11 adjacent map; `tab_rq2-example.csv`, growing s1 episodes 3 and 4).
  - The rooms list (`seed0`, trained on rooms members 0, 4, 8) fails on 142 by 0.0072: `seq_2_before_3` and `complete_sequence` reach 0.7928 against 0.8; every other requirement holds (goals and `seq_1_before_2` 1.0, `avoid_moving` 1.0 / 0.951 / 0.919 against 0.7). The crossing list fails by 0.193, the adjacent list by 0.80.
  - The generated list `gen_t2` (trained on 0, 4, 8 and 142): 34 rules, 5.9 conjuncts per condition on average (at most 9), 11 distinct features. On 142 its smallest margin is +0.022 (`seq_2_before_3` and `complete_sequence` 0.822); on 136, the next episode, +0.172 (binding `avoid_moving_seg3` 0.872 against 0.7), so 136 is retained.
- **Controller sizes** (mean over controllers; `tab_rq1-sizes.csv`):
  - Full rule lists: random grids 34.2 rules (4.1 conjuncts per condition, 7.7 features), families 30.9 (3.7, 8.2), Pac-Man 16.7 (1.8, 5.0), MER across x 27.3 (1.9, 18.7), across n 46.3 (1.1, 11.3).
  - DT-raw trees: 1,888 nodes (depth 16) on random grids, 894 (14) on families, 36,031 (27) on Pac-Man, 7,173 (25) and 2,543 (18) on MER. DT-voc: 373, 256, 2,259, 2,073 and 699 nodes.
  - PPO: 6,789 parameters on the grid worlds, 5,381 on Pac-Man, 7,836 on MER.
  - Hand: the gridworld template is Python code compiled per cell; the Pac-Man maze rules have 5 and 6 rules.

## Missing

- **T1, Pac-Man permissive Full.** Narrowing (`rules.early_narrow`) is applied after `any` is resolved, so no permissive narrowed list exists. Smallest experiment: certify the pick's list before resolution and narrowing with `any` open (3 runs × 6 targets, PRISM only). This is a different controller from the frozen one.
- **T1, Hand on MER.** No hand-written MER rule list. Smallest experiment: write one over the MER vocabulary and certify it on the 6 + 6 validation targets.
- **T1, permissive column of `mer_one` / `mer_n_one`** (CSV only): never re-certified under the fixed rule.
- **T3 (a), feature-computation time.** Features are computed in Python when the model is rendered, and that time is not recorded; only PRISM's model-construction time is. Smallest experiment: time `domain.model()` per check in `sequential.certify`.
- **T4, NoSP (Full) on the core loop.** `port_ts_noclose` predates the fix. Smallest experiment: `port_ts_history` with `closer` hidden, 3 runs. Families: none.
- **T4, PassFail and Margins.** Never run on the core loop (no config switch).
- **Sealed test, Pac-Man (horizons 50 and 100): left out for now (Asher, 2026-10-19).** Not certifiable within limits for any controller: the model is too large to build. A rule list's induced model grows about 3.5x per 5 steps (Full run 1: 2,839 states at horizon 20, 10,054 at 25, 27,103 at 30). At horizon 50, construction was still enumerating states at 1.8-2.8 million when it failed within the 4 GB Java heap; at 100 it timed out at 900 s, for every solver. Baselines are no smaller (PPO needs the bare model). The four checks made (Full runs 1-2, frozen and no-repair) are in `transfer_sealed_pacman_transfer*.csv` as `not_certifiable`. Smallest experiment: one horizon-50 check with a larger heap and timeout, or PRISM's symbolic engine (MTBDD/hybrid); horizon 100 is likely out of reach either way.
- Sequential studies on Pac-Man and MER, random sequences, synthesis timing, growing-mode libraries and UUV are Rajan's; they are not run here.

## Previous loop only

These numbers exist only for the previous loop, or the pre-fix version of the new one, and do not go in the tables. Values are certified per list, validation, under the loop solver.

- **Selection evidence**: 14.2 vs 5.2 (families), 4.0 vs 1.8 and 6 vs 0 (DECISIONS.md, previous loop). For the core loop see T4b.
- **Feedback vs restarts on families** (`port_fam` vs `port_fam_resample`, 3 runs each): the new loop *before* the Oct 17 gridworld feedback fix and without the strategy hint. Feedback 17.44 of 20 vs restarts 13.67; feedback at the restarts' dollar spend 15.67 (`src/equal_spend.py`).
- **"5.00 at the restarts' spend" (MER across x).** `src/equal_spend.py` cut every chain of each core `port_mer` run to its first R rounds, R the most that keeps the training set's **dollar** spend within the restarts' mean ($0.283 per set, against $0.337 for feedback), and re-ran the pick with repair. The three lists certify 3, 6 and 6 of 6, mean 5.00. It is an equal-cost comparison, not an equal-calls one.
- **Previous loop, Haiku 5.5** (against its reference of grids 6.25/8, families 9.67/20, Pac-Man 2.40/6): restarts only 5.50, 9.20, 2.20; no `closer` (grids) 3.50; "results table only" feedback (closest to Margins) 6.08, 7.89, 2.33.
- **Sonnet 5.5, previous loop**: grids 6.67/8, families 14.00/20, Pac-Man 2.33/6, MER across x 5.00/6, across n 6.00/6, grids without `closer` 2.92/8.

## Setup facts

- **PRISM**: 4.10.1, explicit engine by default (`PrismRunner`), 4 GB Java heap per process, at most 8 concurrent PRISM processes for this work.
  - **Inside the loop** ("fast"): Gauss–Seidel value iteration (`prism.method: gaussseidel`, PRISM's default convergence criterion), falling back to modified policy iteration. This is *not* plain value iteration, which fails on periodic chains.
  - **Exact checks** (final check of each run's lists, every held-out check in the tables, the sequential study): the certification rule above (`prism.exact_methods: [intervaliter, lp]`, `prism.ii_max_iters`, `prism.lp_tolerance`, `prism.exact_fallback_epsilon`).
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
- **Validation vs sealed test**: sealed (`configs/transfer/default.yaml`): `grid_test_large.csv`; family members 15x15-20x20 (ids 36-59, 96-119, 156-179); Pac-Man horizons 50 and 100; MER ids 9-11.
- **Hardware**: AMD Ryzen 7 9800X3D (8 cores, 16 threads), 31 GB RAM, Windows 11.
  - Runs were parallel: several hosted-LLM jobs at once, each with up to `run.workers` PRISM processes, and on 2026-10-19 the re-certification queues alongside them. Wall-clock times are therefore under load; tokens are the cost measure.
- **Baselines compiled for certification**:
  - Gridworld trees and PPO: the frozen controller's action in every state of the target, written as rules over x, y, the goal flags and `obs_idx`, composed with the model like any rule list (`decision_tree.policy_rules`).
  - On MER and Pac-Man, trees compile to rules exactly. Networks and tables are certified on the target's model restricted to the policy, exported as PRISM explicit files and checked with `-importmodel`.
- **RL**: PPO (`src/baselines/ppo.py`).
  - Network: MLP 2×64 tanh with policy and value heads; Adam 3e-4; rollouts 128 × 32 environments; GAE λ = 0.95; grad-norm clip 0.5.
  - γ = 0.99 on gridworld (shaped reward: +1 per goal in order, −1 and end on an early goal or collision, −0.01 per step); γ = 1 on the explicit models (undiscounted requirement).
  - Budget: 2M environment steps per policy on gridworld, one policy per training set and seed (seeds 0-2; the torch seed of training set j is seed + j); 1M steps × 3 seeds on Pac-Man and MER.
  - The frozen policy takes its most likely action. Networks are not stored: each re-certification retrains them.
- **Hand rule lists**:
  - The gridworld template (`domains/gridworld/data/template_policy.py`) was committed on 2026-10-08, before the first transfer runs (Oct 14), but after earlier per-grid LLM rule sets existed.
  - The Pac-Man maze rules were committed on 2026-10-14 with the transfer search.
  - Who wrote them and how long it took is not recorded in the repository. **Asher to fill in.**

## Dropped or blocked

- **E3, Sonnet 5.5** (dropped for now, Asher, 2026-10-19): out of the tables (`viz/paper_results.py` `OTHER_LLMS` is empty), no further runs or checks. The runs stay in `out/results/ablations/port_*_sonnet/` (uncommitted): MER across x and n, Pac-Man and families have 3 runs each (families reruns 1-2 replaced runs that lost chains to OpenRouter 402s); random grids only run 3 (runs 1-2 lost chains, not rerun). OpenRouter lists Sonnet 5.5 at $2 / $10 per M; `rq1_comparison.yaml` says 3 / 15.
- **E4, a third LLM** (blocked, then dropped): `openai/gpt-6-luna` (provider `openai`, $0.10 / $0.50 per M) and the fallback `openai/gpt-5-mini` both fail every call with HTTP 400: "Invalid schema for response_format 'answer': 'additionalProperties' is required to be supplied and to be false". OpenAI's strict structured output needs `additionalProperties: false` in the loop's JSON schema (`core/planner.py` `rule_schema`), a code change.
- **E5, PassFail**: skipped: not a config switch (`planner.feedback` is `blame | table | failures`; `table` is the results table, closer to Margins).

## In progress (2026-10-19 evening)

- **P2** (sealed test set): done for every controller except RL (PPO), OneShot and Pac-Man (left out, see Missing): `transfer_sealed_*.csv`, `--unseal`, the certification rule above. PPO needs the validation networks, which are not stored.
- **P5**: families panel, Full done and drawn (F3); then the MER panel under the fixed rule, the Restart arms, the OneShot point and the equal-spend counts. The random-grids panel last.
- **A7** (effect sizes) and **A5** (replication package): not started.

## Changed since last regeneration

Against this morning's tables (loop solver's checks), moves of more than 0.5 points:

- **T4, families NoRepair: 68.3 → 67.8 %.** One check (run 2, crossing list, target 95) is inconclusive: LP's margin is 0.00038, inside its 1e-3 band; the loop solver had it certified.
- **T1, RL: random grids 68.8 → 66.7 %, families 63.3 → 61.1 %.** RL is now the mean over seeds 0-2 of retrained networks, not one seed's earlier network (see T1).
- **T2, seed library of core run 3 (new block):** 28 → 30 of 36 served when rerun under the certification rule (2 timeouts and 5 inconclusive checks under interval iteration alone had deferred episodes). The library of run 2 is unchanged (31 of 36).
- **Unchanged:** every Full row of T1 and T4 (random grids 92.7 %, families 87.2 %, Pac-Man 66.7 %, MER 100 %), the permissive column, and every DT and Hand row. Interval iteration alone had left the crossing-family lists of core runs 2 and 3 inconclusive (families would have read 69.4 %); linear programming settles all 72 of those checks.
- **New rows:** Restart on random grids, families and Pac-Man; OneShot everywhere (both above).
