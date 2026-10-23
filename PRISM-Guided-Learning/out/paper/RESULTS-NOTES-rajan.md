# Results notes, Rajan's checklist (SEAMS)

Rajan's items from Asher's checklist. Tables are in `out/paper/tables/` and figures in `out/paper/figures/`, under the
names the checklist gives. Rows Asher merges into his own tables carry `.rajan` in the name. Tables are rebuilt by
`python viz/rajan_results.py` (from `PRISM-Guided-Learning/`), which reuses `viz/paper_results.py`'s functions, so
coverage is counted exactly as in T1/T4. **Every number is a validation result.** No sealed instance is read or built:
`src/transfer.py`'s guard refuses them, and every pool below is validation-only.

Status (Fir runs of Oct 19-20, 2026, Pacific time; branch `fir-harness`; Slurm job records in `out/cluster/jobs/`, Slurm
accounting in `out/cluster/sacct.csv`):
- **Done:** items 1-6, with their runs, checks and tables committed:
  - item 3 for both models on all 5 core conditions. Gemini's MER conditions were rerun on Oct 20 with the schema fix
    (Blocked, resolved).
  - item 6 in frozen and growing mode. Growing mode was rerun on Oct 20 after the fix.
- **Not started:** item 7 waits on Asher; item 8 is last.
- Item 1 verdict check: 228 of the list checks have an Asher verdict for the same list and instance, and none differs.

## Results in brief (validation)

- **Item 1.** Synthesis, the joint query on the bare MDP:
  - grid worlds: a median of 29-327 s per instance, against 2-3 s to certify a stored list (15-100×);
  - Pac-Man: about the same as certification up to length 20; 153 s at length 30 (8.3M states, 85×);
  - 4 of 80 grid-world queries timed out at 900 s (family members 26, 31, 35 and 155; sizes 12-14), and 75 of the
    rest were jointly feasible.
- **Item 2.** The three seed libraries serve 114, 115 and 113 of 120 episodes. Library 3 switches 35 times, against
  12 and 11.
- **Item 3.** Frozen coverage, mean over lists:

  | Benchmark | GPT-6 Luna | Gemini 3.8 Flash |
  |---|---|---|
  | Random grids | 86.5% | 91.7% |
  | Families | 76.1% | 92.2% |
  | Pac-Man | 61.1% | 66.7% |
  | MER across x and n | 100% | 100% |

  - No check was inconclusive or timed out.
  - Spend (OpenRouter's reported cost): Luna $2.91; Gemini $79.38, mostly reasoning tokens.
- **Item 4.**
  - Frozen mode served 35 and 33 of 36 episodes; growing mode served 35 and 36.
  - In r02, the 3 generated lists then served 9 later episodes, but frozen mode also served each of those. So the
    count the checklist asks for (episodes frozen mode defers and growing mode serves with an earlier-generated list)
    is 0 in both sequences.
  - In r01, the one generation attempted was not certified.
- **Item 5.** Growing mode with libraries 2 and 3 served 35 of 36 episodes each, with 4 generations attempted per
  library.
- **Item 6.**
  - Frozen mode, as expected from RQ1: the stored lists certify lengths 8-20, and every episode at length 25 or 30 is
    deferred (served 9, 6 and 8 of 12).
  - Growing mode made 13 generation attempts, all at length 25 or 30, and **none was certified**. Smallest margins:
    -0.161 to -0.166 at length 25 (5 attempts), -0.163 to -0.277 at length 30 (8 attempts). The stored lists miss by
    -0.130 and -0.175.
  - So growing mode served exactly what frozen mode served. 824 Haiku calls, $1.68.
  - Each attempt's verdict and margin: `tab_rq2-pacman-generations.rajan.csv`.

## Setup

- **Commit.** Branch `fir-harness` = Asher's `rule-language` at `6eb6ea03` (after `e4a2d20`, the final certification
  rule) plus Rajan's commits:
  - the OpenRouter strict-schema adapter and response metadata (rule 1, below);
  - item scripts and sequence files;
  - the Fir harness.
  Every Slurm job records its commit, `git diff --stat` and hardware in `out/cluster/jobs/<id>_<name>/meta.json`.
- **Machine.** Alliance cluster Fir:
  - CPU nodes with 2 × AMD EPYC 9655 (192 cores) and 750 GB, Rocky/Alliance StdEnv 2023;
  - Java 21.0.1, Python 3.12.4;
  - PRISM 4.10.1 built from source. The prebuilt binary's `liblpsolve55.so` needs GLIBC 2.38, newer than Fir's stack, so
    its native engines (sparse, LP, hybrid) did not load.
  - Each job gets its own cores and memory (Slurm cgroup). Other users' jobs may share the node unless the job is
    exclusive. Per job, `meta.json` records the node, cores and RAM, and our other running jobs (rule 5).
- **PRISM.** As Asher's:
  - explicit engine by default; 4 GB heap per process; 900 s timeout per call;
  - `prism.exact_methods: [intervaliter, lp]`, `exact_epsilon 1e-9`, `lp_tolerance 1e-3`.
  - The concurrent-process cap is per job: `prism.max_concurrent` (8 by default) lock slots in the job's own `TMPDIR`.
    Every generation run's `run.workers` counts against it. Item 3's full-run job raises it to 20 (24 cores), with
    5 runs × 4 workers.
- **Inputs copied.** `cluster/push.sh --with-results` copies them:
  - the frozen lists of `core/port_fam_history/seed_none_{1,2,3}`;
  - `core/port_pacman_narrow_core/seed_none_{1,2,3}`;
  - `configs/sequential/shift_v0*.yaml` and `out/results/sequential/shift_v0*`;
  - `grid_20_large.csv`, `families_v0.csv`, `sequential_v0.csv`, `pacman_transfer.csv`.

## Rule 1: the provider adapter (item 3)

- **Change.** In `src/core/backends/openrouter.py` and `config.py`:
  - `llm.openrouter.schema_adapter` (default `[]`) rewrites a copy of the answer schema for strict providers:
    - `no_additional_properties`: OpenAI's strict mode needs `additionalProperties: false` on every object, which is
      Asher's Blocked E4 error;
    - `inline_refs`;
    - `drop:<keyword>`.
  - The backend also keeps each response's metadata on the result (`meta`: generation id, serving provider, finish
    reason, usage with reasoning tokens and OpenRouter's cost). It reads only the response; the request is unchanged.
  - Nothing in generation, selection, repair, certification or the prompts changed.
- **(a) Haiku's request path is byte-identical.** `cluster/checks/request_dump.py` built the exact OpenRouter request
  for the first 3 logged Haiku tasks of `core/port_fam_history`, `core/port_pacman_narrow_core` and `core/port_mer`
  (seed 1) with Asher's tree (`6eb6ea03`) and with this branch. `cmp` reports the before and after files identical for
  all three conditions. The files are in `out/paper/rajan/rule1a/` (`<condition>_before.json`, `_after.json`); the
  diff is empty.
- **(b) The limits still hold for the adapted model.** The planner validates every answer against its own pydantic
  model (`core/planner.rule_schema`, in `_ask`), whatever schema was sent: actions as an enum, at most 64 rules,
  conditions of at most 200 characters. The adapter keeps `enum`, `maxItems` and `maxLength` in the schema it sends
  unless a `drop:` step removes one. Test: `tests/test_openrouter.py::test_the_planner_still_enforces_the_limits_whatever_schema_was_sent`.
- An earlier commit on this branch, `llm.openrouter.structured: false`, puts the schema in the prompt. It is used
  only by the cluster smoke test with a free model, never by an item run.

## Items

**1. Synthesis time per instance.** `src/synthesis_time.py`; jobs `cluster/checklist.sh item1`.
- **Synthesis.** Grid worlds: the multi-objective query `ceilings.py` runs (sparse engine, exact LP) on the bare MDP.
  Pac-Man: the single `Pmin` query on the bare model, with the certification's exact solvers (interval iteration,
  then LP).
- **Certification.** `recertify.certify_rules` (rule 2), one check at a time:
  - the three `core/port_fam_history/seed_none_1` lists (frozen, `determinized.json`) on the 20 `grid_20_large` grids
    and the 60 validation family members;
  - the three core Pac-Man lists (narrowed, as kept) on lengths 8, 10, 12, 20, 25 and 30.
- **Verdict comparison.** Each list verdict is compared with Asher's for the same list and instance (`asher_outcome`):
  - families: his `transfer_exact_*_determinized.csv`;
  - Pac-Man: his `transfer_exact_pacman_transfer.csv`;
  - grids: shift_v0's frozen `checks.csv`, where the grid occurs.
- **Columns.** `check_s` = wall time of the PRISM call(s) minus PRISM's model-construction time, so it includes the
  JVM's start. Synthesis rows also have `prism_check_s` (PRISM's own "Time for model checking").
- **Jobs.** Three jobs (families, grids, Pac-Man), each with 4 dedicated cores and one PRISM process at a time.
  `EXCLUSIVE=1` runs the three parts side by side on a node of our own; record which was used. The parts are
  `tab_rq2-synthesis-time.<part>.csv`; the table is `tab_rq2-synthesis-time.csv`, with a size summary in
  `tab_rq2-synthesis-time-summary.csv`.
- **Local smoke test** (1 instance per domain, a desktop, not the paper's numbers):
  - joint query: 78 s on a 10x10 family member and 133 s on a 9x9 grid;
  - certifying one list there: 1.4-2.7 s;
  - Pac-Man length 8: `Pmin` 1.1 s, certification 1.1-2.3 s;
  - all 9 list verdicts equal Asher's.

**2. Random sequences, reuse only.**
- **Sequences.** `configs/sequential/random_v0.yaml` (library `seed_none_1`), `random_v0_lib2.yaml` and
  `random_v0_lib3.yaml`: 10 sequences (r01-r10) of 12 episodes each.
- **Draw.** `src/draw_sequences.py grid --count 10 --length 12 --seed 21001` (numpy `default_rng`), uniformly without
  repeats within a sequence, over shift_v0's pool: 60 validation family members and the 20 grids of `grid_20_large`.
  All three files hold the same sequences. They were drawn and committed before any episode ran (rule 7).
- **Runs.** Frozen mode, PRISM only (`cluster/checklist.sh item2`).
- **Output.** `tab_rq2-random-sequences.csv`: sequence, library, retain, switch, defer, served.

**3. Two more LLMs, Full only.**
- **Models.**
  - GPT-6 Luna: `openai/gpt-6-luna`, provider pinned to `openai`, $0.10 / $0.50 per M.
  - Gemini 3.8: `google/gemini-3.8-flash`, pinned to `google-ai-studio`, $0.75 / $3.75 per M (the standard endpoint).
  - OpenRouter lists no smaller Gemini 3.8 model on 2026-10-19, so "the small model" is Flash (Rajan confirmed, Oct 20).
  - Reasoning: `llm.think: true` (OpenRouter `reasoning.enabled`, the provider's default effort), the closest
    equivalent of Haiku's extended thinking.
- **Conditions.** `configs/run/conditions/<core>_luna.yaml` and `<core>_gemini.yaml`, for the 5 core conditions. Each
  differs from its Haiku file only in `llm.model`, `llm.openrouter.providers` and `llm.openrouter.schema_adapter`
  (Luna: `[no_additional_properties]`; Gemini: `[]` unless its smoke test needs one).
- **Smoke test.** `checklist.sh item3-smoke <model>`: one training set (`[0, 4, 8]`) of `port_fam_history_<model>`,
  then `cluster/checks/smoke_report.py`. It reports calls, errors, re-asks and invalid answers, cut-off replies
  (finish reason `length`), serving providers, reasoning tokens and reported cost. A model that fails twice is
  dropped, with the reason recorded here.
- **Full runs.** `checklist.sh item3-full <model>`: 5 conditions × 3 unseeded runs, into
  `out/results/ablations/<core>_<model>/seed_none_{1,2,3}`. These are the suffixes `viz/paper_results.py` reads
  (`_luna` is already in its `OTHER_LLMS`; add `"Gemini 3.8 Flash": "_gemini"`).
- **Certification.** `checklist.sh item3-certify <model>`, i.e. `cluster/jobs/certify_llm.sh`: `recertify.py run` with
  frozen `determinized` and permissive `final` (Pac-Man: `final`, the narrowed lists), on each benchmark's RQ1
  validation targets. It writes `transfer_exact_*.csv` per run, in Asher's format.
- **Outputs.** `tab_rq1-coverage-llm.rajan.{tex,csv}` (one row per LLM per benchmark, with n), and
  `tab_rq1-llm-spend.rajan.csv`: calls, tokens, dollars at the list price, and OpenRouter's reported cost per model.
  Haiku rows are not copied in.
- **NoRepair, if time.** `checklist.sh item3-certify <model> norepair_determinized`, then
  `tab_rq3-ablation-llm.rajan.{tex,csv}`.

**4. Longer sequences on the grid worlds.**
- **Sequences.** `configs/sequential/long_v0.yaml`: 2 sequences of 36 episodes, drawn with
  `src/draw_sequences.py grid --count 2 --length 36 --seed 21002` (rule 7); seed library 1.
- **Runs.** Frozen mode first, then growing mode with Haiku 5.5 and the core families configuration
  (`checklist.sh item4 frozen|growing`). Output in `out/results/sequential/long_v0/{frozen,growing}/r0{1,2}/`.
- **Tables.**
  - `tab_rq2-long-episodes.rajan.csv`, per episode: the list that served it, and whether that list was generated
    earlier in the same sequence;
  - `tab_rq2-long-sequences.rajan.csv`: per sequence, the decisions, the LLM cost, and the count of episodes frozen
    mode defers and growing mode serves with a list it generated earlier.

**5. Growing mode for seed libraries 2 and 3.**
- **Runs.** `configs/sequential/shift_v0_lib2.yaml` and `shift_v0_lib3.yaml` (Asher's P6 files, unchanged) in
  growing mode, with Haiku 5.5 and the core families configuration (`checklist.sh item5`).
- **Output.** `out/results/sequential/shift_v0_lib{2,3}/growing/s{1,2,3}/`, sibling directories of Asher's frozen
  runs.
- **Tables.** `tab_rq2-growing-libs.rajan.csv` (per sequence, with LLM cost) and
  `tab_rq2-growing-libs-episodes.rajan.csv`.

**6. Sequences on Pac-Man.** `configs/sequential/pacman_v0.yaml`, 3 sequences of 12 over ids 0, 1, 2, 6, 7, 9
(lengths 8, 10, 12, 20, 25, 30). Every length occurs at least once per sequence, so lengths below the training lengths
(14-18) mix with lengths above them. Drawn with `src/draw_sequences.py pacman --count 3 --length 12 --seed 21003` and
committed before any episode ran (rule 7).
- **Library.** The three core Pac-Man frozen lists: `seed_run` lists `core/port_pacman_narrow_core/seed_none_{1,2,3}`,
  entries `seed1.0`, `seed2.0`, `seed3.0` (the narrowed lists as kept; `determinized.json` equals the final rules there).
- **Runs.** Frozen mode first (PRISM only), then growing mode with Haiku 5.5 and the core Pac-Man configuration
  (`checklist.sh item6 frozen|growing`). Output in `out/results/sequential/pacman_v0/`.
- **Tables.** `tab_rq2-pacman-sequences.rajan.csv`, and `tab_rq2-pacman-generations.rajan.csv`: every generation
  attempt at lengths 25 and 30, with its verdict and smallest margin.
- **Two changes to `src/sequential.py` were needed (Rajan, Oct 20).** Before them, every Pac-Man list was `unparsed`.
  Neither changes how a list is generated, picked, repaired or certified.
  1. *Certification vocabulary.* With `rules.early_narrow` set, the manager's config turns on `rules.shared_literals`,
     the same two lines `transfer.py` and `recertify.py` already have, so narrowed lists (e.g. `!(xP = 3 & yP = 4)`)
     parse. That config is used only to certify. Generation runs `run_ablation.py` in its own process with the study's
     condition, unchanged. Grid-world studies have `early_narrow: 0`, so they are unaffected.
  2. *Several seed runs.* `seed_run` may be a list; each run's lists join the library as `seed<k>.<id>`. A single run
     keeps its names (`seed0`, `seed1`, ...).
- **Java heap 8 GB for Pac-Man (items 1, 3, 6), a deviation from "4 GB per process".**
  - With 4 GB, PRISM runs out of Java heap building the bare length-30 model (8.3M states) at about 6.6-7.0M states.
    That happened with Java 21 and 25 and with either garbage collector.
  - The verifier needs that model for its forced states, so every length-30 check came out `timeout`, where Asher's
    verdict is `failed`.
  - With 8 GB the model builds in 162 s (PRISM's resident memory about 8 GB). The check of `seed1.0` at length 30 then
    gives Asher's verdict exactly: failed, interval iteration, margin -0.1748, 27,103 states (182 s).
  - The heap is not part of a run's identity and changes no computed value.
  - Where it is set: `pacman_v0.yaml`'s `generation_set`, `synthesis_time.py --java-max-mem 8g` for Pac-Man, and the
    Pac-Man condition's `--set` in item 3's runs. Grid worlds and MER keep 4 GB.
- **Checks of the change.**
  - `tests/test_sequential.py`: entry names and training sets for one and three seed runs; the narrowed lists parse
    with the certification vocabulary and are rejected without it.
  - Regression: shift_v0 s1 in frozen mode, rerun after the change, reproduces Asher's `episodes.csv` decisions and
    `checks.csv` verdicts (below).
  - Validation, pacman_v0 r01 in frozen mode, locally:
    - episodes 0-9 match RQ1: lengths 8, 10, 12 and 20 certified by interval iteration, all three lists identical at
      20 (margin +0.0015);
    - length 25 deferred: every list fails by -0.130 (Asher's T1b);
    - length 30 needed the 8 GB heap above.

**7. Restart arm on the families, one extra LLM.**
- **Prepared, not to start yet:** `port_fam_history_luna_resample.yaml` and `port_fam_history_gemini_resample.yaml`.
  Each is the item 3 condition with `planner.retry: always`, nothing else changed.
- `checklist.sh item7 <model>` runs 3 runs × 3 sets, then certifies them under rule 2.
- Its rows go into `tab_rq3-ablation-llm.rajan`, with Full's and Restart's LLM calls per set.
- **Start only when both hold:** Asher confirms E1 shows a difference, and that model's Full families runs are
  certified.

**8. UUV family. Complete** (past the freeze, Rajan, Oct 22: continue; so not in the SEAMS draft's tables unless
Asher takes it). Checkpoints 1-5 pass; the coverage block is `tab_rq1-coverage-uuv.{tex,csv}`.
Evidence: `out/paper/rajan/item8/`.
- **Checkpoint 1 passes**, from the model alone. `domains/uuv/model.prism.j2` exposes constants that change size and
  structure:
  - `min_visib` / `max_visib`: the range of `water_visib`, and the band edges `med_visib = W/3`,
    `high_visib = 2W/3`;
  - `inspect`: the range of `d_insp`;
  - `infl_tf`.

  Its requirements are properties with thresholds: `no_thruster_failure` (`G`), `done_in_time` (`F<=deadline`) and
  `energy_budget` (an expected reward until "done").
- **Checkpoint 2 passes. The check: a certified witness.** An instance is jointly solvable if some controller is
  certified under rule 2 to meet every threshold at once. It is sound for "yes"; a member without a witness counts as
  not shown solvable.
  - Confirmed on both paper scenarios: `stay` is certified, North Sea 0.6734 / 0.8067 / 26.43.
  - Exact queries tried first, all failed:
    - PRISM's multi-objective LP rejects the step bound, state rewards and `R[F]`.
    - On an exact copy that removes those, PRISM's LP did not finish in 25 min. The copy adds a step counter `t` and a
      failure flag `ok`; energy is total reward `C`.
      - Shown exact on North Sea: best and worst on-time 0.8181 / 0.5164, energy 24.784 / 44.388 and safety
        0.67410 / 0.65383 match the original.
      - 95k-137k states, against 5.6k.
    - On the same copy, a HiGHS LP (`src/uuv_joint.py`, kept for the record) did not finish in 15 min.
    - PRISM's value-iteration multi-objective method answers in 6 s but is wrong here: best on-time 0.472 under the
      other two thresholds, while `stay` is certified at 0.8067. DECISIONS already notes this method understating.
- **Checkpoint 3 passes.**
  - *Family* `domains/uuv/data/uuv_family_v0.csv` (`family_v0.py`, PRISM only):
    - North Sea dynamics; visibility range W = 7-12; inspection 6, 8, 10, 12, 14 m; deadline 3 × inspection.
      30 members: 3 training (ids 0, 6, 10: W 7, 8, 9) and 27 validation.
    - Thresholds per member and requirement: best - 0.1 × (best - worst), from exact bounds. The paper's thresholds
      sit at 6-20% of these ranges; "optimum - 5%" would let every controller pass `no_thruster_failure`, whose range
      is about 0.02.
    - The sealed test members (longer inspections) are not generated: building them is the test pass's (rule 3).
  - *Gate* (`src/uuv_gate.py`, rule 2 on all 30; `out/paper/rajan/item8/gate.csv`):

    | Controller | Certified |
    |---|---|
    | `always_low` | 0 |
    | `always_med` | 26 |
    | `always_high` | 10 |
    | `stay` | 25 |
    | uniform random | 0 |
    | hand-written list | 30 |

    - The hand-written list: search at med; once forced low, stay low until visibility is one level above the poor
      band; keep high.
    - It is every member's joint-solvability witness. Its smallest margin is 0.0002.
    - Uniform random is exact: a pick variable redrawn every step, averaged over the first pick.
  - *Vocabulary* (`domains/uuv/domain.py` `rule_vocabulary`, `spec.yaml.j2`). Written from the model and requirements
    alone in about 10 min (01:19-01:29), and committed (`2e719a07`) before any LLM list was checked on a held-out
    member.
    - Constants: `MIN_VISIB`, `MAX_VISIB`, `FIRST_AVG`, `FIRST_GOOD`, `INSPECT`, `DEADLINE`, the mission-state codes
      and the altitude codes.
    - State features: `poor_visib`, `average_visib`, `good_visib`, `starting`, `searching`, `recovering`.
    - Action features: `keeps_altitude`, `allowed_now`.
    - The hand-written list in this vocabulary gives the same margins as its numeric form.
- **Checkpoint 4, prepared.**
  - Condition `port_uuv`: `port_mer` with only the domain's settings changed (domain, dataset and training set;
    `planner.branch: per_requirement`; `feedback.horizon: domain`; `done_margin 0.0001`).
  - Plumbing smoke test with the free model (Nemotron, schema in the prompt, 1 chain × 2 rounds): passed. 2 calls; the
    answers use the vocabulary; rounds, pick, final and determinized lists all run.
  - **Waiting:** the Haiku 5.5 smoke test and three full runs need the paid key.
  - **Haiku 5.5 smoke test: passed** (local, Oct 22; `port_uuv` on its training set, cut to 1 chain × 2 rounds;
    `cluster/checks/smoke_report.py`).
    - 3 calls, all finish reason `stop`, all served by Anthropic; $0.0053.
    - One invalid answer named the state codes as numbers (`s = 4`), which the general rule language rejects. The
      re-ask with that error came back valid, naming the constants (`s = START_SEARCH`, `good_visib`).
    - The final list passes 2 of the 3 training members after 2 rounds.
  - **Full runs** (Fir, Oct 22, job `item8-full`, 30 min): 3 runs of `port_uuv`.
    - 61-75 calls each, no failed call, $0.43 in all.
    - Runs 1 and 3 solve their training set; run 2's list passes members 0 and 10 but not 6.
- **Checkpoint 5 passes: the UUV block** `tab_rq1-coverage-uuv.{tex,csv}`.
  - Validation: the 27 held-out members (ids other than 0, 6, 10), certified under rule 2 (`transfer_exact_*`).
  - Built by `viz/rajan_results.py --items 8`.

  | Controller | Frozen coverage | n | Certified per list | Misses |
  |---|---|---|---|---|
  | Full (Haiku 5.5) | 91.4% ± 2.1 | 3 runs | 24.67 of 27 | runs 1 and 3: w7_i12, w7_i14 (safety); run 2: w8_i10, w8_i12, w8_i14 (energy) |
  | DT-raw | 55.6% | 1 tree | 15 of 27 | 12 members, all on safety |
  | DT-voc | 0.0% | 1 tree | 0 of 27 | all, on safety |
  | RL (PPO, our features) | 92.6% ± 0.0 | 3 seeds | 25 of 27 | every seed: w7_i12, w7_i14 (safety) |
  | Hand | 100% | 1 list | 27 of 27 | none (written while building the family; a reference, not blind) |

  - No check was inconclusive or timed out.
  - Permissive equals frozen: the Haiku lists have no `any` left after the pick; the baselines are deterministic.
  - **Sizes:** Haiku lists have 5-6 rules. The raw tree has 67 leaves (depth 9), the features tree 11 leaves. PPO is
    a 5,252-parameter network, certified as one rule per decision state.
  - **Context:** on all 30 members (from the gate), `always_med` certifies on 26 and `stay` on 25. The family is
    non-trivial (each simple controller fails somewhere), but simple controllers come close, and the hardest members
    (w7_i12, w7_i14) are where both Haiku and PPO miss.
  - The trees learned "done within the deadline and no thruster failure". Even that objective's optimum misses the
    safety threshold on 2 of 3 training members, which explains their safety misses.
  - Jobs: `item8-full` (30 min) and `item8-baselines` (2 h 27 min, mostly PPO's 3 × 1M steps), both completed.
- **Checkpoint 5, the baselines: implemented and tested** (Rajan, Oct 22: modified option A).
  - *Objective.* The trees and PPO learn "done within the deadline and no thruster failure", `F "goal"` with
    "goal" = done & t <= D & ok. It is evaluated on the product of the model with the step counter and the failure
    flag (`UUV.objective`, `domains/uuv/domain.py` `product_model`).
    - On member 0, numpy's optimum (0.567032) equals PRISM's interval iteration (0.5670320).
    - It is below both the best on-time value (0.658) and the best safety value (0.773), as it must be.
  - *Code.* No generation, selection, repair, certification or prompt code changed:
    - `Domain.objective` is a new optional hook, `None` by default.
    - `ExplicitMDP(..., objective=True)` builds that model.
    - Trees and PPO take `requirement: objective`.
    - PPO takes `certify: rules`, so its per-state policy is certified as rules by the standard verifier.
      `recertify.py baseline` takes the rule path for any baseline that is not explicit. The explicit-model
      certification cannot check UUV's energy reward.
  - *Configs.* `configs/baselines/uuv_tree_raw.yaml`, `uuv_tree_features.yaml`, `uuv_ppo_features.yaml` (1M steps,
    seeds 0-2) and `uuv_hand.yaml` (the gate's list in the vocabulary), with condition `uuv_baseline` (extended
    vocabulary, numbers allowed).
  - *Tests.* `tests/test_uuv_objective.py`; the existing baseline, explicit-model, UUV and domain tests pass.
  - *Local trial* (rule 2 on all 30 members, desktop; results deleted, the Fir job `item8-baselines` makes the
    recorded ones):
    - hand-written list: 30/30;
    - raw tree (67 leaves): 15/30;
    - features tree (11 leaves): 0/30;
    - PPO, at a 20k-step pipeline test only: 3/3 on targets 0-2.
    - Every tree failure is `no_thruster_failure`, including on the training members (margins -0.012 to -0.017).
  - **This comes from the objective, not the code.** Even the optimal controller for the objective, which can see
    `t` and `ok`, misses the safety threshold on 2 of 3 training members (member 0: 0.7703 against 0.7709). The
    objective trades a little safety for punctuality, and energy is not in it at all.
- **Item 8 follow-up: Asher's UUV plan** (Oct 22, branch `rajanmaghera/uuv-2`). Steps 1-7 are prepared; no Fir job
  has run.
  - *Step 1, done* (`a3b355f9`). 12 sealed test members, ids 30-41 (W 7-12 at inspection 16 and 18 m), each bound
    by interval iteration. Rows 0-29 are byte-identical; `configs/transfer/default.yaml` seals 30-41.
  - *Step 2, done* (`d7d826c8`). The fast checker equals PRISM within 1e-7 on members 0-29 for every list the plan
    names (`tests/test_uuv_fast_checker.py`). "Reached almost surely" uses the exact graph algorithms; value
    iteration only approaches 1, which made expected energy read as infinite.
  - *Steps 4 and 5, code done* (`d7d826c8`). Training rows 42-47; conditions `port_uuv_sets` and
    `port_uuv_sets_words`; the words hooks and their test.
  - *Step 3, prepared* (`cluster/checklist.sh item8-repick`). Item 8's runs moved to `out/results/prepick/port_uuv`.
    `src/equal_spend.py --as-run` re-picks them into `out/results/core/port_uuv` as finished runs (rounds, results
    parquet, task log). Then `cluster/checks/uuv_repick_check.py` runs the sanity check.
    - Local trial: the NoRepair list equals item 8's on runs 1 and 2. Run 3 differs: its `any` rules are now resolved.
    - **Finding: repair makes Full worse on UUV** (DECISIONS, [REVIEW]). Repair maximizes the members' capped margins
      summed, so it trades safety's 1e-4 passing margin for energy and deadline gains.
      - Run 1's list passing all three members is repaired into one passing none.
      - Full then keeps a 2-of-3 list (frozen: fails validation members 1 and 2, which NoRepair certifies).
      - With `planner.repair_score: members_first`, repair never loses a member. Run 1 keeps its 3-of-3 list (=
        NoRepair), and run 3's repair lifts two candidates from 2 to 3 members.
      - On the other benchmarks' stored runs, 5 of about 900 repaired candidates lost a member; no chosen list did.
  - *Step 6, prepared.* Restart arm (`item8-resample`), OneShot and budget curves (`item8-curves`), and baselines
    (`item8-baselines2`: `uuv_ppo_raw`, `uuv_random_policies`, `uuv_one_instance`, and both PPO configs refit with
    `--nets`).
    - Random policies and the one-instance table learn the objective and are certified as rules.
    - Local trial on 3 members: both run end to end, and the checker agrees with PRISM.
    - The paper pipeline is a separate commit: `rq1_comparison.yaml` benchmark `uuv`, plus `viz/paper_results.py`'s
      `RQ1_BENCHES`, `RQ3_BENCHES`, `RESTARTS`, `BUDGET_PANELS`, `SIZE` and `SEALED`.
      - A dry run with the output in scratch completes. UUV's baseline rows equal item 8's.
      - UUV's uniform controller has no baseline config (the gate computes it), so the paper tables lack it; it is
        in `tab_rq1-coverage-uuv`.
  - *Step 7, prepared* (`SEALED=yes ... item8-sealed`). `uuv_gate.py --targets 30-41 --unseal` writes
    `gate_sealed.csv`; without `--unseal` the gate refuses sealed members.
    - `viz/rajan_results.py --items 8 --sealed` builds `tab_rq1-coverage-uuv-sealed`.
  - *Smoke tests.*
    - Free model (Nemotron, 1 chain × 2 rounds): `port_uuv_sets_words` on set 1 (42-44) and `port_uuv_resample`
      both pass, 2 calls each.
      - The words arm's second prompt names states and losses in words, with no `s=` / `water_visib=` form.
      - The Restart arm's second prompt is identical to its first.
    - Haiku: `cluster/smoke/uuv2_haiku_smoke.sh`, waiting for approval.
  - *Report.* `tab_rq1-coverage-uuv` gets the plan's rows, with permissive, frozen and library coverage, and LLM calls
    and reported cost per run. DT-raw and DT-voc are in the CSV only, with the note.
  - *Job records.* Item 8's `meta.json` recorded `git_dirty: null`. `cluster/push.sh` copies `.git`, so on Fir the
    index's file stats are stale and `git status` re-reads every file; on Lustre that likely exceeded the 120 s
    limit.
    - The `out/` exclusion was also resolved from the working directory.
    - `cluster/lib/job.sh` now refreshes the index first (900 s), takes the diff from the top of the repository, and
      sets `git_dirty` from it.


## Blocked

- **Resolved: item 3, Gemini 3.8 Flash on MER.** Cause found and fixed by config (Oct 20); rerun on Fir on the afternoon of Oct 20:
  6 runs, no failed call, certified.
  - Failure: every call of `port_mer_gemini` and `port_mer_n_gemini` failed with HTTP 400 from Google AI Studio,
    "Request contains an invalid argument" (24 calls per condition, 0 tokens).
  - Diagnosis, one request per variant (rejected requests cost nothing):
    - the families prompt with the MER schema is rejected, so the schema is the cause;
    - with `$ref` inlined, rejected; without `maxLength`, rejected; without titles and descriptions, rejected;
    - with only 5 of the 28 actions, accepted;
    - with all 28 actions and no `maxItems: 64` on the rules array, accepted.
  - Fix: `llm.openrouter.schema_adapter: [drop:maxItems]` on the two Gemini MER conditions only. Haiku's and the
    other Gemini conditions' requests are unchanged.
  - Rule 1(b): the 64-rule limit is still enforced, by the planner's validation of every answer
    (`tests/test_openrouter.py`).
  - Smoke test, local, Oct 20: `port_mer_gemini` with 1 chain × 2 rounds passed. 1 call; the answer parsed, no
    re-ask, the training set was solved in round 1, final rules written; $0.065. `port_mer_n_gemini` sends the
    identical schema.
  - Diagnosis and smoke spend: about $0.08.
- **Resolved: item 6, growing mode.** Fixed (Oct 20); rerun on Fir on the afternoon of Oct 20, all 3 sequences complete. It crashed at the first deferral: `sequential.py` sorted
  the closest list's training instances by grid size `n`, which Pac-Man instances lack (`None`). Instances without a
  size now keep their order; grid worlds are unchanged. The frozen runs are unaffected.
  - Local checks, Oct 20:
    - the crashed path, with generation stubbed (no LLM): a 2-episode sequence starting at length 25 defers, picks
      training `[3, 4, 5, 7]` (the closest list's horizons plus the length met), calls generation and continues;
    - one real generation as growing mode launches it, at the hardest length: `port_pacman_narrow_core` with
      training `[3, 4, 5, 9]` (length 30), 8 GB heap, cut to 1 chain × 1 round. It completed with final rules
      (1 Haiku call, $0.0013), and `determinize.py` succeeded. 16 min, peak memory 10.3 GB.


## Decisions (Rajan, Oct 20)

- Gemini 3.8 = `google/gemini-3.8-flash`.
- Item 1 runs non-exclusive: three jobs, each with 4 dedicated cores and one PRISM process at a time. Other users' jobs
  may share the node, sharing memory bandwidth and caches but not cores or memory. Each job's `meta.json` records its
  node and our other running jobs.
- Everything is queued at once by `cluster/queue_all.sh`, ordered by Slurm dependencies: item 1 first and alone among
  our jobs, then the rest. Item 7 waits for Asher.

## Missing (until run)

Everything above except rule 1's evidence. Per rule 4, no number is filled in from another run.
