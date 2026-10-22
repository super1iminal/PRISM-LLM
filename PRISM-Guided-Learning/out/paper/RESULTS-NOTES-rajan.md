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

**8. UUV family. Stopped at the freeze (Wed Oct 21, 23:59 Toronto) at checkpoint 2.** Not in the paper's tables; what
was produced goes to the replication package (`out/paper/rajan/item8/`).
- **Checkpoint 1 passes**, from the model alone. `domains/uuv/model.prism.j2` exposes constants that change size and
  structure:
  - `min_visib` / `max_visib`: the range of `water_visib`;
  - `inspect`: the range of `d_insp`;
  - `infl_tf`.

  Its requirements are properties with thresholds: `no_thruster_failure` (`G`), `done_in_time` (`F<=deadline`) and
  `energy_budget` (an expected reward until "done").
- **Checkpoint 2, the replacement check: built and shown exact on one instance; LP result not reached.**
  - The check is a copy of the model with a step counter (`stepcount`, +1 on every transition, saturating at
    deadline + 1). `F<=D "done"` becomes the reachability `F ("done" & t<=D)`.
  - Two more obstacles turned up and are handled in the same copy:
    - PRISM's multi-objective engine takes only transition rewards, so the energy reward moves from each state to the
      transitions leaving it;
    - it takes only total reward (`C`), which equals reward until "done" here, because "done" is reached with
      probability 1 under every scheduler and no energy accrues there.
  - On North Sea (deadline 30), the copy gives exactly the original's best and worst probability of finishing on time
    (0.8181 / 0.5164) and best and worst energy (24.784 / 44.388), at 95,215 states against 5,580.
  - The joint query `multi(P>=0.67 [G !"thruster_failure"], P>=0.8 [F ("done" & t<=30)], R{"energy"}<=26.5 [C])`
    (sparse engine, LP) had not finished after 25+ minutes on a desktop at the freeze.
  - Possible speed-up, untested: replace the `G` objective with a flag that turns false on leaving a failure state, so
    every objective is plain reachability and needs no LTL product.
- **Not reached:** checkpoint 3 (a new family varying visibility range, deadline and inspection length, since the
  failure-rate family is trivial (positioning Check 2), and a vocabulary), checkpoint 4 (Haiku smoke and full runs),
  checkpoint 5 (certification of Haiku lists, trees and PPO).
- **Smallest experiment that would supply the block:** finish checkpoint 2 on one instance, then the family and
  vocabulary (PRISM only), then 3 Haiku runs of the core configuration on its training set.

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
