# Results notes, Rajan's checklist (SEAMS)

Rajan's items from Asher's checklist. Tables are in `out/paper/tables/` and figures in `out/paper/figures/`, under the
names the checklist gives. Rows Asher merges into his own tables carry `.rajan` in the name. Tables are rebuilt by
`python viz/rajan_results.py` (from `PRISM-Guided-Learning/`), which reuses `viz/paper_results.py`'s functions, so
coverage is counted exactly as in T1/T4. **Every number is a validation result.** No sealed instance is read or built:
`src/transfer.py`'s guard refuses them, and every pool below is validation-only.

Status (Fir run of Oct 19-20, 2026; branch `fir-harness`, Slurm job records in `out/cluster/jobs/`):
- **Done:** items 1, 2, 4 and 5; item 3 for GPT-6 Luna (all 5 conditions) and for Gemini 3.8 Flash on random grids,
  families and Pac-Man; item 6 in frozen mode. Their runs, checks and tables are committed.
- **Not done:** Gemini on both MER benchmarks failed every call (Blocked), and item 6's growing mode crashed (fixed,
  to rerun). Their failed outputs are not committed.
- **Not started:** item 7 waits on Asher; item 8 is last.
- Item 1 verdict check: 228 of the list checks have an Asher verdict for the same list and instance, and none differs.

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

**8. UUV family.** Not started; last item.
- **Checkpoint 1 passes** (from the model alone). `domains/uuv/model.prism.j2` exposes constants that change size and
  structure:
  - `min_visib` / `max_visib`: the range of `water_visib`;
  - `inspect`: the range of `d_insp`;
  - `infl_tf`.

  Its requirements are properties with thresholds: `no_thruster_failure`, `done_in_time` (`F<=deadline`) and
  `energy_budget`.
- **Checkpoint 2 is next.** `done_in_time` is step-bounded, which the LP query cannot decide (T3b). The planned check
  is a step counter in a copy of the model, so the bound becomes reachability that `multi(...)` with LP accepts. It is
  to be confirmed on one instance before anything else.

## Blocked

- **Item 3, Gemini 3.8 Flash on MER: cause found and fixed by config (Oct 20); rerun on Fir.**
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
- **Item 6, growing mode** crashed at the first deferral: `sequential.py` sorted the closest list's training instances
  by grid size `n`, which Pac-Man instances lack (`None`). Fixed on Oct 20: instances without a size keep their order.
  Grid worlds are unchanged. Rerun with `checklist.sh item6 growing`; the frozen runs are unaffected.

## Decisions (Rajan, Oct 20)

- Gemini 3.8 = `google/gemini-3.8-flash`.
- Item 1 runs non-exclusive: three jobs, each with 4 dedicated cores and one PRISM process at a time. Other users' jobs
  may share the node, sharing memory bandwidth and caches but not cores or memory. Each job's `meta.json` records its
  node and our other running jobs.
- Everything is queued at once by `cluster/queue_all.sh`, ordered by Slurm dependencies: item 1 first and alone among
  our jobs, then the rest. Item 7 waits for Asher.

## Missing (until run)

Everything above except rule 1's evidence. Per rule 4, no number is filled in from another run.
