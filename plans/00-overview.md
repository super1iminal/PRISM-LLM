# Plans: overview, order and coordination

Twelve workstreams, each in its own file, written to run as separate sessions. This page has the shared rules, the current state, the order, and the decisions only Asher can make.

## Standing rules (every plan)

- **Test set sealed.** Never pass `--unseal`. Sealed targets are listed in `configs/transfer/default.yaml`. Any new sealed target needs Asher's say.
- **Haiku only.** No Sonnet runs until we truly think we have found an improvement or the final method; then ask Asher, each time. Sonnet cells keep `launch: false`. Haiku runs may run in parallel without asking.
- **PRISM may choose among versions of the LLM's output for now** (`rules.any_order`, hidden members, polishing) until Marsha answers. Every such choice keeps a feedback-only counterpart in the ablations.
- **Story:** the LLM writes a generic strategy and the verifier helps refine it (feedback, plus the order that resolves `any` rules), then certifies it on every held-out instance.
- **Organization:** RQ1 is a straight comparison of our best configuration against the baselines (`configs/experiments/rq1_comparison.yaml`, runs in `out/results/core/`). RQ2 is a set of ablation experiments that share runs through the run store (`configs/experiments/rq2_*.yaml`, runs in `out/results/ablations/`). Core runs stay apart from ablation runs. Baselines are methods that produce one general solution for a family of MDPs, and more of them is better. Figures show only the best configuration of each method; tables may show every variant.
- **Memory runs are tagged `-memory`** in every condition name, run directory and experiment file (plan 07), so they are never compared with memoryless runs by accident.
- **Every new behaviour is a config key, off by default**, with a condition file and a line in `docs/config.md` (and `docs/semantics.md` if it changes the loop or the rule language). The run store and the saved-run replays (`tests/test_scheduler.py`) depend on this.
- In prose, name conditions by what they do ("restarts only"), not by code.
- Not planned: evolving a population of rule lists (token cost); option B, where the LLM also writes the model (parked); the LLM-written vocabulary (parked).

## Where we stand (validation numbers, `out/results/experiments/*/REPORT.md`)

| Benchmark | Ours (Haiku, 5 runs) | Best baseline | Verdict |
|---|---|---|---|
| Random grids, 8 held-out | 6.25 per list, library 8/8 | template 6, search 4.5; trees and PPO not yet PRISM-checked | small win; the library is the result |
| Random grids, no `closer` | 3.4 | template 6 | loss |
| Families, 60 held-out | 9.7 per list (rooms 8.4, crossing 4.6, adjacent 16), library 33.8 | PPO 12.7, trees over our features 12.3 (crossing 19), library 43 | loss |
| Pac-Man, 5 held-out horizons (8, 10, 12, 20, 25) | 3 of 5 (8, 10, 12 at the optimum; 20 and 25 fail, 0.749 at 25) | hand-written maze rules 1; nothing else | no real comparison |
| UUV | – | – | abandoned (the failure-rate family is trivial) |

RQ2, at 3 runs per cell, mostly within noise. What matters for the story:
- Feedback against restarts only: ahead on grids (+0.83), level on families, where restarts use a third of the tokens.
- Random blame (placebo) matches blame by probability mass. A failing trace did not help.
- The verifier's choice of action order is what helps most: families lose 5.6 of 9.7 without it. A fixed declared order is better on grids (+0.9) but fails on crossing.
- Half of the RQ2 cells have no runs: the results table only, feature-space feedback, and six of the eight feature cells.

So the current evidence for "the verifier helps refine it" is the order resolution and keep-best, not the content of the feedback. Plans 03–05 test that directly, and plan 09 decides what goes into the core configuration.

## Workstreams

| # | Plan | Kind | Needs | Feeds | Wave |
|---|---|---|---|---|---|
| 01 | [Results organization](01-results-organization.md) | code + reports, no LLM | – | everything (do its step 1 first) | 1 |
| 02 | [Pac-Man baselines: 1-2-3-Go! reimplemented, PPO, simple ones](02-pacman-baselines.md) | code + PRISM | – | 08, 09, 10, 12 | 1 |
| 03 | [Feedback against pure restarts](03-feedback-vs-restarts.md) | analysis, then Haiku | 01 step 1 (Pac-Man in RQ2) | 09, 10 | 1 |
| 04 | [What the verifier tells the LLM](04-verifier-output.md) | Haiku runs, little code | 01 step 1 | 09, 10 | 1 |
| 05 | [Failing trajectories in the feedback](05-failing-trajectories.md) | core code + Haiku | 04's yaml | 09, 10 | 2 |
| 06 | [Feature engineering in the domain](06-features.md) | Haiku runs, little code | 01 step 1 | 09, 10 | 1 |
| 07 | [One bit of memory (`-memory`)](07-memory.md) | core code, PRISM check, Haiku | – | 09 (separately), ICAPS | 1 (design), 2 (runs) |
| 08 | [More 1-2-3-Go! benchmark models](08-qvbs-families.md) | new domains | 02 (generic baselines) | ICAPS | 1 (scouting), 3 |
| 09 | [Core configuration, RQ1, test pass, hand-off to Rajan](09-core-config-and-rq1.md) | decisions + runs | 02–06 | 10, ICAPS | 2–3 |
| 10 | [SEAMS paper](10-seams-paper.md) | writing | 01–04, 06 | – | 1–2 |
| 11 | [UUV family (later)](11-uuv-family-later.md) | PRISM check first | – | ICAPS or later | later |
| 12 | [LLM-written program baseline](12-llm-program-baseline.md) | code + Haiku | 02 | 09, ICAPS | 2 |

## Timeline

- **Oct 15–17 (wave 1):** 01 step 1 first (a few hours), then 01 rest, 02, 03 (analysis first), 04, 06, 07 (design and PRISM-only check), 08 (scouting), 10 (sections that do not depend on numbers).
- **Oct 17–20 (wave 2):** 05, 07 runs, 12, and 09's selection once 03, 04 and 06 report.
- **Oct 20, evening: SEAMS freeze.** Whatever is in the reports then goes in the paper: validation numbers only, labelled as such (decision 3).
- **Oct 21–23:** 10 writes the results. Experiments keep running for ICAPS.
- **After SEAMS:** 09's test pass (when Asher unseals), 08's domains, 11, Sonnet once Asher agrees we have the final method, hand-off to Rajan. ICAPS abstract Dec 7.

## Coordination between parallel sessions

- **Code goes in a worktree branch** off `rule-language`, one per plan. The app may cut a session's worktree from `main`: check `git log -1 --oneline` against `rule-language` first, and if they differ, run `git switch -C <plan branch> rule-language` before any edit (the fresh worktree has nothing to lose). Never merge `main` into a plan branch. Before merging, rebase on `rule-language` and run `../.venv/Scripts/python -m pytest -q tests -k "not replay"`, plus the replays if prompts or the loop were touched. Merge into `rule-language` and push.
- **LLM runs are launched only from the main checkout** (`C:\Users\asher\Documents\Coding\llmtrust\PRISM-LLM`, branch `rule-language`), after the code they need is merged. The run store sees started runs there and never launches over them, so two plans that need the same cell (for example the Pac-Man reference cell) do not run it twice. A run made in a worktree is invisible to the other plans.
- **Commit at every step**, so Asher can trace each iteration in `git log`. The message starts with `[plan NN] step K:` and says what changed and where to look (report path, figure, run directory). Coordination commits (reviews of a plan's report, plan edits) start with `[coordination]`. Find them with `git log --oneline --grep "^\[plan 03\]"`.
- **Commit results path by path** from the main checkout (`git add <your run dirs and report>`), never `git add -A`. If `index.lock` is held, wait and retry.
- **Machine budget (Asher, Oct 15):** at most **8 PRISM processes at once across all sessions**, half of the 16 logical cores, so Asher keeps CPU for his own work. Each job keeps up to `run.workers` PRISM processes busy, so that is about 2 jobs in all. Before launching, count the Java processes (`Get-Process java`) and pass a `--max-parallel` that fits. Once the speed-up work merges, `prism.max_concurrent` (default 8) enforces this in code. (RAM: 31 GB, PRISM heap 4 GB per process.)
- **Files with one owner** (others append through their own merge and keep their edit small):
  - `configs/experiments/benchmarks.yaml` and `only:` scoping in every `rq2_*.yaml`: 01 (step 1).
  - `src/experiments/*`, `viz/`, `out/results/README.md`: 01. Exception: 07 adds the `-memory` guard to `src/experiments/base.py` and `store.py`.
  - `rq1_comparison.yaml`: 01 owns its structure. 02 and 12 add baseline entries; 09 changes the cells.
  - `rq2_feedback.yaml`: 03. `rq2_verifier_output.yaml`: 04 (05 appends trace cells). `rq2_features.yaml`: 06.
  - `src/core/analysis.py` and the trace template: 05. `src/core/rules.py`, `verifier.py` and the planner's answer schema: 07. Both touch `src/config.py`, `configs/run/default.yaml`, `docs/config.md`, `docs/semantics.md` and maybe `src/core/prompts.py`: keep additions in one block each, so the merges stay trivial.
  - `src/baselines/*`: 02, then 12 (new module only).
  - `domains/<new>/`: 08. `domains/pacman/`: 02 (dataset rows) and 06 (features).
  - `paper/`: 10.
- **After a run:** regenerate the report (`src/run_experiment.py <yaml>`), commit, and add a line to `DECISIONS.md` only if the result changes a decision. Delete failed runs; never keep partial ones.

## Asher's decisions (Oct 15)

Plan 01 copies these into `DECISIONS.md`, deleting the **[REVIEW]** items they settle.

1. **Pac-Man horizons:** add 30, 50, 100 and 200 to `pacman_transfer.csv`. Horizons 50, 100 and 200 are sealed as Pac-Man's test set (in `configs/transfer/default.yaml`, in the same commit that adds them); 30 is a validation target. Plan 02 first checks that PRISM solves 200 in reasonable time.
2. **Pac-Man threshold:** keep 0.56. Certified counts use it. The Pac-Man headline is a crash-probability-against-horizon figure (ours, 1-2-3-Go! reimplemented, PPO, the optimum, 1-2-3-Go!'s published points).
3. **SEAMS reports validation numbers only**, labelled as such. The test set stays sealed through SEAMS.
4. **Thresholds for the new benchmark models** (plan 08): the optimum plus 5% of it per instance (minus 5% when maximizing).
5. **Sonnet: none for now.** Haiku only until we truly think we have found an improvement or the final method. Then ask Asher.
6. **`closer` stays in the configuration we claim, for now.** Removing it may be tried again later, so the no-`closer` benchmark and ablation cells stay.
7. **Feedback in feature space: tested and dropped from the core** (Asher, later on Oct 15). Plan 04 found it worse than coordinates (grids −0.50, families −2.13, Pac-Man level), so the core keeps coordinate feedback (`feedback.space: states`). Feature space stays an ablation cell.

## Still open

8. **Simulation feedback instead of model checking** (plan 04, step 5). Each round, the loop would *simulate* the rules on the training instances instead of asking PRISM: run, say, 1,000 sampled episodes per instance, estimate each requirement's probability from them, and tell the LLM those estimates plus one failed episode. PRISM would only certify the final list. It answers "does the exact verifier help, or would any way of scoring the rules do?". That is the closest our loop gets to related work that debugs LLM policies by running them. Plan 12 already makes the outside-baseline version of this: LLM-written programs with simulation feedback. **Recommendation: don't build it now.** Revisit for ICAPS if plan 12 lands close to us.
9. ~~Commit `plans/`?~~ Committed on `rule-language`.
