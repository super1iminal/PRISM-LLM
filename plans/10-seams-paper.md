# 10 · SEAMS paper (submission Oct 23 AoE)

Read `plans/00-overview.md` first. This plan writes; it never launches runs. Numbers come only from the generated reports.

## Goal

A complete, honest SEAMS 2027 research-track submission: IEEE format, 10 pages plus 2 of references, double-anonymous. It is mainly for the reviews, so it claims only what the reports show. Asher submits.

## The paper as it is

`paper/seams27/`: `main.tex` and `sections/*.tex` (about 3,700 words). Results draft with validation numbers; figures in `figures/transfer_*.png`, which still come from `viz/plot_transfer_results.py` (numpy-model values for baselines). The build command is in `third_party/README.md` (tectonic).

## Story and claims

- **Story:** the LLM writes a generic strategy as an ordered rule list. The verifier refines it, through feedback and by resolving `any` rules into the best action order on the training instances, and certifies it on every held-out instance.
- **SEAMS angle:** a library of frozen rule lists, certified at run time on the instance met; the system adapts by switching to one that certifies (the "library" metric). On random grids, a run's four lists cover all 8 validation grids in every run.
- **Claims discipline:** `docs/positioning.md`, "What to say and not say":
  - "certified on every instance we check", never "correct for all sizes";
  - "sound", never "maximally permissive";
  - the decision-list form is not ours;
  - feedback beats restarts only where plan 03's planned comparisons show it.
- **Say plainly where we lose** (families against trees over our features and against PPO; grids without `closer`) and what the solver's choice contributes ("ties open" column; feedback-only variants in RQ2).

## Timeline

- **Oct 15–19, no numbers needed:**
  - introduction, background, approach (rule language, `any` resolution, training sets, the library), related work (from `docs/positioning.md` and `docs/related_work_survey.md`);
  - evaluation setup: benchmarks, baselines (with Pac-Man's from plan 02), metrics, the sealed test protocol;
  - threats: validation numbers only (Asher's decision: the test set stays sealed through SEAMS, so say the configuration was chosen on the same validation targets it is reported on); the solver choosing; engineered features and `closer`; LLM nondeterminism and unseeded repeats; OpenRouter provider pinning.
  - Wire the figures to plan 01's generated paths, with `\todo{}` where a figure is not ready.
- **Oct 20, evening:** freeze. Take the RQ1 and RQ2 reports as they are then. Note in a comment in `results.tex` the commit the numbers come from.
- **Oct 21–22:**
  - results: RQ1 per benchmark, the Pac-Man horizon figure, RQ2 (one compact figure or table of the key ablations: feedback against restarts at equal cost, what the verifier tells, who resolves the choices, features);
  - cost and size: tokens and dollars per set, rule counts against tree sizes;
  - the discussion.
- **Oct 23:** page limit, anonymization (no names, no repository URLs, the `anonymized` branch if an artifact link is wanted), references, a final build. Hand to Asher.

## Figures (only the best configuration of each method)

| Figure | From |
|---|---|
| RQ1 per benchmark (grids, without `closer`, families with per-group panels) | plan 01 |
| Pac-Man crash probability against horizon | plans 01 and 02 |
| Feedback against restarts, certified against dollars | plan 03 |
| One ablation summary (verifier output, choices, features) | plans 04 and 06, existing `rq2_choices` |
| Optional: one example rule list, typeset | a core run's `final_rules` |

## Rules

- Every number has a source: a report path plus the commit, kept in comments in the `.tex` files.
- The RQ names in the paper may differ from the harness's RQ1/RQ2 (the draft has RQ3 cost and RQ4 larger instances, which come from the same runs). Keep a mapping comment.
- Do not edit `docs/` or code. Report gaps to the owning plan instead.

## Done when

`main.pdf` builds within the page limit, every `\todo{}` is gone or accepted by Asher, it is anonymized, and committed.

## Files

Owns `paper/seams27/`.
