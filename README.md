# PRISM-Guided-Learning

LLM-based planning with probabilistic verification. The LLM writes a **symbolic, possibly partial policy** (ordered
`condition -> action` rules over features of the state) for a user-specified MDP. PRISM checks it in the worst case
over all completions, and the loop revises the whole list from every training member's failures. The frozen list is
then certified unchanged on larger held-out instances, and a library of certified lists serves new instances
(retain, switch, or generate when none is certified).

Predecessor paper: [LLM-Based Grid-World Path Planning With Probabilistic Model Checking](https://doi.org/10.1145/3803437.3806714).

## Requirements

- Python 3.11+ and `pip install -r requirements.txt`
- [PRISM](https://www.prismmodelchecker.org/download.php) 4.10, with `prism` on `PATH` or `PRISM_PATH` set
- An [OpenRouter](https://openrouter.ai) API key for the core conditions (Claude Haiku 5.5; docs/openrouter.md), or
  [Ollama](https://ollama.com) for the default local model in `configs/run/default.yaml`

## Layout

```
PRISM-Guided-Learning/
  src/
    core/              the domain-agnostic approach: rules, PRISM runner, verifier (exact solvers), planner (the
                       loop), failure feedback, the pick and repair, prompts (core/templates/), LLM tasks, backends,
                       schedulers
    baselines/         1-2-3-Go! trees, PPO, the hand-written template, rule-list search, random policies
    experiments/       the run store and the RQ1 comparison / RQ2 ablation harnesses
    legacy/            the predecessor's gridworld model (read by tests and old result files)
    run_symbolic.py    one run of a condition;  run_ablation.py: conditions x seeds;  run_search.py: runs, then their
                       held-out checks
    transfer.py        a run's frozen rules, certified on held-out instances (the loop's solver)
    recertify.py       every check again with the exact solvers (the paper's certification rule)
    determinize.py     a run's `any` rules fixed in the order its pick chose
    run_baseline.py    a baseline, fitted and certified
    run_experiment.py  plans, launches and reports an experiment (configs/experiments/)
    sequential.py      the sequential adaptation study (configs/sequential/);  sequential_report.py: its report
    budget_transfer.py, budget_ii.py, equal_spend.py   budget curves and equal-spend estimates from finished runs
    calibrate.py, ceilings.py   threshold calibration, and what any controller can achieve per instance
  configs/             run/ (default.yaml + conditions/), experiments/, baselines/, sequential/, transfer/, ablation/
                       (docs/config.md)
  domains/             case studies (domains/README.md): gridworld/ (reference), pacman/ (QVBS), mer/ (Mars
                       Exploration Rovers), uuv/, consensus/
  viz/                 paper_results.py (the evaluation's tables and figures, out/paper/), theme.py
  tests/
  out/results/         core/ (the RQ1 runs), ablations/, baselines/, experiments/ (report pages), sequential/
  out/paper/           tables and figures for the paper (RESULTS-NOTES.md)
DECISIONS.md           design decisions
docs/                  semantics.md (formal semantics), config.md (all settings), experiments.md (research questions,
                       the run store), scheduler.md, openrouter.md, related_work_survey.md and positioning.md
```

Results of older versions of the approach (the previous loop, the legacy planner, the qwen-era runs) were removed from
the tree on Oct 19, 2026; they are in the git history.

## Running (from `PRISM-Guided-Learning/`, with the OpenRouter key in the environment)

```bash
# the core condition on random grids, 3 repeats, then each run's held-out checks
python src/run_search.py port_ts_history --unseeded 3 --targets 12-19 --determinize --max-parallel 1 --transfer-workers 1
```

```bash
# the RQ1 report over the run store (add --run to launch missing runs, --check to certify them)
python src/run_experiment.py configs/experiments/rq1_comparison.yaml
```

```bash
# certify a run's frozen lists with the exact solvers
python src/recertify.py run out/results/core/port_ts_history/seed_none_1 --rules determinized --targets 12-19
```

```bash
python src/sequential.py configs/sequential/shift_v0.yaml --mode frozen
```

```bash
python viz/paper_results.py
```

```bash
python -m pytest -q tests
```
