# PRISM-Guided-Learning

LLM-based planning with probabilistic verification refinement. The LLM writes a **symbolic,
possibly partial policy** (ordered rules over the state variables) for a user-specified MDP.
PRISM checks it in the best and worst case over all completions of the policy, and the feedback
tells the LLM whether to *refine* existing rules or *extend* coverage, and where.

Predecessor paper: [LLM-Based Grid-World Path Planning With Probabilistic Model Checking](https://doi.org/10.1145/3803437.3806714).
Its per-state approach is the `legacy` baseline.

## Requirements

- Python 3.11+ and `pip install -r requirements.txt`
- [PRISM](https://www.prismmodelchecker.org/download.php), with `prism` on `PATH` or `PRISM_PATH` set
- [Ollama](https://ollama.com) with the model in `configs/run/default.yaml` (default `qwen3:14b-q4_K_M`, thinking off)

## Layout

```
PRISM-Guided-Learning/
  src/
    core/            domain-agnostic approach: rules, PRISM runner, verifier, mass analysis, planner, prompts
    legacy/          the predecessor's per-state approach (gridworld only), the baseline
    run_symbolic.py  symbolic approach on any domain
    run_legacy.py    legacy approach on gridworld
    run_ablation.py  named conditions x seeds
    ceilings.py      what any controller can achieve per instance
    regression.py    legacy policies -> symbolic rules -> identical PRISM results
    compare.py       end-to-end comparison report
    results_io.py    reading finished runs (result files, recorded settings, legacy keep-best)
  configs/           settings, one folder per tool (see docs/config.md):
                       run/ (default.yaml + conditions/), ablation/, regression/, plot/ (one per figure)
  domains/           case studies (see domains/README.md), each with its datasets:
                       gridworld/ (reference), uuv/ (pipeline-inspection AUV, Paessler et al. 2023)
  viz/               figures, each made from a configs/plot/ file (see viz/README.md)
  tests/
  out/results/       run outputs
DECISIONS.md         design decisions to review
docs/                plan.md (work plan), semantics.md (formal semantics), config.md (all settings), ablations.md (+ grids)
```

## Running (from `PRISM-Guided-Learning/`)

```bash
python src/run_legacy.py --data grid_20_balanced.csv --out out/results/legacy_grid20
```

```bash
python src/run_symbolic.py --domain gridworld --data grid_20_balanced.csv --out out/results/symbolic_grid20
```

```bash
python src/run_symbolic.py --domain uuv --data uuv_paper.csv --out out/results/symbolic_uuv
```

```bash
python src/run_ablation.py B2 R1 R2 R3 R4 R5 --dry-run
```

```bash
python src/ceilings.py
```

```bash
python src/regression.py
```

```bash
python src/compare.py configs/plot/compare_grid20.yaml
```

```bash
python viz/plot_runs.py configs/plot/catchall_ablation.yaml
```

```bash
python -m pytest tests
```
