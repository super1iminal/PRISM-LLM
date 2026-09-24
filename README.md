# PRISM-Guided-Learning

LLM-based planning with probabilistic verification refinement. The LLM writes a **symbolic,
possibly partial policy** (ordered rules over the state variables) for a user-specified MDP.
PRISM checks it in the best and worst case over all completions of the policy, and the feedback
tells the LLM whether to *refine* existing rules or *extend* coverage, and where.

Predecessor paper: [LLM-Based Grid-World Path Planning With Probabilistic Model Checking](https://doi.org/10.1145/3803437.3806714).
Its per-state approach survives as the `legacy` baseline.

## Requirements

- Python 3.11+ and `pip install -r requirements.txt`
- [PRISM](https://www.prismmodelchecker.org/download.php), with `prism` on `PATH` or `PRISM_PATH` set
- [Ollama](https://ollama.com) with the model in `src/settings.py` (default `qwen3:14b-q4_K_M`, thinking off)

## Layout

```
PRISM-Guided-Learning/
  src/
    core/            domain-agnostic approach: rules, PRISM runner, verifier, mass analysis, planner, prompts
    legacy/          previous per-state approach (gridworld only), kept as a baseline
    run_symbolic.py  new approach on any domain
    run_legacy.py    legacy approach on gridworld
    regression.py    legacy policies -> symbolic rules -> identical PRISM results
    compare.py       end-to-end comparison report
  domains/           case studies (see domains/README.md), each with its datasets:
                       gridworld/ (reference), uuv/ (pipeline-inspection AUV, Paessler et al. 2023)
  tests/
  out/results/       run outputs
DECISIONS.md         design decisions to review
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
python src/regression.py --legacy-run out/results/legacy_grid20
```

```bash
python src/compare.py --legacy out/results/legacy_grid20 --symbolic out/results/symbolic_grid20
```

```bash
python -m pytest tests
```
