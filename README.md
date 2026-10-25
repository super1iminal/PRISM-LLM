# Verified Reuse of LLM-Written Controllers for Self-Adaptive Systems

This repository contains the implementation, configurations and results of the paper.

## Requirements

- Python 3.11+ with the packages in `requirements.txt`
- Java 11+ (needed by PRISM)
- PRISM Model Checker 4.10
- An OpenRouter API key (the LLM runs: Claude Haiku 5.5 / Sonnet 5.5, GPT-6 Luna, Gemini 3.8 Flash)
- Optional: Docker, only for the Storm stage of `src/certs.py` (image `movesrwth/storm:stable`). Nothing else uses
  Storm.

## Installation

1. Download the repository
2. Install the Python packages: `pip install -r requirements.txt`
3. Install PRISM 4.10 (https://www.prismmodelchecker.org/download.php). On Linux:
   `tar xf prism-4.10-linux64-x86.tar.gz && cd prism-4.10-linux64-x86 && ./install.sh`
4. Put PRISM's `bin/` on `PATH`, or set `PRISM_PATH` to the `prism` executable
5. Set `OPENROUTER_API_KEY` to your OpenRouter key: `export OPENROUTER_API_KEY=...`

## Structure

- `artifact/src`: source (`core/` the approach, `baselines/`, `experiments/`)
- `artifact/configs`: run conditions, baselines, experiments, sequential studies
- `artifact/domains`: case studies (gridworld, Pac-Man, MER, UUV, consensus)
- `artifact/out/results`: results; `artifact/out/paper`: the paper's tables and figures
- `artifact/tests`: tests

## Running (from `artifact/`)

```bash
# one condition (random grids), 3 repeats, then each run's held-out checks
python src/run_search.py port_ts_history --unseeded 3 --targets 12-19 --determinize --max-parallel 1 --transfer-workers 1

# a baseline, fitted and certified
python src/run_baseline.py configs/baselines/tree_raw_grid_large.yaml

# experiment reports over the stored runs (add --run to launch missing runs, --check to certify them)
python src/run_experiment.py configs/experiments/rq1_comparison.yaml

# certify a run's frozen lists with the exact solvers
python src/recertify.py run out/results/core/port_ts_history/seed_none_1 --rules determinized --targets 12-19

# sequential study
python src/sequential.py configs/sequential/shift_v0.yaml --mode frozen

# the paper's tables and figures from the stored results
python viz/paper_results.py

# tests (tests that need PRISM or an API key are skipped without them)
python -m pytest -q tests
```
