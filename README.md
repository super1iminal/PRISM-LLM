# Verified Reuse of LLM-Written Controllers for Self-Adaptive Systems

This replication package contains the implementation, configurations and results of the paper.

## Requirements

- Python 3.11+ with the packages in `requirements.txt`
- Java 11+ (needed by PRISM)
- PRISM Model Checker 4.10.1
- An OpenRouter API key (the LLM runs: Claude Haiku 5.5, GPT-6 Luna, Gemini 3.8 Flash)
- Optional: Docker, only for the Storm stage of `src/certs.py` (image `movesrwth/storm:stable`). Nothing else uses
  Storm.

## Installation

1. Download the replication package
2. Install the Python packages: `pip install -r requirements.txt`
3. Install PRISM 4.10.1 (https://www.prismmodelchecker.org/download.php). On Linux:
   `tar xf prism-4.10.1-linux64-x86.tar.gz && cd prism-4.10.1-linux64-x86 && ./install.sh`
4. Put PRISM's `bin/` on `PATH`, or set `PRISM_PATH` to the `prism` executable
5. Set `OPENROUTER_API_KEY` to your OpenRouter key: `export OPENROUTER_API_KEY=...`

## Structure

- `artifact/src`: source (`core/` the approach, `baselines/`, `experiments/`)
- `artifact/configs`: run conditions, baselines, experiments, sequential studies
- `artifact/domains`: case studies and their datasets
- `artifact/tests`: tests
- `artifact/out/paper`: the paper's tables and figures, and `certs.csv` (every verification verdict)
- `artifact/out/results`: the stored runs
  - `core/`: Full with Claude Haiku 5.5, with its NoRepair and OneShot lists (RQ1, RQ4)
  - `ablations/`: GPT-6 Luna, Gemini 3.8 Flash, Restart and earlier loop variants (RQ1, RQ4); `B2`, `D7`, `R1`, `S1`
    are replayed by the tests
  - `baselines/`: DT-raw, DT-voc, RL and the other baselines (RQ1)
  - `sequential/`: adaptation sequences (RQ3)
  - `ceilings/`: the best any controller can do on each grid-world instance

The paper's numbers come from `out/paper/`, built by `viz/paper_results.py` from `certs.csv` and the stored runs.
Reports written by `src/run_experiment.py` use the loop's own solver and may differ slightly.

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

# synthesis vs. verification time (Fig. 7), measured again into a new file that viz/paper_results.py then reads
python src/synthesis_time.py --out out/paper/tables/tab_rq2-synthesis-time.rerun.csv

# the paper's tables and figures from the stored results
python viz/paper_results.py

# tests (tests that need PRISM or an API key are skipped without them)
python -m pytest -q tests
```
