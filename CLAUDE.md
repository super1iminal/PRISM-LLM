# CLAUDE.md

LLM-based planning with probabilistic verification. An LLM writes a **symbolic, possibly partial policy** (ordered `condition -> action` rules, first match wins) for a user-specified MDP. PRISM checks it in the best and worst case over all completions, and the loop refines or extends the rules. See `README.md` for layout and `DECISIONS.md` for design choices, results and open TODOs. **Read `DECISIONS.md` before changing the approach.**

## Where things live
- `PRISM-Guided-Learning/src/core/`: the domain-agnostic approach (rules, PRISM runner, verifier, mass analysis, planner, prompt templates in `core/templates/`). **No domain-specific code here.** If a domain truly needs a core change, make it generic and log it in `DECISIONS.md`.
- `PRISM-Guided-Learning/domains/<name>/`: one directory per case study. `domains/README.md` describes the contract; `domains/gridworld/` is the reference implementation.
- `PRISM-Guided-Learning/src/legacy/`: the old per-state gridworld planner, kept only as a baseline. Don't extend it.
- `PRISM-Guided-Learning/viz/`: plotting (`plot_comparison.py`: legacy vs one symbolic run; `plot_runs.py`: legacy vs up to two symbolic runs, e.g. ablations).
- `PRISM-Guided-Learning/out/results/<run>/`: run outputs (parquet + per-sample JSON in `outputs/`).

## Environment (Windows)
- Python venv at the repo root: `.venv/Scripts/python`. Install with `.venv/Scripts/python -m pip install -r requirements.txt`.
- PRISM 4.10.1 is on PATH as `prism` (`prism.bat`), or set `PRISM_PATH`.
- Ollama serves `qwen3:14b-q4_K_M` locally on a 16 GB RTX 4080 Super. It needs ~11.7 GB of VRAM, so **a running game or other GPU app will crash runs with CUDA out-of-memory**. Check `nvidia-smi` first. Settings are in `src/settings.py` (thinking off, 16k context).
- Run scripts from `PRISM-Guided-Learning/`, e.g. `../.venv/Scripts/python src/run_symbolic.py --domain gridworld --data grid_20_balanced.csv --out out/results/<name>`.

## Commands
- Tests (fast, ~12 s, need PRISM): `../.venv/Scripts/python -m pytest -q tests`
- New approach: `src/run_symbolic.py --domain <name> --data <dataset> [--limit N] --workers 2 --out out/results/<name>`
- Legacy baseline (gridworld only): `src/run_legacy.py`
- Regression (legacy policies reproduced in the new pipeline): `src/regression.py --legacy-run out/results/legacy_grid20`
- Comparison report: `src/compare.py`; figures: `viz/plot_comparison.py`, `viz/plot_runs.py`

## Experiment etiquette
- **Ask before starting any LLM run.** A 20-grid run takes 30–60 min of GPU time. Use `--limit 1` for smoke tests.
- Run experiments **one after another**, never concurrently, so timings stay comparable. Keep `--workers 2` and `--max-attempts 5` unless deliberately changing them.
- If a run fails partway (OOM, loops), stop it and delete its output directory rather than keeping bad results.
- After a run: build the figure/table, add a short entry to `DECISIONS.md` (what changed, key numbers, reading), and commit.
- Results must name what differed between runs (prompt, caps, settings). The prompt currently includes the catch-all instruction and example; see the ablation in `DECISIONS.md`.

## Gotchas
- **Editing code:** prefer the Edit tool. Python heredocs doing `str.replace` break on `\n` escapes in this shell (it happened several times), and a silent no-op edit is easy to miss.
- **Windows directory locks:** don't `cd` into an output directory from a shell. It locks the directory and blocks later `mv`/`rm`.
- Run logs (`out/**/*.log`) are gitignored. Commit parquet, `outputs/` JSON, reports and figures. Force-add logs only when they're a run's only record.
- PRISM value iteration stops early on nested-until (LTL) properties at default settings. Use `-intervaliter` (with a `-gaussseidel -epsilon 1e-12` fallback) when exact values matter.
- LLM output is schema-capped (64 rules, 200-char conditions) in `core/planner.py`. Without the caps, qwen falls into repetition loops.

## Conventions
- Keep `DECISIONS.md` current. Mark uncertain choices **[REVIEW]** and delete entries once the user settles them.
- Match the surrounding code style: small modules, docstrings on public functions, few comments.
- Work on the `symbolic-policies` branch (or a branch off it); don't commit to `main`.
