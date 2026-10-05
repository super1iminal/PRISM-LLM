# CLAUDE.md

LLM-based planning with probabilistic verification. An LLM writes a **symbolic, possibly partial policy** (ordered `condition -> action` rules, first match wins) for a user-specified MDP. PRISM checks it in the best and worst case over all completions, and the loop refines or extends the rules. See `README.md` for layout and `DECISIONS.md` for design choices, results and open TODOs. **Read `DECISIONS.md` and `docs/plan.md` before changing the approach.**

## Docs (`docs/`)
- `plan.md`: the current work plan. Code changes (Phase A) all land before any run.
- `semantics.md`: the formal semantics of rule sets, the induced MDP and the loop branch. **Update it whenever the inputs, the rule language or the REFINE/EXTEND branch change.**
- `config.md`: every run setting, its default, where it lives and whether it is ablated. Keep it in sync with `configs/`.
- `testing_harness.md`: how the LLM task / backend / lockstep change was tested (replays of saved runs, scheduler tests, mutation checks, what is not covered).
- `ablations.md` + `ablation_run.png` / `ablation_not_run.png`: the ablation conditions, run and not run. Regenerate the PNGs with `viz/plot_ablation_grid.py` after editing its tables. Results: `PRISM-Guided-Learning/out/results/ablations/summary/SUMMARY.md` (`viz/ablation_summary.py`).

## Where things live
- `PRISM-Guided-Learning/src/core/`: the domain-agnostic approach (rules, PRISM runner, verifier, mass analysis, planner, prompt templates in `core/templates/`). **No domain-specific code here.** If a domain truly needs a core change, make it generic and log it in `DECISIONS.md`.
- LLM calls in the symbolic planner go through **tasks**: `core/planner.py` yields an `LLMTask` (`core/tasks.py`) and is sent an `LLMResult`; a backend (`core/backends/`, one class per serving engine, `llm.backend`) runs them; `core/scheduler.py` drives instances one task at a time or in lockstep batches (`run.scheduler`). Never call a model from the planner directly. The legacy planner still uses `core/llm.py`.
- `PRISM-Guided-Learning/domains/<name>/`: one directory per case study. `domains/README.md` describes the contract; `domains/gridworld/` is the reference implementation. `domains/uuv/` is the second case study (the pipeline-inspection AUV from arXiv:2308.14663); its bare MDP must keep reproducing the paper's numbers (`tests/test_uuv.py`), and `domains/uuv/data/calibrate.py` shows what its thresholds are calibrated against.
- `PRISM-Guided-Learning/src/legacy/`: the old per-state gridworld planner, kept only as a baseline. Only change it to keep the comparison fair (e.g. the obstacle-visibility switch in `docs/plan.md` A1).
- `PRISM-Guided-Learning/viz/`: plotting (`plot_comparison.py`: legacy vs one symbolic run; `plot_runs.py`: legacy vs up to two symbolic runs, e.g. ablations).
- `PRISM-Guided-Learning/out/results/<run>/`: run outputs (parquet + per-sample JSON in `outputs/`).

## Environment (Windows)
- Python venv at the repo root: `.venv/Scripts/python`. Install with `.venv/Scripts/python -m pip install -r requirements.txt`.
- PRISM 4.10.1 is on PATH as `prism` (`prism.bat`), or set `PRISM_PATH`.
- Ollama serves `qwen3:14b-q4_K_M` locally on a 16 GB RTX 4080 Super. It needs ~11.7 GB of VRAM, so **a running game or other GPU app will crash runs with CUDA out-of-memory**. Check `nvidia-smi` first. Run settings are in `configs/default.yaml` (thinking off, 16k context); `src/settings.py` only holds paths.
- With `run.scheduler: lockstep`, the Ollama backend sends each batch as concurrent requests. They only run in parallel if the server is started with `OLLAMA_NUM_PARALLEL` ≥ the batch size (`run.workers`), which costs extra VRAM; otherwise Ollama queues them.
- Run scripts from `PRISM-Guided-Learning/`, e.g. `../.venv/Scripts/python src/run_symbolic.py --domain gridworld --data grid_20_balanced.csv --out out/results/<name>`.

## Commands
- Tests (need PRISM; a few minutes, most of it the saved-run replays in `tests/test_scheduler.py`, skip them with `-k "not replay"`): `../.venv/Scripts/python -m pytest -q tests`
- New approach: `src/run_symbolic.py --domain <name> --data <dataset> [--limit N] --workers 2 --out out/results/<name>`
- Lockstep batches instead of threads: add `--set run.scheduler=lockstep` (tasks logged to `<run>/llm_tasks.jsonl`)
- Legacy baseline (gridworld only): `src/run_legacy.py`
- Regression (legacy policies reproduced in the new pipeline): `src/regression.py --legacy-run out/results/legacy_grid20`
- Conditions × seeds: `src/run_ablation.py B2 R1 --seeds 1 2 [--dry-run] [--set key=value]` (settings: `configs/`, `docs/config.md`)
- Ceilings: `src/ceilings.py --domain gridworld --data grid_20_balanced.csv`
- Comparison report: `src/compare.py`; figures: `viz/plot_comparison.py`, `viz/plot_runs.py`, `viz/plot_budget.py`

## Experiment etiquette
- **Ask before starting any LLM run.** A 20-grid run takes 30–60 min of GPU time. Use `--limit 1` for smoke tests.
- Run experiments **one after another**, never concurrently, so timings stay comparable. Keep `--workers 2` and `--max-attempts 5` unless deliberately changing them.
- If a run fails partway (OOM, loops), stop it and delete its output directory rather than keeping bad results.
- After a run: build the figure/table, add a short entry to `DECISIONS.md` (what changed, key numbers, reading), and commit.
- Results must name what differed between runs (prompt, caps, settings, scheduler). All runs so far used `run.scheduler: threads`; lockstep wall-clock times are not comparable with them (tokens are). The prompt currently includes the catch-all instruction and example; see the ablation in `DECISIONS.md`.

## Gotchas
- **Editing code:** prefer the Edit tool. Python heredocs doing `str.replace` break on `\n` escapes in this shell (it happened several times), and a silent no-op edit is easy to miss.
- **Windows directory locks:** don't `cd` into an output directory from a shell. It locks the directory and blocks later `mv`/`rm`.
- Run logs (`out/**/*.log`) are gitignored. Commit parquet, `outputs/` JSON, reports and figures. Force-add logs only when they're a run's only record.
- PRISM value iteration stops early on nested-until (LTL) properties at default settings. Use `-intervaliter` (with a `-gaussseidel -epsilon 1e-12` fallback) when exact values matter.
- LLM output is schema-capped (64 rules, 200-char conditions) in `core/planner.py`. Without the caps, qwen falls into repetition loops.
- `tests/test_scheduler.py` replays saved runs in `out/results/ablations/` (D7, B2, R1, S1) and expects every prompt byte-identical. A deliberate change to prompts or the loop will fail it: say so in `DECISIONS.md` and point the replay cases at runs made with the new code.

## Conventions
- Keep `DECISIONS.md` current. Mark uncertain choices **[REVIEW]** and delete entries once the user settles them.
- Match the surrounding code style: small modules, docstrings on public functions, few comments.
- Work on the `symbolic-policies` branch (or a branch off it); don't commit to `main`.
