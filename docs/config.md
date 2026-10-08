# Configuration

Settings live in `PRISM-Guided-Learning/configs/`, one folder per tool. The run config says what a run does (domain, dataset, model, planner); the domains themselves are defined in `domains/<name>/`, not here.

| Folder | Used by | Contents |
|---|---|---|
| `run/` | `run_symbolic.py`, `run_legacy.py`, `run_ablation.py`, `ceilings.py` | `default.yaml` (every key in the table below) and `conditions/<name>.yaml` |
| `ablation/` | `run_ablation.py` | `default.yaml`: `seeds` (default `[1, 2]`; `--seeds` overrides; `--unseeded N` runs N repeats without a seed instead) |
| `regression/` | `regression.py` | `default.yaml`: `legacy_run`, `workers`, `epsilon`, `fallback_epsilon`, `tol_stored`, `tol_exact` (documented in the file) |
| `plot/` | `viz/` scripts, `compare.py` | one file per figure or report (no default): `script:` names the script it is for, then its runs, labels, title and output. Run `python viz/<script>.py configs/plot/<figure>.yaml` |

- `run/conditions/<name>.yaml`: one file per named run condition (B1, B2, R1–R5, D7, S1, S4, S5, V1, V2, L1, L2, U1, `pre_phase_a`), each a small override of the default, selected with `--condition <name>`. A new kind of run gets a new file. B2 and the conditions compared with it set `planner.retry: "stall:2"`.
- Each folder's `default.yaml` is the only place its defaults live: the schemas (`src/config.py` for the run, a dataclass in each tool) have none, so a missing key is an error, as are unknown keys and unsupported values. Tools take `--config FILE` and `--set key=value`.
- Every run writes its resolved config to `<run>/config.json`. Reports and figures read a run's dataset, model and round budget from there (`src/results_io.py`), so plot configs never repeat them. Runs without a `config.json` (the single-seed runs) read as a fixed record of their settings (gridworld, `grid_20_balanced.csv`, 5 rounds, qwen3:14b).

```bash
python src/run_symbolic.py --condition R2 --set run.limit=1 --set llm.seed=1
```

```bash
python src/run_ablation.py B2 R1 R2 R3 R4 R5
```

**Keep this table in sync with `run/default.yaml`.** Domain constants that define the problem (thresholds, dynamics) stay in the domain's own files and are listed at the end for reference.

| Key | Default | What it does | Ablated by |
|---|---|---|---|
| `approach` | `symbolic` | `symbolic` or `legacy` planner | B1 |
| `domain.name` / `domain.dataset` | `gridworld` / `grid_20_balanced.csv` | case study and instances | — |
| `domain.visible_extra` | `[obs_idx]` | extra state variables rules (and legacy) may read, if the model has them | `pre_phase_a` (hidden) |
| `llm.model` | `qwen3:14b-q4_K_M` | model name for the backend: an Ollama tag, or an OpenRouter id such as `qwen/qwen3-14b` | `sonnet5`, `sonnet5_5`, `U1_sonnet5_5` (Claude Sonnet on OpenRouter) |
| `llm.think` | `false` | thinking mode (OpenRouter: `reasoning.enabled`) | `sonnet5_5` (its endpoints refuse to turn it off) |
| `llm.num_ctx` / `llm.num_predict` | 16384 / 8192 | context and output token limits (with thinking, the output limit covers thinking and answer) | `sonnet5_5` (16384 output) |
| `llm.seed` | `null` | run seed (set per seed by `run_ablation.py`); each call uses a seed derived from it and the call's index within the instance | seeds 1, 2 |
| `llm.temperature` | `null` | `null` = the model's default | — |
| `llm.backend` | `ollama` | serving engine, `ollama` or `openrouter` (`src/core/backends/`): runs the symbolic planner's tasks, and legacy's calls through `core/llm.py` | — |
| `llm.openrouter.base_url` | `https://openrouter.ai/api/v1` | OpenRouter's OpenAI-compatible API (`openrouter` backend only, like every `llm.openrouter` key; see `docs/openrouter.md`) | — |
| `llm.openrouter.api_key_env` | `OPENROUTER_API_KEY` | environment variable holding the API key; the key itself never goes in a config or `config.json` | — |
| `llm.openrouter.providers` | `[]` | provider slugs to use, in order, with no fallback to others (e.g. `[deepinfra]`); `[]` lets OpenRouter pick per request, so runs may mix providers | — |
| `llm.openrouter.quantizations` | `[]` | accepted weight precisions (e.g. `[fp8, bf16]`); `[]` = any | — |
| `llm.openrouter.timeout_s` / `max_retries` | 600 / 2 | per-request timeout; the client's retries on rate limits and server errors | — |
| `planner.max_rounds` | 5 | generate → verify rounds per instance | F1 (budget curves, free), D7 (7 rounds) |
| `planner.max_fixups` | 2 | re-asks per round for invalid answers | — |
| `planner.retry` | `gain:0.05` | when to start over from the initial prompt: `stall:k`, `never`, `every:k`, `gain:ε`, `always`. The default is R4's, which won the retry sweep; B2 and its comparisons use `stall:2` | R1–R5 |
| `planner.branch` | `joint` | REFINE vs EXTEND: one completion must pass every requirement (`joint`), or each on its own (`per_requirement`) | `pre_phase_a` |
| `planner.feedback` | `blame` | `blame`: REFINE/EXTEND prompts with blame; `table`: results table and previous rules only, no branch | S1 |
| `planner.max_rules` / `max_condition_chars` | 64 / 200 | schema caps (stop repetition loops) | — |
| `feedback.blame` | `mass` | blame signal: `mass`, `regret` (one-step regret, no occupancy), `random` (placebo), `none` (no blame section) | S5, V1, V2 |
| `feedback.horizon` | 100 | mass-analysis occupancy horizon in steps, or `domain` for the domain's own (UUV: the mission deadline, 30 / 70; U1 sets it) | — |
| `feedback.top_k` / `states_per_rule` | 10 / 3 | how much blame / how many hotspots the prompt shows | — |
| `prompt.catch_all_instruction` | `true` | "cover every state, e.g. end with `true -> …`" | catch-all ablation (single-seed runs) |
| `prompt.examples` | `true` | include the domain's `examples.md.j2` | S4 |
| `prism.method` | `gaussseidel` | PRISM solver; plain value iteration fails on periodic chains (rules that read `obs_idx`) | — |
| `prism.fallback_methods` | `[modpoliter]` | tried in order when `method` does not converge (seen on a real qwen rule set) | — |
| `prism.multi_engine` / `multi_method` | `sparse` / `lp` | joint queries (the explicit engine can't do them; LP is exact) | — |
| `prism.java_max_mem` / `max_iters` / `timeout_s` | 4g / 1,000,000 / 900 | PRISM limits | — |
| `prism.exact_check` | `true` | re-verify the final policy with interval iteration; the run reports those values (the loop's are kept as `loop_best` / `loop_worst`). Seconds per policy, minutes on chains that leak probability slowly. Runs whose `config.json` has no `exact_check` report the loop's values | — |
| `prism.exact_epsilon` / `exact_fallback_epsilon` / `exact_max_iters` | `"1e-9"` / `"1e-12"` / 100,000,000 | interval iteration's precision; Gauss-Seidel's where interval iteration does not converge; their iteration cap | — |
| `rules.max_enumeration` | 200,000 | state-space size up to which first-match guards are simplified | — |
| `legacy.max_rounds` | 5 | legacy rounds; with `obs_idx` visible, one call per (goal, obstacle phase) | — |
| `legacy.retry` | `never` | `stall:k`: after k rounds without improvement, the next round uses the initial prompt | L1 |
| `legacy.examples` | `true` | the two worked examples in legacy's initial prompt | L2 |
| `run.workers` / `run.limit` | 2 / `null` | parallel instances (threads, or the lockstep batch size) / first N instances only | — |
| `run.scheduler` | `threads` | symbolic only. `threads`: each worker solves an instance and calls the LLM itself. `lockstep`: every step sends one task per active instance as one batch and logs it to `<run>/llm_tasks.jsonl`; with Ollama, batches only run in parallel if `OLLAMA_NUM_PARALLEL` ≥ `run.workers` | — |


**Domain constants (not run settings):** gridworld dynamics 0.7/0.15/0.15 and thresholds (goals 0.8, ordering 0.8, avoid 0.7) in `domains/gridworld/domain.py`; UUV thresholds per scenario in `domains/uuv/data/uuv_paper.csv`.
