# Configuration

Every setting that affects a run lives in `PRISM-Guided-Learning/configs/`:
- `default.yaml`: every key below with its default.
- `conditions.yaml`: named conditions (B1, B2, R1–R5, S1, S4, S5, V1, V2, L1, L2, U1, `pre_phase_a`), each a small override of the default.
- Schema and validation: `src/config.py`. Unknown keys and unsupported values are errors.
- Every run writes its resolved config to `<run>/config.json`.

```bash
python src/run_symbolic.py --condition R2 --set run.limit=1 --set llm.seed=1
```

```bash
python src/run_ablation.py B2 R1 R2 R3 R4 R5 --seeds 1 2
```

**Keep this table in sync with `default.yaml`.** Domain constants that define the problem (thresholds, dynamics) stay in the domain's own files and are listed at the end for reference.

| Key | Default | What it does | Ablated by |
|---|---|---|---|
| `approach` | `symbolic` | `symbolic` or `legacy` planner | B1 |
| `domain.name` / `domain.dataset` | `gridworld` / `grid_20_balanced.csv` | case study and instances | — |
| `domain.visible_extra` | `[obs_idx]` | extra state variables rules (and legacy) may read, if the model has them | `pre_phase_a` (hidden) |
| `llm.model` | `qwen3:14b-q4_K_M` | Ollama model | deferred (stronger model) |
| `llm.think` | `false` | qwen3 thinking mode | — |
| `llm.num_ctx` / `llm.num_predict` | 16384 / 8192 | context and output token limits | — |
| `llm.seed` | `null` | run seed (set per seed by `run_ablation.py`); each call uses a seed derived from it and the call's index within the instance | seeds 1, 2 |
| `llm.temperature` | `null` | `null` = the model's default | — |
| `llm.backend` | `ollama` | serving engine for the symbolic planner's tasks (`src/core/backends/`); legacy always uses `core/llm.py` | — |
| `planner.max_rounds` | 5 | generate → verify rounds per instance | F1 (budget curves, free), D7 (7 rounds) |
| `planner.max_fixups` | 2 | re-asks per round for invalid answers | — |
| `planner.retry` | `gain:0.05` | when to start over from the initial prompt: `stall:k`, `never`, `every:k`, `gain:ε`, `always`. Was `stall:2` until Oct 3 (the conditions that ran with it pin it) | R1–R5 |
| `planner.branch` | `joint` | REFINE vs EXTEND: one completion must pass every requirement (`joint`), or each on its own (`per_requirement`, used by the old runs) | `pre_phase_a` |
| `planner.feedback` | `blame` | `blame`: REFINE/EXTEND prompts with blame; `table`: results table and previous rules only, no branch | S1 |
| `planner.max_rules` / `max_condition_chars` | 64 / 200 | schema caps (stop repetition loops) | — |
| `feedback.blame` | `mass` | blame signal: `mass`, `regret` (one-step regret, no occupancy), `random` (placebo), `none` (no blame section) | S5, V1, V2 |
| `feedback.horizon` | 100 | mass-analysis occupancy horizon (steps) | — |
| `feedback.horizon_by_domain` | `{uuv: domain}` | per-domain horizon in steps, or `domain` to ask the domain (UUV: the mission deadline, 30 / 70) | — |
| `feedback.top_k` / `states_per_rule` | 10 / 3 | how much blame / how many hotspots the prompt shows | — |
| `prompt.catch_all_instruction` | `true` | "cover every state, e.g. end with `true -> …`" | done (catch-all ablation) |
| `prompt.examples` | `true` | include the domain's `examples.md.j2` | S4 |
| `prism.method` | `gaussseidel` | PRISM solver; plain value iteration fails on periodic chains (rules that read `obs_idx`) | — |
| `prism.fallback_methods` | `[modpoliter]` | tried in order when `method` does not converge (seen on a real qwen rule set) | — |
| `prism.multi_engine` / `multi_method` | `sparse` / `lp` | joint queries (the explicit engine can't do them; LP is exact) | — |
| `prism.java_max_mem` / `max_iters` / `timeout_s` | 4g / 1,000,000 / 900 | PRISM limits | — |
| `rules.max_enumeration` | 200,000 | state-space size up to which first-match guards are simplified | — |
| `legacy.max_rounds` | 5 | legacy rounds; with `obs_idx` visible, one call per (goal, obstacle phase) | — |
| `legacy.retry` | `never` | `stall:k`: after k rounds without improvement, the next round uses the initial prompt | L1 |
| `legacy.examples` | `true` | the two worked examples in legacy's initial prompt | L2 |
| `run.workers` / `run.limit` | 2 / `null` | parallel instances (threads, or the lockstep batch size) / first N instances only | — |
| `run.scheduler` | `threads` | symbolic only. `threads`: each worker solves an instance and calls the LLM itself (all runs so far). `lockstep`: every step sends one task per active instance as one batch and logs it to `<run>/llm_tasks.jsonl` | — |


**Domain constants (not run settings):** gridworld dynamics 0.7/0.15/0.15 and thresholds (goals 0.8, ordering 0.8, avoid 0.7) in `domains/gridworld/domain.py`; UUV thresholds per scenario in `domains/uuv/data/uuv_paper.csv`.
