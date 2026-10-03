# Configuration

Every setting that affects a run, in one place. **Today these are scattered across the code (the "Where today" column). Phase A0 in `docs/plan.md` moves them into one config file.** Keep this table in sync with that file.

## Target layout (A0)
- `PRISM-Guided-Learning/configs/default.yaml`: every setting below with its default, grouped as in the tables.
- `PRISM-Guided-Learning/configs/conditions.yaml`: named conditions (B1, B2, R1–R5, S1…, L1…), each a small override of the default.
- `src/config.py`: loads the default, applies a condition plus any CLI overrides, and validates against dataclasses (unknown keys are errors).
- Every run writes its fully resolved config to `<run>/config.json`, so any result can be traced to its exact settings.
- Domain constants that **define the problem** (thresholds, dynamics) stay in the domain's files. They're listed here for reference but aren't run settings.

## LLM
| Key | Default | Where today | Ablated? |
|---|---|---|---|
| `llm.model` | `qwen3:14b-q4_K_M` | `settings.py` (or `$OLLAMA_MODEL`) | deferred (stronger model) |
| `llm.think` | `false` | `settings.py` | no |
| `llm.num_ctx` | 16384 | `settings.py` | no |
| `llm.num_predict` | 8192 | `settings.py` | no |
| `llm.seed` | none (new: one per seed) | not set (new, A5) | seeds 1, 2 |
| `llm.temperature` | model default | not set | no |

## Symbolic planner
| Key | Default | Where today | Ablated? |
|---|---|---|---|
| `planner.max_rounds` | 5 | `PlannerConfig.max_attempts` | F1 (budget curves, free) |
| `planner.max_fixups` | 2 re-asks per round | `PlannerConfig.max_fixups` | no |
| `planner.retry` | `stall:2` | `PlannerConfig.stall_limit` | **R1** `never`, **R2** `stall:1`, **R3** `every:3`, **R4** `gain:0.05`, **R5** `always` |
| `planner.branch` | `joint` (new) | per-requirement (A2) | no |
| `planner.feedback` | `blame` | hard-coded | S1 `table`, R5 `none` |
| `planner.max_rules` / `max_condition_chars` | 64 / 200 | `rule_schema()` | no |
| `planner.keep_best_score` | (worst fails, best fails, worst shortfall, best shortfall) | `_score()` | no |

## Feedback (mass analysis)
| Key | Default | Where today | Ablated? |
|---|---|---|---|
| `feedback.blame` | `mass` | `MassAnalyzer` | S5 `regret` |
| `feedback.horizon` (H) | 100 steps | `PlannerConfig.horizon` | no (per-domain value is an open decision) |
| `feedback.top_k` | 10 rules / hotspots | `PlannerConfig.top_k` | no |
| `feedback.states_per_rule` | 3 | `analysis.py` | no |

## Prompt
| Key | Default | Where today | Ablated? |
|---|---|---|---|
| `prompt.catch_all_instruction` | on | `core/templates/_problem.md.j2` | done (`DECISIONS.md` catch-all ablation) |
| `prompt.examples` | on (domain's `examples.md.j2`, incl. catch-all rule) | domain template | S4 off, L2 off (legacy) |

## Domain / observation
| Key | Default | Where today | Ablated? |
|---|---|---|---|
| `domain.name` / `domain.dataset` | `gridworld` / `grid_20_balanced.csv` | CLI | no |
| `domain.visible_extra` | `[obs_idx]` for gridworld (new) | hidden today (A1) | no (decided: visible for all new runs) |
| Gridworld dynamics | 0.7 / 0.15 / 0.15 | `domains/gridworld/domain.py` | constant |
| Gridworld thresholds | goals 0.8, ordering 0.8, avoid 0.7 | `domains/gridworld/domain.py` | constant |
| UUV thresholds | per scenario | `domains/uuv/data/uuv_paper.csv` | constant |

## Verifier (PRISM)
| Key | Default | Where today | Ablated? |
|---|---|---|---|
| `prism.engine` | `explicit` (`sparse` for `multi`) | `core/prism.py` | no |
| `prism.java_max_mem` | 4g | `PrismRunner` | no |
| `prism.max_iters` | 1,000,000 | `PrismRunner` | no |
| `prism.timeout_s` | 900 | `PrismRunner` | no |
| `rules.max_enumeration` | 200,000 valuations | `rules.py` | no |

## Legacy
| Key | Default | Where today | Ablated? |
|---|---|---|---|
| `legacy.max_rounds` | 5 | CLI | no |
| `legacy.retry` | `never` | n/a | L1 `stall:2` |
| `legacy.examples` | 2 worked examples | `legacy/prompting.py` | L2 off |
| `legacy.calls` | per goal (new: per goal × obstacle phase) | `legacy/planner.py` | no |

## Run
| Key | Default | Where today | Ablated? |
|---|---|---|---|
| `run.workers` | 2 | CLI | no |
| `run.seeds` | [1, 2] | n/a (new) | n/a |
| `run.limit` | all instances | CLI | no |
