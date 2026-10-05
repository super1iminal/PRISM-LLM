# Testing the LLM task harness (commit `ef58ee1`, branch `harness`)

This is a record of how commit `ef58ee1` was tested. That commit made the symbolic planner build LLM tasks instead of calling Ollama. It also added backends, and `run.scheduler: lockstep` alongside the original threads. The design is in `DECISIONS.md` ("LLM tasks, backends and lockstep batching"), and the tests are in `PRISM-Guided-Learning/tests/test_scheduler.py`.

**Goal:** the refactor must not change what the planner does. With the same LLM answers, every round must produce the same prompt, rules, PRISM values and invalid-answer handling as before. This must hold under both schedulers. **No LLM or GPU was used:** every test answers tasks with a scripted backend (`FakeBackend`).

## Environment
- Linux container (Claude Code cloud session), not the Windows workstation in `CLAUDE.md`.
- Python 3.11.15 in a fresh venv, with `requirements.txt` installed (pydantic 2.13.5, pandas 3.0.6, ollama 0.6.3).
- PRISM 4.10.1 (the `linux64-x86` release, the same version used for the runs) on `PATH`, with OpenJDK 21.
- Ollama was not installed and is not needed.

## 1. Baseline before any change
- **Existing suite on the unchanged code:** 49 passed in 115 s.
  - Without PRISM on `PATH`, 18 of these tests skip. Every result below is with PRISM present.
- **Replays reproduce the saved runs on the unchanged code.** A throwaway script fed the saved `raw_outputs` of a run back through the *unchanged* planner, using a stand-in LLM client. It then compared each round with the saved `outputs/sample_*.json`.
  - Fields compared: mode, prompt, rules, invalid answers, number of rules, `improved`, and best and worst values within 1e-6.
  - Every gridworld sample tried matched exactly: D7 seed 1 samples 9 and 15, B2 seed 2 sample 15, R1 seed 2 sample 6, S1 seed 1 sample 0, S5 seed 2 sample 3.
  - This establishes that saved runs are a valid reference: PRISM on Linux gives the same values as the Windows runs, and the prompts are unchanged since those runs.
  - **UUV is excluded.** The U1 sample did *not* match, because commit `bdc128f` changed the UUV description after U1 ran. The tests don't use it.

## 2. New tests (`tests/test_scheduler.py`, 14 tests)

### Equivalence with saved runs
**What the replays compare.** In both replay tests, every round is compared with the saved run on:
- round type, full prompt text, rules and number of rules
- invalid answers, `improved`, number of LLM calls and raw outputs
- reachable and uncovered situations
- best and worst values within 1e-9
- `success` and `final_rules`

**`test_replay_one_at_a_time`** (4 cases). This is the path the default `threads` scheduler uses: `planner.solve()` answering one task at a time.

| Saved run | Samples | What it exercises |
|---|---|---|
| D7 seed 1 | 1, 14, 15 | 7 rounds, extend, retries, 2 and 6 invalid answers (re-asks) |
| B2 seed 2 | 3, 15 | extend then refine, invalid answers on extend rounds |
| R1 seed 2 | 6 | four extend rounds in a row |
| S1 seed 1 | 0 | table feedback (`planner.feedback: table`) |

**`test_replay_lockstep`** (D7 and B2 cases). The same comparison through `LockstepScheduler` with 2 slots. D7 has 3 instances, so a slot must be refilled. The test also checks:
- no batch is larger than 2;
- the total number of tasks equals the saved `llm_calls`;
- `llm_tasks.jsonl` has one line per task.

### Scheduler behaviour (no PRISM needed)
These tests use a toy generator in place of the planner.

**`test_lockstep_batches_and_refill`:** 5 instances needing 1, 3, 2, 2 and 1 rounds, with 3 slots.
- The batches are exactly `[g0,g1,g2] → [g1,g2,g3] → [g1,g3,g4]`.
- All instances finish, and `on_finish` fires once per instance.
- Answers reach the right instance, and `total_time` and `instance` are set on each result.

**`test_lockstep_failures_stay_local`:**
- An instance whose planner raises ends with `ValueError: planner crashed`.
- An instance whose backend call fails ends with `ConnectionError: server gone`, the same `<Type>: <message>` text `run_symbolic` used to record.
- The third instance completes normally.

**`test_lockstep_respects_backend_max_batch`:** with 5 slots and a backend limit of 2, each step's batch of 5 is sent as 2, 2 and 1.

**`test_drive_and_old_error_format`:**
- `drive()` runs a generator to completion.
- A failed result raises `LLMError`.
- `failed_result()` keeps the old error format for both backend failures and other exceptions.

### Lockstep against one-at-a-time, with the real planner and PRISM
**`test_lockstep_matches_one_at_a_time`:** the first 4 gridworld instances, 2 rounds each, with seed 3 and fixed scripted rules.
- One instance gets an invalid first answer, which forces a re-ask.
- Lockstep (3 slots) and `planner.solve()` give identical round type, prompt, rules, best and worst values, and invalid answers for every round.

### Seeds
**`test_task_seeds_follow_the_old_formula`:**
- Seeds are `seed * 1_000_003 + n` for n = 0, 1, 2, …
- The parameters are exactly `think`, `num_ctx`, `num_predict`, `temperature` and `seed`.
- There is no seed when `llm.seed` is null.

**`test_planner_seeds_count_calls_per_instance`:** 2 instances, 2 rounds, two invalid answers per round, so 6 calls each.
- Each instance numbers its seeds from 0 to 5 across rounds and re-asks. This matches the old per-thread counter, which was reset for every instance.

### `run_symbolic` wiring
**`test_run_symbolic_both_schedulers`:** `run_symbolic.run()` on 3 instances (2 rounds each) under `threads` and under `lockstep`, with `make_backend` patched to a fake.
- Both write 3 sample JSONs and a parquet file. The two parquet files are identical once the timing columns are dropped.
- Only lockstep writes `llm_tasks.jsonl`. It has one line per LLM call and covers every instance.

## 3. Checking that the tests catch regressions
Each mutation was applied, the tests were run, and the file was restored. `git status` was clean afterwards.
- **Seed formula changed by +1** (`core/tasks.py`): both seed tests failed.
- **One extra space in `core/templates/refine.md.j2`:** `test_replay_one_at_a_time[D7/seed_1]` failed.

## 4. Results
- New tests: 13 passed in 184 s; the `run_symbolic` test was added next and passed in 18 s.
- **Full suite after the change: 63 passed (49 existing + 14 new) in 312 s.**
- `pyflakes` reported nothing on the changed files.

## Not covered
- **No live model.** The Ollama backend has not run against a server, so neither have its request options, error handling under real failures, or concurrent requests with `OLLAMA_NUM_PARALLEL`. This is `docs/plan.md` H4 and needs a go.
- **Timing.** Wall-clock times under lockstep were not measured; the tests compare results only.
- **UUV replays,** for the reason given in section 1.
- **Legacy** is unchanged and still covered only by its existing tests.

## Rerunning
From `PRISM-Guided-Learning/`, with PRISM on `PATH`:
```bash
python -m pytest -q tests/test_scheduler.py               # ~3-4 min, mostly the replays
python -m pytest -q tests/test_scheduler.py -k "not replay"
python -m pytest -q tests                                  # full suite
```
The replay tests skip if their saved run is missing from `out/results/ablations/`.
