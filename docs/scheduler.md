# Schedulers: threads, lockstep and event

`run.scheduler` decides how the instances of a run share the LLM and PRISM. All three run the same loop (`SymbolicPlanner.solve_steps`) and give the same per-instance results. They differ in speed, in what can run at the same time, and in what the task log records. This page explains the design and when to use which. How the change was tested: `docs/testing_scheduler.md`.

| | `threads` (default) | `lockstep` | `event` |
|---|---|---|---|
| Instances in flight | `run.workers` threads, one instance each | `run.workers` slots | `run.workers` slots |
| Who runs planner code | each worker thread | the calling thread only | the calling thread only |
| LLM calls | each thread calls `backend.execute` | one `execute_batch` per step | `backend.submit` per task |
| PRISM calls | each thread waits on its own PRISM | parallel processes, between batches | parallel processes, whenever an instance needs one |
| An instance waits for | its own results | the slowest reply in the batch, then everyone's PRISM | its own results |
| Task log (`llm_tasks.jsonl`) | none | each task with its batch number | each task in the order its result arrived (`seq`) |
| Deterministic batches | — | yes | — |
| Legacy runs | yes | no | no |

## The planner as a generator

`solve_steps(instance, logger)` is a generator. Whenever the loop needs the model it yields an `LLMTask`; whenever it needs PRISM it yields a `PrismCall` (the PRISM command line and its timeout). Whoever drives the generator runs the item and sends back the result: an `LLMResult`, or PRISM's output as text. A timeout is thrown into the generator as `subprocess.TimeoutExpired`, at the point where `subprocess.run` used to raise it.

```
solve_steps ── yield LLMTask ──▶ driver ── LLMResult ──▶ solve_steps ── yield PrismCall ──▶ driver ── output ──▶ ...
```

PRISM calls used to happen deep inside ordinary functions (`verifier.verify` → `runner.run` → `subprocess.run`). For PRISM to be yielded, every function between the planner and the subprocess became a generator too, passing the yield up with `yield from`:

- `PrismRunner.run_steps` / `check_steps` (`core/prism.py`), including the fallback over solver methods;
- `PolicyVerifier.verify_steps`, `verify_exact_steps`, `optimum_steps`, `jointly_feasible_steps`, `forced_states_steps` (`core/verifier.py`);
- the planner's `_round`, `_feedback`, `_joint_conflict` and `solve_steps`.

The blocking methods (`run`, `verify`, `optimum`, …) remain as thin wrappers that run the generator to the end. So `ceilings.py`, `regression.py` and the mass analyzer, which only reads the cached optimum, are unchanged. A `yield` now marks every point where an instance can wait, for the LLM or for PRISM. Between two yields, an instance's code runs without interruption.

The three schedulers drive these generators differently:

- **`threads`**: `planner.solve` drives one instance per worker thread with `drive()`, which answers LLM tasks with `backend.execute` and runs PRISM calls with `subprocess.run`, in that thread. This is the behaviour of all runs so far.
- **`lockstep`** and **`event`**: run every instance from the calling thread. PRISM calls start as background processes (`PrismProcess`), with output going to a temporary file rather than a pipe, so no thread needs to read it, and are checked every 10 ms. LLM tasks go to the backend, which runs them concurrently.

## The event scheduler

This is "Variant B" from the design discussion: conceptually, an array of slots, one instance per slot, and one loop that checks each slot for a ready result and hands it over.

```
slots:   [ inst 3: waiting on LLM ][ inst 7: waiting on PRISM ][ inst 1: waiting on LLM ][ ... ]
loop:    start instances while a slot is free
         wait until any slot's LLM future or PRISM process is done
         for each ready slot, in slot order:
             send the result into its generator, run it to its next yield
             start what it now waits for (backend.submit or a PRISM process), or record it as finished
         free the finished slots for the next instances
```

- **One thread runs all planner code.** Each slot holds its own generator and its own state (verifier, analyzer, kept rules, iterations, all local to `solve_steps`). Only the loop thread touches slots, so two instances' code never runs at once.
- **LLM tasks:** `LLMBackend.submit(task)` returns a future. By default it runs `execute` on worker threads the backend owns. That concurrency is inside the backend, the same requirement the threads scheduler already places on `execute`. Only the result crosses back to the loop, through the future, and backend threads never call into planner code. A backend with an asynchronous client, or a batch API that can be polled, can override `submit`.
- **PRISM calls:** each runs as its own process, so PRISM work of different instances runs in parallel on separate cores, with no thread involved. A process past `prism.timeout_s` is killed and its instance receives the timeout. If the scheduler stops on an exception, its running processes are killed.
- **Responses are handled as they arrive.** When a reply comes back, only its instance resumes; nobody waits for the slowest reply.

## Lockstep, now single-threaded

Lockstep keeps its batches: every step sends the pending LLM task of every active instance as one `execute_batch`. What changed is between batches. Previously, each instance was resumed on a thread of a pool and ran PRISM there, so several instances' planner, verifier and PRISM-runner code ran at the same time. Now each step has two phases on one thread:

1. **PRISM phase:** start a process for every instance waiting on PRISM. As each finishes, resume its instance, which may yield another PRISM call (a fallback solver, the joint query) that starts right away. The phase ends when every instance waits on its next LLM task or has finished.
2. **LLM phase:** send all pending LLM tasks as one batch, as before.

The PRISM phase always runs until every instance reaches its next LLM call, so each batch contains the same tasks as before, whatever order PRISM finishes in. The saved-run replays confirm the batches and rounds are unchanged.

## Benefits

### 1. No waiting for the slowest reply (event versus lockstep)

In lockstep, each step lasts as long as its longest reply, and every instance waits for it. With the event scheduler, as with threads, each instance moves at its own pace. A run then lasts as long as the instance with the most total work, not the sum of every step's slowest reply.

An estimate from the saved ablation runs (20 instances), assuming all instances run at once and a reply's time is proportional to its output tokens:

| run | lockstep (tokens on the critical path) | event / threads | lockstep slower by |
|---|---|---|---|
| B2 seed 1 | 16,040 | 14,949 | 7% |
| R1 seed 1 | 12,976 | 12,117 | 7% |
| D7 seed 1 | 22,394 | 15,588 | 44% |

The cost is usually small. When long replies fall in different steps for different instances, as in D7, lockstep waits on each of them and the gap grows.

### 2. PRISM overlaps generation (event versus lockstep)

In lockstep the GPU, or the API, idles during each PRISM phase, and PRISM idles during each batch. With the event scheduler, while one instance runs PRISM, others are waiting on the LLM. In the saved runs PRISM takes about 1.4 s per round against about 39 s of generation (B2, sample 0), so the overlap matters more for fast LLMs, such as a hosted API, than for local Ollama.

### 3. No shared-state races in planner code (event and lockstep versus threads)

What can run at the same time, and so where shared state must be safe:

| | What runs at the same time | Shared state must be safe in |
|---|---|---|
| threads | everything, including concurrent calls into the backend | backend, and planner, verifier and PRISM runner |
| lockstep (before this change) | instances' planner, verifier and PRISM-runner code (the resume pool) | planner, verifier, PRISM runner |
| lockstep and event (now) | nothing in planner code; instances take turns at a `yield` | only across `yield`s, which are visible in the code (and inside the backend) |

Two hypothetical bugs show the difference:

- **Threads only:** a backend that names its upload files with a counter (`n = self._batch; self._batch = n + 1`). Two threads read the same value, write the same file, and both instances receive one instance's answer. Lockstep and event never call `execute_batch` concurrently, so this cannot happen there. (Event calls `submit`, whose default runs `execute` on worker threads, so the backend's own state must still be safe there, as for threads.)
- **Old lockstep, not the new schedulers:** a `PrismRunner` that stored the current fallback method on `self` while retrying. One instance's fallback could leak into another instance's PRISM call running in parallel. In the single-threaded schedulers this can only interleave at a `yield`, and the yields are visible in the code.

### 4. Reproducible per instance

Each instance's prompts depend only on its own replies. Its verifier, analyzer and task factory are created per instance, and the analyzer's seed comes from `crc32("{seed}/{instance id}")`. So given the same replies, instance 3's round-2 prompt is identical under every scheduler. The event scheduler's task log, sorted by task id (`<instance>/r<round>/f<fixup>`), has the same content as lockstep's. Only the line order follows timing.

What lockstep adds is a fixed grouping into batches. That matters only for the model's outputs on a local GPU, where the number of requests decoded together can change floating-point results slightly even with a fixed seed. On a hosted API the provider batches as it likes anyway.

## When to use which

- **Hosted API (OpenRouter) or any backend that serves concurrent requests:** `event`. No waiting on the slowest reply, PRISM overlaps generation, and planner code is single-threaded.
- **A local engine that batches a list natively** (vLLM offline, `LLM.chat(list)`), or when fixed batches matter: `lockstep`.
- **Legacy runs, and comparability with the runs so far:** `threads`, the default. Wall-clock times of the three schedulers are not comparable with each other; tokens are.

`run.workers` is the number of slots for all three. Under `event` it is also the maximum number of concurrent LLM requests: keep it within the API's rate limit.

## Limits and costs

- **Python-side work runs one instance at a time:** building models, parsing PRISM output, mass analysis. It did before too, because of Python's global interpreter lock; only PRISM itself runs in parallel, as processes.
- **The loop polls PRISM processes every 10 ms.** This is negligible next to PRISM's roughly second-long calls.
- **Some per-round times include waiting.** A round's `llm_wall_time` and `prism_time` are wall-clock times: under `event` and `lockstep` they include time the instance waited for its turn on the loop thread, and in lockstep for the rest of the batch. `llm_time` stays the tasks' own time in the backend.
- **The global order of the event log follows timing,** so it is not reproducible line by line; sorted by task id it is.

## Why not asyncio?

The event scheduler is close to what asyncio does internally: an event loop that waits on many operations and resumes whichever is ready. Python's asyncio grew out of generator coroutines driven by `yield from`. Here the generators are the coroutines, and the loop knows two kinds of wait: an LLM future and a PRISM process.

An asyncio version would have needed the same refactor to avoid threads: PRISM as `asyncio.create_subprocess_exec`, `await` through the verifier. Without that refactor it would have run each PRISM call in a worker thread with `asyncio.to_thread`, which reintroduces concurrent planner code. Once PRISM is yielded, a hand-written loop needs no new concepts beyond generators, which the planner already used, and the same generators could later be driven by asyncio if that became useful.
