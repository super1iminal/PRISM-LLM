# How the scheduler change was tested

A record of the checks behind the `harness-optimization` commits. Those commits yield PRISM calls from the planner, make lockstep single-threaded, and add the `event` scheduler (design: `docs/scheduler.md`).

All three change *how* the loop's work is scheduled, not *what* the loop does. So the main question was: given the same inputs, does every instance send the same prompts, run the same PRISM commands, and get the same results as before?

**Environment**
- Linux container: Python 3.11, PRISM 4.10.1 (GitHub release) on `PATH`, no GPU, no Ollama.
- Every test with PRISM ran for real, including the saved-run replays.

**Commits**
1. `b36ec75`: record the baseline (below), before any code change.
2. `0f2c77d`: yield PRISM. `drive()` and the old lockstep still run each call in place, so this step changes only the mechanism.
3. The next commit: single-threaded two-phase lockstep, the `event` scheduler, and these docs.

Each step ran the full suite, including the replays, before its commit.

## 1. Baseline: the PRISM commands each instance runs

The replays check results, but yielding PRISM changes *how* PRISM is called. `tests/prism_calls.py` therefore records every PRISM command `core/prism.py` starts, by hooking `subprocess.run` and `subprocess.Popen` inside that module. Commands are normalized so they compare across runs: input files (model, properties) by name plus a hash of their content, temporary paths by name, the PRISM path as `prism`.

On the code before the change, it recorded **57 commands** for 8 cases, one instance at a time:
- the 7 replay instances (D7 seed 1: 1, 14, 15; B2 seed 2: 3, 15; R1 seed 2: 6; S1 seed 1: 0);
- a tiny-grid run with the default config (joint query, blame, retries, and the exact final check with interval iteration).

Two recordings were identical, so the baseline is deterministic. It is committed as `tests/data/prism_calls.json`.

## 2. Equivalence with the earlier code

| Check | What it compares | Where |
|---|---|---|
| Replays, one instance at a time | Every round of saved D7, B2, R1 and S1 runs: prompts, rules, PRISM values (to 1e-9), invalid answers, final rules | `test_replay_one_at_a_time` |
| Replays, lockstep | The same, through the single-threaded lockstep, with batches of at most 2 | `test_replay_lockstep` |
| Replays, event | The same, for all four cases, through the event scheduler, with every answer delayed at random (up to 0.2 s), so answers arrive out of order | `test_replay_event` (new) |
| PRISM commands, in order | Each of the 8 cases runs exactly the 57 recorded commands, in the recorded order | `test_each_instance_runs_the_recorded_prism_commands_in_order` (new) |
| PRISM commands, under the schedulers | With every instance of a replay case in flight at once, under lockstep and under event, the same commands, each as often (compared as a multiset, since instances interleave) | `test_the_schedulers_run_the_recorded_prism_commands` (new) |
| `run_symbolic` end to end | Threads, lockstep and event write the same per-round parquet (timing columns aside); lockstep and event log one line per LLM call | `test_run_symbolic_every_scheduler` |

The replays failing would mean a step changed the loop's behaviour; the PRISM-command checks would catch a change the results could hide. Examples: a skipped re-verification after a `PrismError`, a fallback solver tried in another order, the exact check using different solvers, or a cached optimum recomputed.

## 3. The single-threaded property

That only one thread runs planner code is the point of the change, so the tests check it directly rather than assuming it:

- **Replays:** in the lockstep and event replays, every thread that verifies a policy is recorded, and must be the test's own thread. Both ways of running PRISM in place (`core.prism.run_call` and the scheduler's) are replaced with a function that fails. So a PRISM call still blocking anywhere in the planner, verifier or analyzer fails the replay.
- **Toy instances** (`tests/test_event_scheduler.py`): whose code runs in one thread under both schedulers, and no PRISM call runs in place.

## 4. The new machinery (`tests/test_event_scheduler.py`)

Toy instances stand in for the planner. Their "PRISM" calls are small Python processes, so these tests need no PRISM and run on Windows too.

| Test | What it checks |
|---|---|
| `test_a_background_process_reads_its_output_like_subprocess_run` | `PrismProcess` returns exactly what `subprocess.run(text=True)` returns, `\r\n` and `\r` included, and stderr merged |
| `test_a_background_process_is_killed_at_its_timeout` | a process past its timeout is killed, and its output raises `subprocess.TimeoutExpired` |
| `test_a_timeout_is_thrown_into_the_instance` (both) | the instance receives the timeout as an exception at its `yield` |
| `test_every_instance_gets_its_own_results_whatever_the_order` (both, 3 seeds) | 7 instances, 3 slots, random rounds, PRISM durations and answer delays: each instance sees only its own PRISM outputs and answers, and its result equals driving it alone |
| `test_planner_code_runs_in_one_thread` (both) | see section 3 |
| `test_no_prism_call_runs_in_the_scheduler_thread` (both) | see section 3 |
| `test_prism_calls_of_different_instances_run_at_once` (both) | 4 one-second PRISM calls finish in under 3 s, not 4 s |
| `test_the_event_log_matches_lockstep_by_task_id` | the two task logs, sorted by task id, hold the same tasks and answers |
| `test_a_slow_instance_does_not_hold_back_the_others` | with one slow answer, event finishes the fast instance's 4 rounds first; lockstep makes it wait |
| `test_at_most_slots_instances_are_in_flight` | 6 instances, 2 slots: never more than 2 LLM tasks at once |
| `test_failures_stay_local` | an instance that crashes, and one whose backend fails, end with their errors; the others finish normally |
| `test_running_prism_processes_are_killed_when_the_scheduler_stops` (both) | an exception that escapes the scheduler (like Ctrl+C) kills the PRISM processes still running |
| `test_submit_runs_execute_off_the_calling_thread` | `LLMBackend.submit`'s default runs `execute` on the backend's worker threads, and `close()` stops them |

Also new: `tests/test_verifier.py:test_a_prism_timeout_moves_the_exact_check_to_its_fallback_solver`. It drives `verify_exact_steps` with real PRISM, throws a timeout into the interval-iteration call, and checks the exact check falls back to Gauss-Seidel with the same values. The existing fallback test only covered PRISM errors.

Several of these tests depend on wall-clock time. `tests/test_event_scheduler.py` passed in 5 runs in a row, about 33 s each.

## 5. Mutation check

Each defect below was put into the code by hand (`mutate.py` in the session's scratch space: replace one line, run the named tests with no bytecode cache, restore the file). Every one was caught.

The first round of this check left a stale `.pyc` behind. Python validates cached bytecode by file size and modification time in whole seconds, so after the same-length "fallback order" mutation was undone within the same second, later runs still loaded the mutated `prism.py`. The full suite caught it (11 failures, all from the reversed fallback order). The bytecode was cleared, the harness changed to write none, and all 11 mutations were run again from a clean state, with the results below.

| Defect | Caught by |
|---|---|
| event: resumes an instance before its result is ready | `test_event_scheduler.py` |
| event: ignores the slot limit | `test_at_most_slots_instances_are_in_flight` |
| event: leaves PRISM running when it stops | `test_running_prism_processes_are_killed_when_the_scheduler_stops` |
| lockstep: runs PRISM in place again, one call at a time | `test_no_prism_call_runs_in_the_scheduler_thread` |
| a timeout arrives as empty output instead of being thrown | `test_a_timeout_is_thrown_into_the_instance` |
| background PRISM output keeps `\r` (unlike `text=True`) | `test_a_background_process_reads_its_output_like_subprocess_run` |
| lockstep: batch order follows the PRISM phase | the lockstep tests in `test_scheduler.py` |
| exact check: a timeout no longer falls back | `test_a_prism_timeout_moves_the_exact_check_to_its_fallback_solver` |
| PRISM: solver fallback order changed | `test_prism.py` |
| planner: one PRISM call left blocking (`verify` in `_round`) | `test_replay_event` (R1) |
| event: drops the reply's token counts | `test_run_symbolic_every_scheduler` |

## 6. Full suite

| | Result |
|---|---|
| Before (`9075253`, `harness`) | 163 passed, 1 skipped |
| After yielding PRISM (`0f2c77d`) | 164 passed, 1 skipped (+ the baseline test) |
| After the schedulers (this commit) | 194 passed, 1 skipped (+ 30 new tests, run from clean bytecode caches) |

The skip is the opt-in live OpenRouter test.

## Not covered here

- **A live LLM.** Real speed with a hosted API (how much `event` gains over lockstep and threads), rate limits, and the model's nondeterminism. The OpenRouter smoke test (`docs/openrouter.md`, step 5) runs `event` live. Since hosted models do not reproduce exactly, that is a sanity check (no errors, tokens and success rate in line), not an equality.
- **Windows.** Everything ran on Linux. The toy tests use the running Python as their "PRISM", so they are portable, but `PrismProcess` (killing a process, removing its temporary directory) has not been exercised on Windows.
- **Long runs.** The full 20-instance grid has not been run under `event` or the new lockstep.
