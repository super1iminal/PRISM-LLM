"""The single-threaded schedulers (docs/scheduler.md): PRISM calls as background processes, lockstep's PRISM
phase, and the event scheduler. Toy instances stand in for the planner; their "PRISM" calls are small
Python processes, so these tests need no PRISM. The planner itself is covered by the replays in
test_scheduler.py and the recorded PRISM commands in test_prism_calls.py."""
import io
import json
import logging
import random
import subprocess
import sys
import threading
import time

import pytest

import core.prism
import core.scheduler
from config import load_config
from core.prism import PrismCall, PrismProcess, run_call
from core.scheduler import EventScheduler, LockstepScheduler, drive
from core.tasks import LLMError, TaskFactory
from fakes import FakeBackend

LLM = load_config().llm
LOG = logging.getLogger("test_event_scheduler")
SCHEDULERS = [LockstepScheduler, EventScheduler]


def py(code: str, timeout: float = 60) -> PrismCall:
    """A PRISM call that runs a short Python program instead of PRISM."""
    return PrismCall((sys.executable, "-c", code), timeout)


class Toy:
    def __init__(self, id, rounds=2, prism_seconds=0.0):
        self.id, self.rounds, self.prism_seconds = id, rounds, prism_seconds


def toy_steps(instance, logger):
    """Per round: one PRISM call, then one LLM task whose prompt carries the PRISM output. Records the
    thread that runs each step, and fails on an answer "boom" or a failed task."""
    tasks, answers, threads = TaskFactory(LLM, instance.id), [], {threading.get_ident()}
    for r in range(1, instance.rounds + 1):
        out = yield py(f"import time; time.sleep({instance.prism_seconds}); print('{instance.id}:{r}')")
        threads.add(threading.get_ident())
        result = yield tasks.make(f"{instance.id} round {r} after {out.strip()}", None, round=r)
        threads.add(threading.get_ident())
        if result.error is not None:
            raise LLMError(result.error)
        if result.text == "boom":
            raise ValueError("planner crashed")
        answers.append(result.text)
    return {"success": True, "answers": answers, "threads": threads, "iterations": []}


def echo(task):
    return f"answer to {task.prompt}"


def solve_with(kind, instances, backend, slots, **kwargs):
    return kind(toy_steps, backend, slots, lambda i: LOG, **kwargs).run(instances)


def expected(instance):
    return [f"answer to {instance.id} round {r} after {instance.id}:{r}" for r in range(1, instance.rounds + 1)]


# ---------------------------------------------------------------- PRISM calls as background processes

def test_a_background_process_reads_its_output_like_subprocess_run():
    call = py(r"import sys; sys.stdout.write('a\r\nb\rc\n'); sys.stdout.flush(); sys.stderr.write('err\n')")
    process = PrismProcess(call)
    while not process.done():
        time.sleep(0.01)
    assert process.output() == run_call(call)
    assert run_call(call).count("\n") == 4 and "\r" not in run_call(call)


def test_a_background_process_is_killed_at_its_timeout():
    start = time.monotonic()
    process = PrismProcess(py("import time; time.sleep(30)", timeout=0.3))
    while not process.done():
        time.sleep(0.01)
    assert time.monotonic() - start < 5
    with pytest.raises(subprocess.TimeoutExpired):
        process.output()
    assert process._proc.poll() is not None


@pytest.mark.parametrize("kind", SCHEDULERS)
def test_a_timeout_is_thrown_into_the_instance(kind):
    def steps(instance, logger):
        try:
            yield py("import time; time.sleep(30)", timeout=0.3)
        except subprocess.TimeoutExpired:
            return {"success": False, "timed_out": True, "iterations": []}
        return {"success": True, "iterations": []}
    results = kind(steps, FakeBackend(echo), 1, lambda i: LOG).run([Toy("t")])
    assert results["t"]["timed_out"]


# ---------------------------------------------------------------- the same answers as one at a time

@pytest.mark.parametrize("kind", SCHEDULERS)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_instance_gets_its_own_results_whatever_the_order(kind, seed):
    """Random LLM delays and PRISM durations: each instance sees its own PRISM outputs and answers."""
    rng = random.Random(seed)
    instances = [Toy(f"i{k}", rounds=rng.randint(1, 4), prism_seconds=rng.uniform(0, 0.2)) for k in range(7)]
    results = solve_with(kind, instances, FakeBackend(echo, delay=0.1), slots=3)
    for inst in instances:
        assert results[inst.id]["answers"] == expected(inst)
        assert results[inst.id] == {**drive(toy_steps(inst, None), lambda t: FakeBackend(echo).execute(t)),
                                    "total_time": results[inst.id]["total_time"], "instance": inst.id}


@pytest.mark.parametrize("kind", SCHEDULERS)
def test_planner_code_runs_in_one_thread(kind):
    """The point of both schedulers: only the calling thread ever runs an instance's code."""
    instances = [Toy(f"i{k}", rounds=3, prism_seconds=0.05) for k in range(5)]
    results = solve_with(kind, instances, FakeBackend(echo, delay=0.05), slots=3)
    assert all(r["threads"] == {threading.get_ident()} for r in results.values())


@pytest.mark.parametrize("kind", SCHEDULERS)
def test_no_prism_call_runs_in_the_scheduler_thread(kind, monkeypatch):
    """PRISM calls start as background processes; nothing waits on subprocess.run."""
    def blocked(call):
        raise AssertionError("a PRISM call ran in place")
    monkeypatch.setattr(core.prism, "run_call", blocked)
    monkeypatch.setattr(core.scheduler, "run_call", blocked)
    results = solve_with(kind, [Toy("a"), Toy("b")], FakeBackend(echo), slots=2)
    assert all(r["success"] for r in results.values())


@pytest.mark.parametrize("kind", SCHEDULERS)
def test_prism_calls_of_different_instances_run_at_once(kind):
    instances = [Toy(f"i{k}", rounds=1, prism_seconds=1.0) for k in range(4)]
    start = time.monotonic()
    results = solve_with(kind, instances, FakeBackend(echo), slots=4)
    assert all(r["success"] for r in results.values())
    assert time.monotonic() - start < 3.0   # one after another would take at least 4 s


def test_the_event_log_matches_lockstep_by_task_id():
    """Same tasks and answers; only the order of the lines differs."""
    instances = [Toy(f"i{k}", rounds=k % 3 + 1, prism_seconds=0.02 * k) for k in range(5)]
    logs = {}
    for kind in SCHEDULERS:
        log = io.StringIO()
        solve_with(kind, instances, FakeBackend(echo, delay=0.05), slots=2, task_log=log)
        lines = [json.loads(line) for line in log.getvalue().splitlines()]
        logs[kind] = sorted(((l["task"], l["result"]["text"]) for l in lines), key=lambda x: x[0]["id"])
    assert logs[LockstepScheduler] == logs[EventScheduler]
    assert len(logs[EventScheduler]) == sum(i.rounds for i in instances)


# ---------------------------------------------------------------- what only the event scheduler does

def test_a_slow_instance_does_not_hold_back_the_others():
    """Lockstep waits for the slowest answer in every batch; the event scheduler does not."""
    def answer(task):
        time.sleep(0.8 if task.meta["instance"] == "slow" else 0.01)
        return echo(task)
    instances = [Toy("slow", rounds=1), Toy("fast", rounds=4)]
    finished = {}
    for kind in SCHEDULERS:
        order = []
        solve_with(kind, instances, FakeBackend(answer), slots=2, on_finish=lambda inst, r: order.append(inst.id))
        finished[kind] = order
    assert finished[EventScheduler] == ["fast", "slow"]      # four rounds done before the slow answer
    assert finished[LockstepScheduler] == ["slow", "fast"]   # round 1 waits for it


def test_at_most_slots_instances_are_in_flight():
    running, peak, lock = [0], [0], threading.Lock()

    def answer(task):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return echo(task)
    instances = [Toy(f"i{k}", rounds=2) for k in range(6)]
    results = solve_with(EventScheduler, instances, FakeBackend(answer), slots=2)
    assert len(results) == 6 and peak[0] == 2


def test_failures_stay_local():
    def answer(task):
        if task.meta["instance"] == "crash":
            return "boom"
        if task.meta["instance"] == "llm_error":
            raise TimeoutError("backend down")
        return echo(task)
    instances = [Toy("crash"), Toy("ok1"), Toy("llm_error"), Toy("ok2")]
    results = solve_with(EventScheduler, instances, FakeBackend(answer, delay=0.02), slots=2)
    assert results["crash"]["error"] == "ValueError: planner crashed"
    assert results["llm_error"]["error"] == "TimeoutError: backend down"
    assert results["ok1"]["answers"] == expected(Toy("ok1")) and results["ok2"]["answers"] == expected(Toy("ok2"))


class Stop(BaseException):
    """Escapes the schedulers' per-instance error handling, like a Ctrl+C."""


@pytest.mark.parametrize("kind", SCHEDULERS)
def test_running_prism_processes_are_killed_when_the_scheduler_stops(kind, monkeypatch):
    started = []

    class Tracked(PrismProcess):
        def __init__(self, call):
            super().__init__(call)
            started.append(self)
    monkeypatch.setattr(core.scheduler, "PrismProcess", Tracked)

    def steps(instance, logger):
        yield py("import time; time.sleep(30)" if instance.id == "long" else "pass")
        if instance.id == "short":
            raise Stop()
        return {"success": True, "iterations": []}
    start = time.monotonic()
    with pytest.raises(Stop):
        kind(steps, FakeBackend(echo), 2, lambda i: LOG).run([Toy("long"), Toy("short")])
    assert time.monotonic() - start < 10
    assert len(started) == 2 and all(p._proc.poll() is not None for p in started)


def test_submit_runs_execute_off_the_calling_thread():
    callers = []
    backend = FakeBackend(lambda task: callers.append(threading.get_ident()) or "ok")
    task = TaskFactory(LLM, "x").make("p", None)
    assert backend.submit(task).result().text == "ok"
    assert callers != [threading.get_ident()]
    backend.close()
    assert backend._submit_pool is None
