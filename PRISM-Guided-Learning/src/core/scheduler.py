"""Drive planner generators: one at a time (`drive`), in lockstep batches (`LockstepScheduler`), or as each
result arrives (`EventScheduler`).

A planner's `solve_steps(instance, logger)` is a generator that yields an `LLMTask` whenever it needs the
model and a `PrismCall` whenever it needs PRISM, and is sent the result. `drive` runs one generator,
answering each item in turn. Both schedulers keep up to `slots` instances in flight, run from a single
thread, and start every pending PRISM call as its own process, so the PRISM work of different instances
runs in parallel while no planner code ever runs in two threads at once (docs/scheduler.md):

* lockstep: every step sends the pending LLM task of every active instance as one backend batch. After
  the batch, each instance runs its PRISM calls until it reaches its next LLM task, then the next batch
  goes out. Batches are the same whatever order PRISM finishes in.
* event: each instance moves on as soon as its own LLM result or PRISM output is ready, without waiting
  for the others. LLM tasks go to the backend one at a time (`LLMBackend.submit`).

Finished instances free their slot for the next one.
"""
import json
import subprocess
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
from time import sleep, time
from typing import Any, Callable, Dict, Generator, IO, Iterable, List, Optional, Sequence, Union

from core.backends.base import LLMBackend
from core.prism import PrismCall, PrismProcess, run_call
from core.tasks import LLMError, LLMResult, LLMTask

POLL_SECONDS = 0.01   # how often a waiting scheduler checks its PRISM processes


def drive(steps: Generator, answer: Callable[[LLMTask], LLMResult]) -> Any:
    """Run one generator to the end, answering each LLM task and running each PRISM call as it comes;
    return its return value."""
    try:
        item = run_prism_here(steps, next(steps))
        while True:
            item = run_prism_here(steps, steps.send(answer(item)))
    except StopIteration as done:
        return done.value


def run_prism_here(steps: Generator, item: Any) -> Any:
    """Run the generator's PRISM calls in this thread, one at a time, until it yields something else
    (returned) or returns (StopIteration propagates)."""
    while isinstance(item, PrismCall):
        try:
            output = run_call(item)
        except subprocess.TimeoutExpired as e:
            item = steps.throw(e)
        else:
            item = steps.send(output)
    return item


def error_message(e: Exception) -> str:
    """How a failed instance reports its error (backend failures keep the backend's own message)."""
    return str(e) if isinstance(e, LLMError) else f"{type(e).__name__}: {e}"


def failed_result(e: Exception) -> Dict[str, Any]:
    return {"success": False, "error": error_message(e), "iterations": []}


@dataclass
class _Slot:
    instance: Any
    steps: Generator
    logger: Any
    started: float = field(default_factory=time)
    task: Union[LLMTask, PrismCall, None] = None   # what the instance waits for; None once finished
    result: Optional[Dict[str, Any]] = None
    pending: Union[Future, PrismProcess, None] = None   # event scheduler: the running task or process

    def advance(self, step: Callable[[], Any]) -> None:
        """Run the instance up to its next LLM task or PRISM call; on return or error, mark it finished."""
        try:
            self.task = step()
        except StopIteration as done:
            self.task, self.result = None, done.value
        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.task, self.result = None, failed_result(e)

    def resume_with_prism(self, process: PrismProcess) -> None:
        """Send the finished process's output, or throw its timeout, into the instance."""
        try:
            output = process.output()
        except subprocess.TimeoutExpired as e:
            self.advance(lambda: self.steps.throw(e))
        else:
            self.advance(lambda: self.steps.send(output))


def _wait_any(processes: Iterable[PrismProcess], futures: Iterable[Future] = ()) -> None:
    """Return once any of the processes or futures is done."""
    processes, futures = list(processes), list(futures)
    while not (any(p.done() for p in processes) or any(f.done() for f in futures)):
        if futures:
            wait(futures, timeout=POLL_SECONDS, return_when=FIRST_COMPLETED)
        else:
            sleep(POLL_SECONDS)


class _Scheduler:
    """What both schedulers share: slots, starting instances, recording finished ones."""

    def __init__(self, make_steps: Callable[[Any, Any], Generator], backend: LLMBackend, slots: int,
                 logger_for: Callable[[Any], Any], task_log: Optional[IO[str]] = None,
                 on_finish: Optional[Callable[[Any, Dict[str, Any]], None]] = None):
        """`make_steps(instance, logger)` starts an instance's generator (e.g. `planner.solve_steps`) and
        `logger_for(instance)` gives its logger. With `task_log`, every LLM task and its result is
        written as a JSON line."""
        if slots < 1:
            raise ValueError("slots must be at least 1")
        self.make_steps = make_steps
        self.backend = backend
        self.slots = slots
        self.logger_for = logger_for
        self.task_log = task_log
        self.on_finish = on_finish
        self.prism_calls = 0

    def _start(self, instance: Any) -> _Slot:
        """A slot for `instance`, run up to its first LLM task or PRISM call."""
        logger = self.logger_for(instance)
        slot = _Slot(instance, self.make_steps(instance, logger), logger)
        slot.advance(lambda: next(slot.steps))
        return slot

    def _collect(self, slots: List[_Slot], results: Dict) -> List[_Slot]:
        """Record finished instances; return the ones still running."""
        running = []
        for slot in slots:
            if slot.task is not None:
                running.append(slot)
                continue
            slot.result["total_time"] = time() - slot.started
            slot.result["instance"] = slot.instance.id
            results[slot.instance.id] = slot.result
            if self.on_finish:
                self.on_finish(slot.instance, slot.result)
        return running

    def _log(self, key: str, value: int, task: LLMTask, reply: LLMResult) -> None:
        if self.task_log is not None:
            self.task_log.write(json.dumps({key: value, "task": task.to_dict(), "result": reply.to_dict()},
                                           default=str) + "\n")
            self.task_log.flush()


class LockstepScheduler(_Scheduler):
    """Solve instances in lockstep: every step sends one LLM task per active instance as a single batch.

    Between batches, the instances' PRISM calls run as parallel processes until every instance waits on
    its next LLM task, so a batch never depends on which PRISM process finishes first. The task log
    records each task with its batch number.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches = 0

    def run(self, instances: Sequence[Any]) -> Dict[Any, Dict[str, Any]]:
        """Solve every instance; return {instance.id: result dict}."""
        queue, results = deque(instances), {}
        active = self._refill([], queue, results)
        while active:
            replies = self._execute([slot.task for slot in active])
            for slot, reply in zip(active, replies):
                slot.advance(lambda s=slot, r=reply: s.steps.send(r))
            active = self._collect(self._run_prism(active), results)
            active = self._refill(active, queue, results)
        return results

    def _refill(self, active: List[_Slot], queue: deque, results: Dict) -> List[_Slot]:
        """Start instances until every slot is busy, each up to its first LLM task."""
        new = []
        while queue and len(active) + len(new) < self.slots:
            new.append(self._start(queue.popleft()))
        return active + self._collect(self._run_prism(new), results)

    def _run_prism(self, slots: List[_Slot]) -> List[_Slot]:
        """Run the slots' PRISM calls, all at once, until each waits on an LLM task or has finished."""
        running: Dict[int, PrismProcess] = {}
        try:
            while True:
                for i, slot in enumerate(slots):
                    if isinstance(slot.task, PrismCall) and i not in running:
                        running[i] = PrismProcess(slot.task)
                        self.prism_calls += 1
                if not running:
                    return slots
                _wait_any(running.values())
                for i in [i for i, process in running.items() if process.done()]:
                    slots[i].resume_with_prism(running.pop(i))
        finally:
            for process in running.values():
                process.kill()

    def _execute(self, tasks: List[LLMTask]) -> List[LLMResult]:
        """One step's batch, split into chunks of the backend's max_batch."""
        size = self.backend.info.max_batch or len(tasks)
        replies: List[LLMResult] = []
        for i in range(0, len(tasks), size):
            chunk = tasks[i:i + size]
            out = self.backend.execute_batch(chunk)
            if len(out) != len(chunk):
                raise RuntimeError(f"backend returned {len(out)} results for {len(chunk)} tasks")
            replies.extend(out)
        for task, reply in zip(tasks, replies):
            self._log("batch", self.batches, task, reply)
        self.batches += 1
        return replies


class EventScheduler(_Scheduler):
    """Solve instances as their results arrive: an instance moves on as soon as its own LLM result or
    PRISM output is ready, whatever the other instances are doing.

    One thread runs every instance's planner code; LLM tasks run in the backend (`LLMBackend.submit`)
    and PRISM calls as their own processes. When several results are ready at once, instances resume in
    slot order. The task log records each task in the order its result arrived (`seq`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_tasks = 0

    def run(self, instances: Sequence[Any]) -> Dict[Any, Dict[str, Any]]:
        """Solve every instance; return {instance.id: result dict}."""
        queue, results, active = deque(instances), {}, []
        try:
            while queue or active:
                while queue and len(active) < self.slots:
                    slot = self._start(queue.popleft())
                    self._dispatch(slot)
                    active.append(slot)
                active = self._collect(active, results)
                if not active:
                    continue
                _wait_any([s.pending for s in active if isinstance(s.pending, PrismProcess)],
                          [s.pending for s in active if isinstance(s.pending, Future)])
                for slot in active:
                    if slot.pending.done():
                        self._resume(slot)
                        self._dispatch(slot)
                active = self._collect(active, results)
        finally:
            for slot in active:
                if isinstance(slot.pending, PrismProcess):
                    slot.pending.kill()
        return results

    def _dispatch(self, slot: _Slot) -> None:
        """Start what the instance waits for."""
        if isinstance(slot.task, LLMTask):
            slot.pending = self.backend.submit(slot.task)
        elif isinstance(slot.task, PrismCall):
            slot.pending = PrismProcess(slot.task)
            self.prism_calls += 1
        else:
            slot.pending = None

    def _resume(self, slot: _Slot) -> None:
        """Hand the instance the result it waited for and run it up to its next task or call."""
        pending, slot.pending = slot.pending, None
        if isinstance(pending, PrismProcess):
            slot.resume_with_prism(pending)
            return
        try:
            reply = pending.result()
        except Exception as e:   # a backend must not raise, but if it does, only this instance fails
            reply = LLMResult(slot.task.id, None, error=f"{type(e).__name__}: {e}")
        self._log("seq", self.llm_tasks, slot.task, reply)
        self.llm_tasks += 1
        slot.advance(lambda: slot.steps.send(reply))
