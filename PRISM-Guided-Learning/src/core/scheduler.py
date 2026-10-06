"""Drive planner generators: one at a time (`drive`), or many in lockstep batches (`LockstepScheduler`).

A planner's `solve_steps(instance, logger)` is a generator that yields an `LLMTask` whenever it needs
the model and is sent the `LLMResult`. The lockstep scheduler keeps up to `slots` instances in flight;
each step it collects the pending task of every active instance, runs them as one backend batch,
and resumes every instance with its result (their PRISM work runs in parallel threads). Finished
instances free their slot for the next one.
"""
import json
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import time
from typing import Any, Callable, Dict, Generator, IO, List, Optional, Sequence

from core.backends.base import LLMBackend
from core.tasks import LLMError, LLMResult, LLMTask


def drive(steps: Generator, answer: Callable[[LLMTask], LLMResult]) -> Any:
    """Run one generator to the end, answering each task as it comes; return its return value."""
    try:
        task = next(steps)
        while True:
            task = steps.send(answer(task))
    except StopIteration as done:
        return done.value


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
    task: Optional[LLMTask] = None             # pending task; None once finished
    result: Optional[Dict[str, Any]] = None


class LockstepScheduler:
    """Solve instances in lockstep: every step sends one task per active instance as a single batch.

    `make_steps(instance, logger)` starts an instance's generator (e.g. `planner.solve_steps`) and
    `logger_for(instance)` gives its logger. With `task_log`, every task and result is written as a
    JSON line together with its batch number.
    """

    def __init__(self, make_steps: Callable[[Any, Any], Generator], backend: LLMBackend, slots: int,
                 logger_for: Callable[[Any], Any], task_log: Optional[IO[str]] = None,
                 on_finish: Optional[Callable[[Any, Dict[str, Any]], None]] = None):
        if slots < 1:
            raise ValueError("slots must be at least 1")
        self.make_steps = make_steps
        self.backend = backend
        self.slots = slots
        self.logger_for = logger_for
        self.task_log = task_log
        self.on_finish = on_finish
        self.batches = 0

    def run(self, instances: Sequence[Any]) -> Dict[Any, Dict[str, Any]]:
        """Solve every instance; return {instance.id: result dict}."""
        queue, results = deque(instances), {}
        with ThreadPoolExecutor(max_workers=self.slots) as pool:
            active = self._refill([], queue, results, pool)
            while active:
                tasks = [slot.task for slot in active]
                replies = self._execute(tasks)
                active = list(pool.map(self._resume, active, replies))
                active = self._collect(active, results)
                active = self._refill(active, queue, results, pool)
        return results

    def _refill(self, active: List[_Slot], queue: deque, results: Dict, pool) -> List[_Slot]:
        """Start instances until every slot is busy. Starting runs setup up to the first task."""
        new = []
        while queue and len(active) + len(new) < self.slots:
            instance = queue.popleft()
            logger = self.logger_for(instance)
            new.append(_Slot(instance, self.make_steps(instance, logger), logger))
        started = list(pool.map(self._start, new))
        return active + self._collect(started, results)

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

    def _start(self, slot: _Slot) -> _Slot:
        return self._advance(slot, lambda: next(slot.steps))

    def _resume(self, slot: _Slot, reply: LLMResult) -> _Slot:
        return self._advance(slot, lambda: slot.steps.send(reply))

    @staticmethod
    def _advance(slot: _Slot, step: Callable[[], LLMTask]) -> _Slot:
        """Run the instance up to its next task; on return or error, mark it finished."""
        try:
            slot.task = step()
        except StopIteration as done:
            slot.task, slot.result = None, done.value
        except Exception as e:
            slot.logger.error(traceback.format_exc())
            slot.task, slot.result = None, failed_result(e)
        return slot

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
        if self.task_log is not None:
            for task, reply in zip(tasks, replies):
                self.task_log.write(json.dumps({"batch": self.batches, "task": task.to_dict(),
                                                "result": reply.to_dict()}, default=str) + "\n")
            self.task_log.flush()
        self.batches += 1
        return replies
