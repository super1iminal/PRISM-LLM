"""The interface every LLM backend implements: run a batch of tasks, return their results in order."""
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

from core.tasks import LLMResult, LLMTask

SUBMIT_WORKERS = 64   # tasks `submit` runs at once; the event scheduler keeps at most run.workers in flight


@dataclass(frozen=True)
class BackendInfo:
    name: str
    max_batch: Optional[int] = None            # largest batch per execute_batch call; None = no limit
    supports_schema: bool = True               # constrained JSON decoding
    supports_seed: bool = True


class LLMBackend(ABC):
    info: BackendInfo

    @abstractmethod
    def execute_batch(self, tasks: List[LLMTask]) -> List[LLMResult]:
        """Run every task and return one result per task, in the same order. A task that fails gets
        `LLMResult.error` set; one failure must not raise or lose the rest of the batch.

        The threads scheduler (and legacy's `LLMClient`) call one shared backend from several threads at
        once, so this must be safe for concurrent calls; lockstep calls it one batch at a time."""

    def execute(self, task: LLMTask) -> LLMResult:
        """Run a single task."""
        return self.execute_batch([task])[0]

    def submit(self, task: LLMTask) -> "Future[LLMResult]":
        """Start a single task without waiting for it; the future's result is its `LLMResult`. The event
        scheduler uses this from its one thread. By default, `execute` runs on a worker thread the
        backend owns, so `execute` must be safe for concurrent calls (as the threads scheduler requires).
        A backend with its own asynchronous client can override it."""
        if getattr(self, "_submit_pool", None) is None:
            self._submit_pool = ThreadPoolExecutor(max_workers=SUBMIT_WORKERS,
                                                   thread_name_prefix=f"{self.info.name}-submit")
        return self._submit_pool.submit(self.execute, task)

    def close(self) -> None:
        """Release clients, servers or GPU memory held by the backend. Subclasses that override this
        call `super().close()`, which stops `submit`'s worker threads."""
        if getattr(self, "_submit_pool", None) is not None:
            self._submit_pool.shutdown(wait=True)
            self._submit_pool = None
