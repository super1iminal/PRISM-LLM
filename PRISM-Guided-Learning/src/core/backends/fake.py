"""A scripted backend for tests: answers come from a function of the task, no model involved."""
from typing import Callable, List, Optional

from core.backends.base import BackendInfo, LLMBackend
from core.tasks import LLMResult, LLMTask


class FakeBackend(LLMBackend):
    """`answer(task)` returns the raw output text; raising makes that task fail with `LLMResult.error`.
    Every executed batch is kept in `batches`, so tests can inspect what was sent together."""

    def __init__(self, answer: Callable[[LLMTask], str], max_batch: Optional[int] = None):
        self.answer = answer
        self.info = BackendInfo(name="fake", max_batch=max_batch)
        self.batches: List[List[LLMTask]] = []

    def execute_batch(self, tasks: List[LLMTask]) -> List[LLMResult]:
        if self.info.max_batch is not None:
            assert len(tasks) <= self.info.max_batch, "batch larger than max_batch"
        self.batches.append(list(tasks))
        results = []
        for task in tasks:
            try:
                results.append(LLMResult(task.id, self.answer(task)))
            except Exception as e:
                results.append(LLMResult(task.id, None, error=f"{type(e).__name__}: {e}"))
        return results
