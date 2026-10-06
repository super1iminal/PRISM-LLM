"""The interface every LLM backend implements: run a batch of tasks, return their results in order."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from core.tasks import LLMResult, LLMTask


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

    def close(self) -> None:
        """Release clients, servers or GPU memory held by the backend."""
