"""Test doubles: scripted LLM backends, and a tiny gridworld that PRISM solves in about a second."""
import json
from pathlib import Path
from typing import Callable, List, Optional

from core.backends import BackendInfo, LLMBackend
from core.tasks import LLMResult, LLMTask

# 4x4 grid, goal 1 top right, goal 2 bottom right, one static obstacle, an obstacle moving between
# (2,1) and (2,2). "true -> right" alone meets every threshold (slips walk the agent down column 3).
TINY_GRID = ('n,goals,static,moving,BFS_steps\n'
             '4,"{1: (0, 3), 2: (3, 3)}","[(1, 1)]","[(2, 1), (2, 2)]",6\n')


def write_tiny_grid(directory: Path) -> Path:
    path = Path(directory) / "tiny_grid.csv"
    path.write_text(TINY_GRID, encoding="utf-8")
    return path


def answer(*rules) -> str:
    """An LLM answer with the given (condition, action) rules."""
    return json.dumps({"rules": [{"condition": c, "action": a} for c, a in rules]})


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
                results.append(LLMResult(task.id, self.answer(task), prompt_tokens=10, output_tokens=5))
            except Exception as e:
                results.append(LLMResult(task.id, None, error=f"{type(e).__name__}: {e}"))
        return results

    @property
    def prompts(self) -> List[str]:
        """Every prompt sent, in order."""
        return [task.prompt for batch in self.batches for task in batch]


class ScriptedBackend(FakeBackend):
    """Answers with `answers` in order, whichever instance asks, then with empty rule lists."""

    def __init__(self, answers=()):
        queue = list(answers)
        super().__init__(lambda task: queue.pop(0) if queue else answer())
