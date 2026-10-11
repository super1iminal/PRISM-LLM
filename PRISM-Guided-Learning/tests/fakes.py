"""Test doubles: scripted LLM backends, a stand-in for the OpenAI client, and a tiny gridworld that PRISM
solves in about a second."""
import json
import random
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

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
    Every executed batch is kept in `batches`, so tests can inspect what was sent together. With `delay`,
    each task first waits a random time of up to `delay` seconds, so concurrent answers arrive out of order."""

    def __init__(self, answer: Callable[[LLMTask], str], max_batch: Optional[int] = None, delay: float = 0.0):
        self.answer = answer
        self.info = BackendInfo(name="fake", max_batch=max_batch)
        self.batches: List[List[LLMTask]] = []
        self.delay = delay

    def execute_batch(self, tasks: List[LLMTask]) -> List[LLMResult]:
        if self.info.max_batch is not None:
            assert len(tasks) <= self.info.max_batch, "batch larger than max_batch"
        self.batches.append(list(tasks))
        results = []
        for task in tasks:
            if self.delay:
                time.sleep(random.uniform(0, self.delay))
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


class FakeChatClient:
    """Stands in for `openai.OpenAI`: records every `chat.completions.create` call and answers with
    `reply(kwargs)` (raising makes the request fail), after a random delay of up to `delay` seconds."""

    def __init__(self, reply: Callable[[Dict[str, Any]], str] = lambda kwargs: "ok", delay: float = 0.0):
        self.reply, self.delay = reply, delay
        self.requests: List[Dict[str, Any]] = []
        self.closed = False
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        with self._lock:
            self.requests.append(kwargs)
        time.sleep(random.uniform(0, self.delay))
        text = self.reply(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7))

    def close(self) -> None:
        self.closed = True
