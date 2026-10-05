"""Test doubles: a scripted stand-in for OllamaLLM, and a tiny gridworld that PRISM solves in about a second."""
import json
from pathlib import Path

from core.llm import LLMCall, LLMUsage

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


class ScriptedLLM:
    """Stands in for OllamaLLM: returns `answers` in order (then empty rule lists) and records usage."""

    def __init__(self, answers=()):
        self.answers = list(answers)
        self.prompts = []
        self._usage = LLMUsage()

    def usage(self) -> LLMUsage:
        return self._usage

    def reset_usage(self) -> None:
        self._usage = LLMUsage()

    def invoke_raw(self, prompt: str, schema=None) -> str:
        raw = self.answers.pop(0) if self.answers else answer()
        self.prompts.append(prompt)
        self._usage.calls.append(LLMCall(prompt, raw, prompt_tokens=10, output_tokens=5, seconds=0.0,
                                         server_seconds=0.0))
        return raw
