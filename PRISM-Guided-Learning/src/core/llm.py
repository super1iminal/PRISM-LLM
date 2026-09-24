"""Minimal structured-output LLM client (local Ollama)."""
import threading
from dataclasses import dataclass, field
from typing import List, Type, TypeVar

import ollama
from pydantic import BaseModel

from settings import OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT, OLLAMA_THINK

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMCall:
    prompt: str
    raw_output: str
    prompt_tokens: int
    output_tokens: int
    seconds: float


@dataclass
class LLMUsage:
    calls: List[LLMCall] = field(default_factory=list)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)


class OllamaLLM:
    """Calls an Ollama model constrained to a pydantic schema.

    `invoke(prompt)` returns an instance of `schema`. Every call is also recorded in
    a per-thread `LLMUsage`, so concurrent workers sharing one client can each read
    back their own raw outputs and token counts via `usage()` / `reset_usage()`.
    """

    def __init__(self, schema: Type[T], model: str = OLLAMA_MODEL, think: bool = OLLAMA_THINK,
                 num_ctx: int = OLLAMA_NUM_CTX, num_predict: int = OLLAMA_NUM_PREDICT):
        self.schema = schema
        self.model = model
        self.think = think
        self.options = {"num_ctx": num_ctx, "num_predict": num_predict}
        self._client = ollama.Client()
        self._local = threading.local()

    def usage(self) -> LLMUsage:
        if not hasattr(self._local, "usage"):
            self._local.usage = LLMUsage()
        return self._local.usage

    def reset_usage(self) -> None:
        self._local.usage = LLMUsage()

    def invoke_raw(self, prompt: str) -> str:
        response = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=self.schema.model_json_schema(),
            think=self.think,
            options=self.options,
        )
        content = response.message.content or ""
        self.usage().calls.append(LLMCall(
            prompt=prompt,
            raw_output=content,
            prompt_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            seconds=(response.total_duration or 0) / 1e9,
        ))
        return content

    def invoke(self, prompt: str) -> T:
        return self.schema.model_validate_json(self.invoke_raw(prompt))
