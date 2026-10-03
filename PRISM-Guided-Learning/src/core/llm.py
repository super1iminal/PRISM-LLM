"""Minimal structured-output LLM client (local Ollama)."""
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

import ollama
from pydantic import BaseModel

from config import LLMConfig

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMCall:
    prompt: str
    raw_output: str
    prompt_tokens: int
    output_tokens: int
    seconds: float          # client-side wall time, including any queueing at the server
    server_seconds: float   # Ollama's own total_duration


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

    `invoke(prompt)` returns an instance of `schema` (the default given at construction, or
    one passed per call). Every call is also recorded in
    a per-thread `LLMUsage`, so concurrent workers sharing one client can each read
    back their own raw outputs and token counts via `usage()` / `reset_usage()`.
    """

    def __init__(self, schema: Optional[Type[T]] = None, config: Optional[LLMConfig] = None):
        config = config or LLMConfig()
        self.schema = schema
        self.model = config.model
        self.think = config.think
        self.options = {"num_ctx": config.num_ctx, "num_predict": config.num_predict}
        if config.seed is not None:
            self.options["seed"] = config.seed
        if config.temperature is not None:
            self.options["temperature"] = config.temperature
        self._client = ollama.Client()
        self._local = threading.local()

    def usage(self) -> LLMUsage:
        if not hasattr(self._local, "usage"):
            self._local.usage = LLMUsage()
        return self._local.usage

    def reset_usage(self) -> None:
        self._local.usage = LLMUsage()

    def invoke_raw(self, prompt: str, schema: Optional[Type[T]] = None) -> str:
        schema = schema or self.schema
        start = time.time()
        response = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema(),
            think=self.think,
            options=self.options,
        )
        content = response.message.content or ""
        self.usage().calls.append(LLMCall(
            prompt=prompt,
            raw_output=content,
            prompt_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            seconds=time.time() - start,
            server_seconds=(response.total_duration or 0) / 1e9,
        ))
        return content

    def invoke(self, prompt: str, schema: Optional[Type[T]] = None) -> T:
        schema = schema or self.schema
        return schema.model_validate_json(self.invoke_raw(prompt, schema))
