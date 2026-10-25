"""A blocking, schema-constrained LLM client over a backend (core/backends), for the legacy planner.

The symbolic planner builds its tasks itself (core/tasks.py), so schedulers can batch them. The legacy
planner calls the model from its own worker threads, through this client.
"""
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from pydantic import BaseModel

from config import LLMConfig
from core.backends import LLMBackend, make_backend
from core.tasks import LLMError, TaskFactory

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMCall:
    prompt: str
    raw_output: str
    prompt_tokens: int
    output_tokens: int
    seconds: float          # client-side wall time, including any queueing at the server
    server_seconds: float   # the server's own processing time, if it reports one


@dataclass
class LLMUsage:
    calls: List[LLMCall] = field(default_factory=list)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)


class LLMClient:
    """Calls the model constrained to a pydantic schema and returns the parsed answer.

    Every call is recorded in a per-thread `LLMUsage`, so concurrent workers sharing one client can each
    read back their own raw outputs and token counts via `usage()` / `reset_usage()`. Call seeds are
    numbered per usage (`TaskFactory`): reproducible, but distinct per call.
    """

    def __init__(self, config: LLMConfig, schema: Type[T], backend: Optional[LLMBackend] = None):
        self.config = config
        self.schema = schema
        self.backend = backend or make_backend(config)
        self._local = threading.local()

    def usage(self) -> LLMUsage:
        if not hasattr(self._local, "usage"):
            self.reset_usage()
        return self._local.usage

    def reset_usage(self) -> None:
        self._local.usage = LLMUsage()
        self._local.tasks = TaskFactory(self.config, "legacy")

    def invoke_raw(self, prompt: str) -> str:
        usage = self.usage()
        result = self.backend.execute(self._local.tasks.make(prompt, self.schema.model_json_schema()))
        if result.error is not None:
            raise LLMError(result.error)
        usage.calls.append(LLMCall(prompt, result.text, result.prompt_tokens, result.output_tokens,
                                   result.seconds, result.server_seconds))
        return result.text

    def invoke(self, prompt: str) -> T:
        return self.schema.model_validate_json(self.invoke_raw(prompt))
