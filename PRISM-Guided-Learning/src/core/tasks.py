"""LLM tasks: what the GPU has to do for one answer, and what came back.

The symbolic planner never calls a model directly. It builds an `LLMTask` (model, messages, output
schema, generation parameters) and receives an `LLMResult`; a backend (core/backends) turns one into
the other. Tasks are self-contained and serializable, so they can be batched, logged and replayed.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from config import LLMConfig


class LLMError(RuntimeError):
    """A backend failed to answer a task. The message is the backend's `LLMResult.error`."""


@dataclass(frozen=True)
class LLMTask:
    id: str                                    # "<instance>/r<round>/f<fixup>"
    model: str
    messages: List[Dict[str, str]]             # chat messages; the planner sends one user message
    schema: Optional[Dict[str, Any]]           # JSON schema for constrained decoding
    params: Dict[str, Any]                     # think, num_ctx, num_predict, temperature, seed (resolved)
    meta: Dict[str, Any] = field(default_factory=dict)   # instance, round, mode, fixup (bookkeeping only)

    @property
    def prompt(self) -> str:
        return self.messages[-1]["content"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMResult:
    task_id: str
    text: Optional[str]                        # raw model output; None when the call failed
    error: Optional[str] = None                # "<ExceptionType>: <message>" when the call failed
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0                       # time spent on this task inside the backend
    server_seconds: float = 0.0                # the server's own processing time, if it reports one

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TaskFactory:
    """Builds the tasks of one instance.

    Each call's seed is `seed * 1_000_003 + n`, where n counts the calls made for this instance:
    reproducible, but distinct per call (one fixed seed would make every repeat of a prompt, e.g. a blind
    retry, return the identical answer).
    """

    def __init__(self, config: LLMConfig, instance_id: Any):
        self.config = config
        self.instance_id = instance_id
        self.calls = 0

    def make(self, prompt: str, schema: Optional[Dict[str, Any]], **meta: Any) -> LLMTask:
        cfg = self.config
        params: Dict[str, Any] = {"think": cfg.think, "num_ctx": cfg.num_ctx, "num_predict": cfg.num_predict}
        if cfg.temperature is not None:
            params["temperature"] = cfg.temperature
        if cfg.seed is not None:
            params["seed"] = cfg.seed * 1_000_003 + self.calls
        self.calls += 1
        task_id = f"{self.instance_id}/r{meta.get('round', 0)}/f{meta.get('fixup', 0)}"
        return LLMTask(id=task_id, model=cfg.model, messages=[{"role": "user", "content": prompt}],
                       schema=schema, params=params, meta={"instance": self.instance_id, **meta})
