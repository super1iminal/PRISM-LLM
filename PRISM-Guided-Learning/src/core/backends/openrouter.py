"""OpenRouter: a hosted, OpenAI-compatible chat API in front of many providers.

There is no endpoint that takes a list of chats, so a batch is sent as concurrent requests; the provider
runs them in parallel. `llm.model` is an OpenRouter model id (e.g. `qwen/qwen3-14b`), not an Ollama tag.
Providers serve their own weights (FP8 or BF16, not Ollama's q4_K_M), so runs are a condition of their
own. `llm.openrouter.providers` pins the providers to use, in order, with no fallback to others.
With `llm.openrouter.structured: false` (models without structured outputs) the schema goes at the end of the
prompt instead, and the reply's JSON object is cut out of any surrounding text or code fence.
`llm.openrouter.schema_adapter` rewrites the answer schema for providers whose strict JSON-schema mode needs another
form (`adapt_schema`; empty by default: the schema is sent as the planner builds it). Whatever the schema sent, the
planner validates every answer against its own model (`core.planner.rule_schema`), so the limits hold either way.
Each result keeps the response's metadata (`response_meta`): the generation id, the provider that served it, the
finish reason (`length`: cut off at `llm.num_predict`) and the usage details OpenRouter reports.
"""
import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from config import LLMConfig
from core.backends.base import BackendInfo, LLMBackend
from core.tasks import LLMResult, LLMTask


SCHEMA_IN_PROMPT = ("\n\nAnswer with one JSON object and nothing else (no code fence, no text around it). "
                    "It must follow this JSON schema:\n")


def json_object(text: str) -> str:
    """The JSON object in a reply: from its first `{` to its last `}` (the reply unchanged if it has none)."""
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text


def adapt_schema(schema: Dict[str, Any], adapter: List[str]) -> Dict[str, Any]:
    """`schema` rewritten by each step of `adapter`, in order (a copy; the planner's schema is never changed):
    - `no_additional_properties`: `additionalProperties: false` on every object (OpenAI's strict mode requires it);
    - `inline_refs`: every `$ref` to `#/$defs/<name>` replaced by that definition, and `$defs` dropped;
    - `drop:<keyword>`: that keyword removed everywhere (e.g. `drop:maxLength` for a provider that rejects it)."""
    out = copy.deepcopy(schema)
    defs = out.get("$defs", {})

    def walk(node: Any, step: str) -> Any:
        if isinstance(node, list):
            return [walk(x, step) for x in node]
        if not isinstance(node, dict):
            return node
        if step == "inline_refs" and "$ref" in node:
            return walk(copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[1]]), step)
        node = {k: walk(v, step) for k, v in node.items()
                if not (step.startswith("drop:") and k == step[len("drop:"):])}
        if step == "no_additional_properties" and node.get("type") == "object":
            node["additionalProperties"] = False
        return node

    for step in adapter:
        if step not in ("no_additional_properties", "inline_refs") and not step.startswith("drop:"):
            raise ValueError(f"unknown llm.openrouter.schema_adapter step: {step}")
        out = walk(out, step)
        if step == "inline_refs":
            out.pop("$defs", None)
        defs = out.get("$defs", {})
    return out


def response_meta(response: Any) -> Dict[str, Any]:
    """What a chat-completions response says about itself beyond the text: generation id, serving provider, finish
    reason and usage details (reasoning, cached tokens, cost), as far as the response carries them."""
    def plain(x: Any) -> Any:
        if x is None or isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [plain(v) for v in x]
        dump = getattr(x, "model_dump", None)      # the openai client's response objects (pydantic)
        if dump:
            return plain(dump())
        return plain(vars(x)) if hasattr(x, "__dict__") else str(x)

    choice = response.choices[0] if getattr(response, "choices", None) else None
    return {"id": getattr(response, "id", None), "provider": getattr(response, "provider", None),
            "model": getattr(response, "model", None),
            "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
            "native_finish_reason": getattr(choice, "native_finish_reason", None) if choice else None,
            "usage": plain(getattr(response, "usage", None))}


class OpenRouterBackend(LLMBackend):
    info = BackendInfo(name="openrouter")

    def __init__(self, config: LLMConfig, client: Any = None):
        """`client` replaces the `openai.OpenAI` client the backend would build (tests pass a fake)."""
        settings = config.openrouter
        if client is None:
            key = os.environ.get(settings.api_key_env)
            if not key:
                raise RuntimeError(f"llm.backend=openrouter needs an API key in the environment variable "
                                   f"{settings.api_key_env} (llm.openrouter.api_key_env)")
            import openai
            client = openai.OpenAI(base_url=settings.base_url, api_key=key, timeout=settings.timeout_s,
                                   max_retries=settings.max_retries)
        self._client = client
        provider: Dict[str, Any] = {"require_parameters": True}   # only providers honouring schema and seed
        if settings.providers:
            provider.update(order=list(settings.providers), allow_fallbacks=False)
        if settings.quantizations:
            provider["quantizations"] = list(settings.quantizations)
        self._provider = provider
        self._structured = settings.structured
        self._adapter = list(settings.schema_adapter)

    def execute_batch(self, tasks: List[LLMTask]) -> List[LLMResult]:
        if len(tasks) <= 1:
            return [self._run(t) for t in tasks]
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            return list(pool.map(self._run, tasks))

    def request(self, task: LLMTask) -> Dict[str, Any]:
        """The chat-completions arguments for `task`. `num_ctx` is not sent: the provider sets the context."""
        params = task.params
        messages = task.messages
        if task.schema is not None and not self._structured:
            *head, last = messages
            messages = [*head, {**last, "content": last["content"] + SCHEMA_IN_PROMPT + json.dumps(task.schema)}]
        kwargs: Dict[str, Any] = {"model": task.model, "messages": messages, "max_tokens": params["num_predict"]}
        for key in ("temperature", "seed"):
            if key in params:
                kwargs[key] = params[key]
        if task.schema is not None and self._structured:
            schema = adapt_schema(task.schema, self._adapter) if self._adapter else task.schema
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "answer", "strict": True, "schema": schema}}
        kwargs["extra_body"] = {"provider": self._provider, "reasoning": {"enabled": bool(params.get("think"))}}
        return kwargs

    def _run(self, task: LLMTask) -> LLMResult:
        start = time.time()
        try:
            response = self._client.chat.completions.create(**self.request(task))
            if not response.choices:
                raise RuntimeError("the response has no choices")
            text = response.choices[0].message.content or ""
            if task.schema is not None and not self._structured:
                text = json_object(text)
        except Exception as e:
            return LLMResult(task.id, None, error=f"{type(e).__name__}: {e}", seconds=time.time() - start)
        usage = response.usage
        return LLMResult(
            task_id=task.id,
            text=text,
            prompt_tokens=(usage.prompt_tokens or 0) if usage else 0,
            output_tokens=(usage.completion_tokens or 0) if usage else 0,
            seconds=time.time() - start,
            meta=response_meta(response),
        )

    def close(self) -> None:
        super().close()
        self._client.close()
