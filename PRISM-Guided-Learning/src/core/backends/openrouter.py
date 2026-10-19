"""OpenRouter: a hosted, OpenAI-compatible chat API in front of many providers.

There is no endpoint that takes a list of chats, so a batch is sent as concurrent requests; the provider
runs them in parallel. `llm.model` is an OpenRouter model id (e.g. `qwen/qwen3-14b`), not an Ollama tag.
Providers serve their own weights (FP8 or BF16, not Ollama's q4_K_M), so runs are a condition of their
own. `llm.openrouter.providers` pins the providers to use, in order, with no fallback to others.
With `llm.openrouter.structured: false` (models without structured outputs) the schema goes at the end of the
prompt instead, and the reply's JSON object is cut out of any surrounding text or code fence.
"""
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
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "answer", "strict": True, "schema": task.schema}}
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
        )

    def close(self) -> None:
        super().close()
        self._client.close()
