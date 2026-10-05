"""Ollama: a local server with no batch endpoint, so a batch is sent as concurrent requests.

The requests only run in parallel on the GPU if the server allows it (OLLAMA_NUM_PARALLEL at least
the batch size); otherwise Ollama queues them and the batch takes as long as running them in turn.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import ollama

from core.backends.base import BackendInfo, LLMBackend
from core.tasks import LLMResult, LLMTask


class OllamaBackend(LLMBackend):
    info = BackendInfo(name="ollama")

    def __init__(self):
        self._client = ollama.Client()

    def execute_batch(self, tasks: List[LLMTask]) -> List[LLMResult]:
        if len(tasks) <= 1:
            return [self._run(t) for t in tasks]
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            return list(pool.map(self._run, tasks))

    def _run(self, task: LLMTask) -> LLMResult:
        options = {k: v for k, v in task.params.items() if k != "think"}
        start = time.time()
        try:
            response = self._client.chat(model=task.model, messages=task.messages, format=task.schema,
                                         think=task.params.get("think"), options=options)
        except Exception as e:
            return LLMResult(task.id, None, error=f"{type(e).__name__}: {e}", seconds=time.time() - start)
        return LLMResult(
            task_id=task.id,
            text=response.message.content or "",
            prompt_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            seconds=time.time() - start,
            server_seconds=(response.total_duration or 0) / 1e9,
        )
