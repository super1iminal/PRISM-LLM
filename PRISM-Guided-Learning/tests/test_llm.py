"""core/llm.py: the legacy planner's blocking client, on top of a backend."""
import threading

import pytest
from pydantic import BaseModel

from config import load_config
from core.llm import LLMClient
from core.tasks import LLMError
from fakes import FakeBackend

SEED = 4 * 1_000_003


class Answer(BaseModel):
    value: int


def test_client_parses_records_usage_and_numbers_seeds_per_usage():
    backend = FakeBackend(lambda task: '{"value": 3}')
    client = LLMClient(load_config(overrides=["llm.seed=4"]).llm, Answer, backend)
    assert client.invoke("a").value == 3 and client.invoke("b").value == 3
    assert [c.prompt for c in client.usage().calls] == ["a", "b"] and client.usage().output_tokens == 10
    client.reset_usage()
    client.invoke("c")
    assert [t.params["seed"] for b in backend.batches for t in b] == [SEED, SEED + 1, SEED]
    assert backend.batches[0][0].schema == Answer.model_json_schema()


def test_each_thread_has_its_own_usage():
    client = LLMClient(load_config().llm, Answer, FakeBackend(lambda task: '{"value": 1}'))
    counts = {}

    def work(name, n):
        client.reset_usage()
        for _ in range(n):
            client.invoke(name)
        counts[name] = [c.prompt for c in client.usage().calls]

    threads = [threading.Thread(target=work, args=(f"t{i}", i + 1)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert counts == {"t0": ["t0"], "t1": ["t1"] * 2, "t2": ["t2"] * 3}


def test_backend_failures_raise_and_are_not_recorded():
    def fail(task):
        raise ConnectionError("server gone")
    client = LLMClient(load_config().llm, Answer, FakeBackend(fail))
    with pytest.raises(LLMError, match="ConnectionError: server gone"):
        client.invoke("a")
    assert client.usage().calls == []
