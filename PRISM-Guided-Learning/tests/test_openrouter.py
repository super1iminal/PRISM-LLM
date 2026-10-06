"""core/backends/openrouter.py against a stand-in client; one opt-in test against the real API."""
import os
import shutil
import threading

import pytest

from config import load_config
from core.backends import make_backend
from core.backends.openrouter import OpenRouterBackend
from core.planner import rule_schema
from core.tasks import TaskFactory
from fakes import FakeChatClient, answer

SCHEMA = {"type": "object", "properties": {"rules": {"type": "array"}}, "required": ["rules"]}


def llm_config(*overrides):
    return load_config(overrides=["llm.backend=openrouter", "llm.model=qwen/qwen3-14b", *overrides]).llm


def make_tasks(n, *overrides):
    factory = TaskFactory(llm_config("llm.seed=3", *overrides), "g0")
    return [factory.make(f"prompt {i}", SCHEMA, round=i) for i in range(n)]


def test_a_task_becomes_a_chat_request():
    client = FakeChatClient()
    backend = OpenRouterBackend(llm_config("llm.openrouter.providers=[deepinfra]",
                                           "llm.openrouter.quantizations=[fp8]"), client=client)
    backend.execute_batch(make_tasks(1))
    request = client.requests[0]
    assert request["model"] == "qwen/qwen3-14b"
    assert request["messages"] == [{"role": "user", "content": "prompt 0"}]
    assert request["max_tokens"] == 8192 and request["seed"] == 3 * 1_000_003
    assert request["response_format"] == {"type": "json_schema",
                                          "json_schema": {"name": "answer", "strict": True, "schema": SCHEMA}}
    assert request["extra_body"] == {
        "provider": {"require_parameters": True, "order": ["deepinfra"], "allow_fallbacks": False,
                     "quantizations": ["fp8"]},
        "reasoning": {"enabled": False}}
    assert "num_ctx" not in request and "temperature" not in request and "think" not in request


def test_unset_options_are_left_to_openrouter():
    client = FakeChatClient()
    factory = TaskFactory(llm_config("llm.think=true", "llm.temperature=0.2"), "g0")
    OpenRouterBackend(llm_config(), client=client).execute(factory.make("p", None))
    request = client.requests[0]
    assert request["extra_body"] == {"provider": {"require_parameters": True}, "reasoning": {"enabled": True}}
    assert request["temperature"] == 0.2
    assert "seed" not in request and "response_format" not in request


def test_results_keep_task_order_and_report_tokens():
    client = FakeChatClient(reply=lambda kwargs: kwargs["messages"][0]["content"].upper(), delay=0.05)
    tasks = make_tasks(10)
    results = OpenRouterBackend(llm_config(), client=client).execute_batch(tasks)
    assert [r.task_id for r in results] == [t.id for t in tasks]
    assert [r.text for r in results] == [f"PROMPT {i}" for i in range(10)]
    assert all(r.error is None and r.prompt_tokens == 11 and r.output_tokens == 7 and r.seconds >= 0
               for r in results)


def test_a_failed_request_only_fails_its_task():
    def reply(kwargs):
        if kwargs["messages"][0]["content"] == "prompt 2":
            raise TimeoutError("upstream timeout")
        return "ok"
    results = OpenRouterBackend(llm_config(), client=FakeChatClient(reply)).execute_batch(make_tasks(4))
    assert results[2].text is None and results[2].error == "TimeoutError: upstream timeout"
    assert [r.text for i, r in enumerate(results) if i != 2] == ["ok"] * 3


def test_a_response_without_choices_is_an_error():
    client = FakeChatClient()
    client.chat.completions.create = lambda **kwargs: type("R", (), {"choices": [], "usage": None})()
    result = OpenRouterBackend(llm_config(), client=client).execute(make_tasks(1)[0])
    assert result.text is None and result.error == "RuntimeError: the response has no choices"


def test_concurrent_calls_from_threads_get_their_own_answers():
    """The threads scheduler calls one shared backend from every worker."""
    backend = OpenRouterBackend(llm_config(), client=FakeChatClient(lambda kw: kw["messages"][0]["content"], 0.02))
    tasks, out = make_tasks(20), {}
    threads = [threading.Thread(target=lambda t=t: out.__setitem__(t.id, backend.execute(t))) for t in tasks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(out[t.id].text == t.prompt for t in tasks)


def test_close_closes_the_client():
    client = FakeChatClient()
    OpenRouterBackend(llm_config(), client=client).close()
    assert client.closed


def test_make_backend_needs_the_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_backend(llm_config())
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")   # no request is made until a task runs
    backend = make_backend(llm_config())
    assert isinstance(backend, OpenRouterBackend)
    backend.close()


@pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")
@pytest.mark.parametrize("scheduler", ["threads", "lockstep"])
def test_run_symbolic_on_openrouter(tmp_path, tiny_grid, monkeypatch, scheduler):
    """The whole path from the config to the backend, with only the HTTP client replaced."""
    import openai
    import pandas as pd
    import run_symbolic
    from results_io import SYMBOLIC_RESULTS

    client = FakeChatClient(lambda kwargs: answer(("true", "right")))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    cfg = load_config(overrides=[f"domain.dataset={tiny_grid.as_posix()}", "llm.backend=openrouter",
                                 "llm.model=qwen/qwen3-14b", "planner.max_rounds=2", f"run.scheduler={scheduler}"])
    run_symbolic.run(cfg, str(tmp_path / "run"))
    df = pd.read_parquet(tmp_path / "run" / SYMBOLIC_RESULTS).reset_index()
    assert df.success.tolist() == [True] and df.llm_calls.tolist() == [1]
    assert client.requests[0]["model"] == "qwen/qwen3-14b" and client.closed


@pytest.mark.skipif(not (os.environ.get("OPENROUTER_API_KEY") and os.environ.get("OPENROUTER_LIVE") == "1"),
                    reason="live OpenRouter call: set OPENROUTER_API_KEY and OPENROUTER_LIVE=1 (costs a few cents)")
def test_live_request_honours_the_schema_without_thinking():
    """One real request (docs/openrouter.md): the answer parses with the planner's schema, and its token
    count is small enough that no thinking can have been generated. Set the model and providers with
    OPENROUTER_MODEL / OPENROUTER_PROVIDERS (comma-separated) if the defaults should not be used."""
    providers = [p for p in os.environ.get("OPENROUTER_PROVIDERS", "").split(",") if p]
    cfg = llm_config("llm.seed=1", f"llm.model={os.environ.get('OPENROUTER_MODEL', 'qwen/qwen3-14b')}",
                     f"llm.openrouter.providers={providers}")
    schema = rule_schema(["up", "down", "left", "right"], max_rules=4, max_condition_chars=40)
    prompt = ('A robot on a grid has integer variables x and y and must reach x = 3. Give at most 2 rules '
              '(condition -> action) as JSON with conditions over x and y, e.g. "x < 3".')
    backend = make_backend(cfg)
    try:
        result = backend.execute(TaskFactory(cfg, "live").make(prompt, schema.model_json_schema()))
    finally:
        backend.close()
    assert result.error is None, result.error
    parsed = schema.model_validate_json(result.text)
    assert 1 <= len(parsed.rules) <= 4
    assert 0 < result.output_tokens < 400, f"{result.output_tokens} output tokens: is thinking still on?"
