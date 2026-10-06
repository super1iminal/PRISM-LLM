# How the OpenRouter backend was tested

A record of the checks behind the commit that adds `llm.backend: openrouter` (`src/core/backends/openrouter.py`). The commit was made in a Linux container with no GPU, no Ollama and no OpenRouter key. Everything below ran offline; the live checks are left to the smoke test in `docs/openrouter.md`.

**Environment**
- Python 3.11, `pip install -r requirements.txt` plus `matplotlib`, which `tests/test_plot_configs.py` imports but `requirements.txt` does not list.
- `openai` 3.24.0.
- PRISM 4.10.1 from its GitHub release (`prism-4.10.1-linux64-x86`), on `PATH`, so the PRISM tests and the saved-run replays ran too.

## 1. Baseline before the change

`python -m pytest -q tests` on the merged branch (`64e0096`), in a separate worktree so the edits could not leak in: **150 passed** (9 min).

## 2. Unit tests against a stand-in client (`tests/test_openrouter.py`)

The backend takes an optional `client`. The tests pass `tests/fakes.py:FakeChatClient`, which records every `chat.completions.create` call, answers with a function of the request, and can wait a random time before answering.

| Test | What it pins down |
|---|---|
| `test_a_task_becomes_a_chat_request` | The request sent for a task: model id, messages, `max_tokens` from `num_predict`, the seed, the schema as a strict `json_schema` response format, `provider` (`require_parameters`, pinned `order`, `allow_fallbacks: false`, `quantizations`), `reasoning.enabled: false`. Also that `num_ctx` and `think` are not sent. |
| `test_unset_options_are_left_to_openrouter` | With no seed, no schema and no providers, none of them is sent. `think: true` becomes `reasoning.enabled: true`, and a set temperature is passed through. |
| `test_results_keep_task_order_and_report_tokens` | 10 tasks with random reply delays come back in task order, with their token counts. |
| `test_a_failed_request_only_fails_its_task` | One raising request gives that task `error`; the other three are answered and `execute_batch` does not raise. |
| `test_a_response_without_choices_is_an_error` | An empty response is a task error, not a crash. |
| `test_concurrent_calls_from_threads_get_their_own_answers` | 20 threads calling `execute` on one shared backend (as the threads scheduler does) each get their own answer. |
| `test_close_closes_the_client` | `close()` releases the client. |
| `test_make_backend_needs_the_api_key` | Without `OPENROUTER_API_KEY`, `make_backend` fails with a message naming the variable; with it, it builds the backend without making a request. |
| `test_run_symbolic_on_openrouter[threads/lockstep]` | `run_symbolic.run` on the tiny grid with `llm.backend=openrouter`, only `openai.OpenAI` replaced: config → `make_backend` → planner → real PRISM checks → results. The instance succeeds in one call under both schedulers, and the client is closed at the end. |
| `test_live_request_honours_the_schema_without_thinking` | Skipped here: it needs `OPENROUTER_API_KEY` and `OPENROUTER_LIVE=1` (see `docs/openrouter.md`). |

`tests/test_config.py:test_bad_config_is_rejected` gains three cases:
- `llm.openrouter.providers=deepinfra`, a string instead of a list, which would otherwise be split into characters;
- `quantizations=[8]`;
- `llm.openrouter.api_key=...`, an unknown key, so a key pasted into a config is an error.

**Mutation check.** Each defect below was put into `openrouter.py` by hand, the unit tests were run, and the file was restored. Every one made at least one test fail:

| Defect | Tests failing |
|---|---|
| results returned in the wrong order | 2 |
| `require_parameters` not sent | 2 |
| fallbacks left on when providers are pinned | 1 |
| thinking always on | 1 |
| a failed request raises out of the batch | 2 |
| seed not sent | 1 |
| `num_ctx` sent | 1 |

## 3. The real `openai` SDK against a local HTTP stub

The unit tests replace the client, so they cannot show what the SDK actually puts on the wire. A one-off script (not committed) therefore ran the real `openai.OpenAI` client, built by `make_backend`, against a local HTTP server that recorded each request and answered with a canned OpenRouter-style completion. It sent a batch of two tasks with the planner's real rule schema and `providers=[deepinfra]`.

- Both requests went to `POST /api/v1/chat/completions` with `Authorization: Bearer <key from OPENROUTER_API_KEY>`.
- The body carried `model`, `messages`, `max_tokens: 8192`, `seed`, and the full pydantic schema (with `$defs`, `maxLength` and `enum`) under `response_format.json_schema`.
- `provider` and `reasoning` arrived as **top-level** body fields, which is where OpenRouter reads them; the backend passes them through `extra_body`.
- Both results came back in order with the stub's text and token counts.

## 4. Full suite after the change

`python -m pytest -q tests`: **163 passed, 1 skipped** (9 min): the 150 earlier tests, 10 new ones in `tests/test_openrouter.py` and 3 new bad-config cases; the skip is the live test. This includes the D7, B2, R1 and S1 replays (every round byte-identical, one task at a time and in lockstep): the Ollama path and the prompts are unchanged.

## Not covered here

- **The real OpenRouter API.** Whether `reasoning.enabled: false` turns Qwen3's thinking off, whether the chosen provider accepts the strict schema, real latency, rate limits and cost. All of this is the smoke test in `docs/openrouter.md`, to be run by someone with a key.
- **Windows.** Everything ran on Linux; the smoke test runs on Windows.
- **Legacy runs on OpenRouter.** `run_legacy.py` now also replaces `/` in the model name, so an OpenRouter id does not nest the legacy log directory; legacy was not run on OpenRouter.
