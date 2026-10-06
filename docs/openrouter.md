# Running on OpenRouter

`llm.backend: openrouter` (`src/core/backends/openrouter.py`) sends the planner's LLM tasks to [OpenRouter](https://openrouter.ai), a hosted, OpenAI-compatible API in front of many providers, so runs need no local GPU. PRISM and the rest of the loop still run locally. Both schedulers work: with `threads` each worker sends its own requests, and with `lockstep` each batch goes out as concurrent requests.

How it differs from the local Ollama runs:

- **Model id.** `llm.model` is an OpenRouter id, e.g. `qwen/qwen3-14b`, not the Ollama tag.
- **Weights.** Providers serve their own weights (FP8 or BF16), not Ollama's `q4_K_M` GGUF, so OpenRouter runs are a condition of their own, not a re-run of the local ones.
- **Providers.** `llm.openrouter.providers` pins the providers to use, in order, with no fallback. Left empty, OpenRouter may route each request to a different provider. Every request asks only for providers that honour all its parameters (`require_parameters`: the JSON schema and the seed).
- **Thinking.** `llm.think` is sent as OpenRouter's `reasoning.enabled`. The smoke test checks that this really turns Qwen3's thinking off.
- **Context.** `llm.num_ctx` is not sent; the provider sets the context window (Qwen3-14B: about 40k tokens).
- **Seeds.** Seeds are sent, but hosted providers do not promise identical samples for the same seed.

## Setup

1. `pip install -r requirements.txt` (adds `openai`, the client the backend uses).
2. Put the API key in the environment, never in a config file. `config.json` only records the variable's name.
   - PowerShell: `$env:OPENROUTER_API_KEY = "sk-or-..."`
   - bash: `export OPENROUTER_API_KEY=sk-or-...`
3. Choose a provider. On the model's page (`https://openrouter.ai/qwen/qwen3-14b`, *Providers*), pick one that supports structured outputs and `seed`, and note its slug (e.g. `deepinfra`). Its quantization is listed there too.

## Smoke test

This checks that the backend works end to end against the real API. It costs well under a dollar; a 20-instance run is about $0.10 at current Qwen3-14B prices. Run from `PRISM-Guided-Learning/` with the repo's venv. On Windows that is `../.venv/Scripts/python`; `python` below means that interpreter. Set `<slug>` to the provider from Setup step 3.

**1. Offline tests** (no key or network needed):

```bash
python -m pytest -q tests/test_openrouter.py tests/test_config.py tests/test_llm.py
```

**2. One live request.** This sends one small prompt with the planner's rule schema and checks three things: no error, the answer parses with the schema, and fewer than 400 output tokens (a reply with thinking left on is far longer).

```powershell
$env:OPENROUTER_LIVE = "1"; $env:OPENROUTER_PROVIDERS = "<slug>"   # optional: $env:OPENROUTER_MODEL
python -m pytest -q tests/test_openrouter.py -k live -rs
```
(bash: `OPENROUTER_LIVE=1 OPENROUTER_PROVIDERS=<slug> python -m pytest -q tests/test_openrouter.py -k live -rs`)

**3. One instance, threads scheduler:**

```bash
python src/run_symbolic.py --set llm.backend=openrouter --set llm.model=qwen/qwen3-14b --set "llm.openrouter.providers=[<slug>]" --set llm.seed=1 --limit 1 --out out/results/smoke/openrouter_threads
```

**4. Two instances, lockstep** (one batch of two concurrent requests per step, logged to `llm_tasks.jsonl`):

```bash
python src/run_symbolic.py --set llm.backend=openrouter --set llm.model=qwen/qwen3-14b --set "llm.openrouter.providers=[<slug>]" --set llm.seed=1 --set run.scheduler=lockstep --limit 2 --workers 2 --out out/results/smoke/openrouter_lockstep
```

**What to check after steps 3 and 4.** For comparison, the local runs average about 1.4k output tokens per call (max about 5k), and the first round's prompt is about 1.6k tokens.

- `main.log`: every instance finishes with no `error`.
- `outputs/sample_*.json`: in each round, `llm_calls` and `invalid_answers`. A few invalid answers are normal; every call failing to parse is not. `llm_output_tokens` and `llm_prompt_tokens` should be close to the local runs; far more output tokens suggests thinking is on.
- `outputs/sample_*.json`: `llm_time` per round, to compare with PRISM's `prism_time`.
- Lockstep only: `llm_tasks.jsonl` has one line per task, and no result carries an `error`.
- OpenRouter's *Activity* page: every request was served by the pinned provider, the reasoning tokens are 0, and the cost is as expected.

**Afterwards.** Add a short entry under "LLM tasks, backends and lockstep batching" in `DECISIONS.md`: model id, provider and its quantization, whether thinking was off, tokens and time per round against the local runs, cost, and any errors. The runs go to `out/results/smoke/openrouter_*`, which is gitignored (like the Ollama smoke runs in `out/results/smoke/`), so they are never committed; delete them when done.

### If something fails

| Symptom | Likely cause and what to do |
|---|---|
| `RuntimeError: ... OPENROUTER_API_KEY` | The key is not in this shell's environment (Setup step 2). |
| `AuthenticationError` (401) | The key is wrong or revoked. |
| `NotFoundError` (404) "No endpoints found ..." | No pinned provider supports every parameter sent (JSON schema, seed, reasoning) or the requested quantization. Pick another provider, or try `providers=[]` to see which provider OpenRouter chooses. |
| `BadRequestError` (400) about the schema or `strict` | The provider rejects the strict JSON schema (it uses `$defs`, `maxLength`, `maxItems`). Try another provider. If none accepts it, report it: the fix is a code change in `openrouter.py`. |
| `RateLimitError` (429) after the client's retries | Lower `--workers`, or raise `llm.openrouter.max_retries`. |
| Hundreds or thousands of output tokens on the live test | `reasoning.enabled: false` did not turn thinking off for this provider. Report it with the provider's name; do not run experiments until it is fixed. |

## For a Claude Code session running the smoke test

Follow the steps above in order and stop at the first failure. Each step after 1 spends money, so confirm with the user before step 2. Never print, log or commit the API key. Report the checks above as a short table (step, pass/fail, tokens, time, provider) and propose the `DECISIONS.md` entry; do not commit the run directories.
