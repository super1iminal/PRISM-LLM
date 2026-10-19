"""One live OpenRouter request through the repo's own backend, with the planner's rule schema.

Checks that the key works from this node, that the model and provider routing accept the request, and that the
answer parses as the planner would parse it. Writes everything (request settings, reply, tokens, timing, the
key's limits) to a JSON file for later analysis; the key itself is never written.

Usage (from anywhere, after `source cluster/env.sh`):
    python cluster/smoke/llm_ping.py --out out/cluster/smoke/ping.json                     # the free smoke model
    python cluster/smoke/llm_ping.py --model anthropic/claude-haiku-5.5 --providers anthropic --structured \
        --think --out ...                                                                  # the real runs' route
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

GL = Path(__file__).resolve().parents[2] / "PRISM-Guided-Learning"
sys.path.insert(0, str(GL / "src"))

from config import load_config                  # noqa: E402
from core.backends import make_backend          # noqa: E402
from core.planner import rule_schema            # noqa: E402
from core.tasks import TaskFactory              # noqa: E402

PROMPT = ('A robot on a grid has integer variables x and y and must reach x = 3. Give at most 2 rules '
          '(condition -> action) as JSON with conditions over x and y, e.g. "x < 3".')


def key_info(env: str) -> dict:
    """OpenRouter's GET /key: usage, limit and free-tier status of the key (costs nothing)."""
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": f"Bearer {os.environ[env]}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r).get("data", {})
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {k: data.get(k) for k in ("usage", "limit", "limit_remaining", "is_free_tier", "rate_limit")}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=os.environ.get("SMOKE_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"))
    p.add_argument("--providers", default="", help="comma-separated provider slugs ('' = OpenRouter routes)")
    p.add_argument("--structured", action="store_true", help="send the schema as response_format")
    p.add_argument("--think", action="store_true", help="reasoning on")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--set", action="append", default=[], help="further run-config overrides (section.key=value)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    providers = [x for x in a.providers.split(",") if x]
    cfg = load_config(overrides=["llm.backend=openrouter", f"llm.model={a.model}", "llm.seed=1",
                                 f"llm.think={str(a.think).lower()}", f"llm.num_predict={a.max_tokens}",
                                 f"llm.openrouter.providers={providers}",
                                 f"llm.openrouter.structured={str(a.structured).lower()}", *a.set]).llm
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": os.uname().nodename,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "model": a.model, "providers": providers,
              "structured": a.structured, "think": a.think, "key": key_info(cfg.openrouter.api_key_env)}
    schema = rule_schema(["up", "down", "left", "right"], max_rules=4, max_condition_chars=40)
    backend = make_backend(cfg)
    try:
        result = backend.execute(TaskFactory(cfg, "ping").make(PROMPT, schema.model_json_schema()))
    finally:
        backend.close()
    record.update(error=result.error, seconds=result.seconds, prompt_tokens=result.prompt_tokens,
                  output_tokens=result.output_tokens, text=result.text)
    try:
        parsed = schema.model_validate_json(result.text or "")
        record.update(parsed=True, rules=[[r.condition, r.action] for r in parsed.rules])
    except Exception as e:
        record.update(parsed=False, parse_error=f"{type(e).__name__}: {e}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("model", "error", "parsed", "seconds", "output_tokens", "key")},
                     indent=2))
    sys.exit(0 if record["error"] is None and record["parsed"] else 1)


if __name__ == "__main__":
    main()
