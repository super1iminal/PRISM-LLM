"""Haiku through OpenRouter (provider anthropic only), with cost accounting."""
import json
import os
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

import openai

MODEL = "anthropic/claude-haiku-5.5"
_lock = threading.Lock()
USAGE = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "cost": 0.0}


def _client():
    return openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                         timeout=300, max_retries=3)


def chat(messages: List[Dict], max_tokens: int = 20000, temperature: float = 1.0,
         reasoning_tokens: int = 8000) -> Tuple[str, Dict]:
    client = _client()
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature,
                extra_body={"provider": {"order": ["anthropic"], "allow_fallbacks": False},
                            "reasoning": {"max_tokens": reasoning_tokens} if reasoning_tokens else {"enabled": False}})
            text = r.choices[0].message.content or ""
            u = r.usage
            cost = float(getattr(u, "cost", 0.0) or 0.0)
            usage = {"prompt_tokens": u.prompt_tokens or 0, "output_tokens": u.completion_tokens or 0, "cost": cost}
            with _lock:
                USAGE["calls"] += 1
                for k in ("prompt_tokens", "output_tokens", "cost"):
                    USAGE[k] += usage[k]
            if not text.strip() and attempt < 3:
                continue   # the budget went to reasoning; ask again (its cost is counted above)
            return text, usage
        except Exception as e:   # transient API errors
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def parse_rules(text: str) -> Optional[List[Tuple[str, str]]]:
    """The last JSON object with a "rules" list in `text` (fenced or bare)."""
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if not candidates:
        start = text.rfind('{"rules"')
        if start < 0:
            start = text.find("{")
        if start >= 0:
            candidates = [text[start:]]
    for c in reversed(candidates):
        for end in range(len(c), 0, -1):
            if c[end - 1] != "}":
                continue
            try:
                obj = json.loads(c[:end])
            except json.JSONDecodeError:
                continue
            rules = obj.get("rules") if isinstance(obj, dict) else None
            if isinstance(rules, list):
                out = []
                for r in rules:
                    if isinstance(r, dict) and "condition" in r and "action" in r:
                        out.append((str(r["condition"]), str(r["action"])))
                return out
            break
    return None
