"""Dump the exact OpenRouter request the backend builds for logged tasks, with the code of one source tree.

Rule 1(a) of the SEAMS checklist: the Haiku request path must stay byte-identical when the strict-schema adapter is
added. Run this once with Asher's tree and once with this one, on the same tasks and condition, and diff the outputs:

    python cluster/checks/request_dump.py --src <tree>/PRISM-Guided-Learning/src --condition port_fam_history \
        --tasks PRISM-Guided-Learning/out/results/core/port_fam_history/seed_none_1/llm_tasks.jsonl --n 3 --out a.json

Tasks are rebuilt from a run's llm_tasks.jsonl (messages, schema, parameters as logged); the backend is given a
stand-in client, so nothing is sent. The output is the request's JSON with sorted keys, one per task.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, required=True, help="the PRISM-Guided-Learning/src of the tree to use")
    p.add_argument("--condition", required=True)
    p.add_argument("--set", action="append", default=[], help="run-config overrides")
    p.add_argument("--tasks", type=Path, required=True, help="a run's llm_tasks.jsonl")
    p.add_argument("--n", type=int, default=3, help="how many tasks (the first n lines)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.src.resolve()))
    from config import load_config
    from core.backends.openrouter import OpenRouterBackend
    from core.tasks import LLMTask

    cfg = load_config(a.condition, a.set).llm
    backend = OpenRouterBackend(cfg, client=object())        # request() never touches the client
    requests = []
    for line in a.tasks.read_text(encoding="utf-8").splitlines()[:a.n]:
        t = json.loads(line)["task"]
        task = LLMTask(**{k: t[k] for k in LLMTask.__dataclass_fields__ if k in t})
        requests.append(backend.request(task))
    a.out.write_text(json.dumps(requests, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"{a.out}: {len(requests)} requests, model {cfg.model}")


if __name__ == "__main__":
    main()
