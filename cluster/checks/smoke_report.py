"""Item 3 smoke test (Rajan's checklist): did structured output and re-asks behave on one training set?

Reads a finished smoke run (one training set of port_fam_history_<model>) and reports, from its llm_tasks.jsonl and
outputs/sample_*.json: calls, failed calls (with their errors), answers that did not parse (re-asks, fix-up > 0),
replies cut off at the output limit (finish_reason "length"), the providers that served the calls, reasoning tokens,
OpenRouter's reported cost, and whether the run produced final rules. Writes <run dir>/smoke_report.json.

    python cluster/checks/smoke_report.py <run dir>       # exit 0 = passed, 1 = failed (reasons printed)

Passed means: every call returned (no error), at least one answer parsed in every round, the run ended with final rules,
and every call was served by the pinned provider (reported as provider_matches_pin: OpenRouter enforces the pin with
no fallback, so a mismatch there is a naming difference to read, not a failure). Re-asks may occur; they must then
lead to a parsed answer.
"""
import json
import sys
from collections import Counter
from pathlib import Path


def main():
    run = Path(sys.argv[1])
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    pinned = cfg["llm"]["openrouter"]["providers"]
    lines = [json.loads(l) for l in (run / "llm_tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    errors = [l["result"]["error"] for l in lines if l["result"].get("error")]
    fixups = [l for l in lines if not l["task"]["id"].endswith("/f0")]
    metas = [l["result"].get("meta") or {} for l in lines]
    finish = Counter(m.get("finish_reason") for m in metas)
    providers = Counter(m.get("provider") for m in metas)
    reasoning = sum(((m.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                    for m in metas)
    cost = sum((m.get("usage") or {}).get("cost") or 0 for m in metas)
    samples = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run / "outputs").glob("sample_*.json"))]
    rounds = [it for s in samples for it in s.get("iterations", [])]
    unparsed_rounds = [it for it in rounds if not it.get("rules")]
    report = {
        "run": str(run), "model": cfg["llm"]["model"], "pinned_providers": pinned,
        "schema_adapter": cfg["llm"]["openrouter"].get("schema_adapter"), "think": cfg["llm"]["think"],
        "calls": len(lines), "failed_calls": len(errors), "errors": sorted(set(e[:300] for e in errors))[:10],
        "reasks": len(fixups), "invalid_answers": sum(len(it.get("invalid_answers") or []) for it in rounds),
        "rounds": len(rounds), "rounds_without_rules": len(unparsed_rounds),
        "finish_reasons": dict(finish), "providers": dict(providers), "reasoning_tokens": reasoning,
        "prompt_tokens": sum(l["result"].get("prompt_tokens") or 0 for l in lines),
        "output_tokens": sum(l["result"].get("output_tokens") or 0 for l in lines),
        "reported_cost_usd": cost, "final_rules": all("final_rules" in s for s in samples) and bool(samples),
    }
    reasons = []
    if errors:
        reasons.append(f"{len(errors)} failed calls")
    if unparsed_rounds:
        reasons.append(f"{len(unparsed_rounds)} rounds without a parsed answer")
    if not report["final_rules"]:
        reasons.append("no final rules")
    served = {p for p in providers if p}          # OpenRouter names the provider ("Google AI Studio"); pins are slugs
    report["provider_matches_pin"] = (not pinned or not served or
                                      {p.lower().replace(" ", "-") for p in served} <= {p.lower() for p in pinned})
    report["passed"], report["reasons"] = not reasons, reasons
    (run / "smoke_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    sys.exit(0 if not reasons else 1)


if __name__ == "__main__":
    main()
