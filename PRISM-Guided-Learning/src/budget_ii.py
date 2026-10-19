"""Budget and chain-count curves of finished multi-chain runs (no LLM calls), certified by the paper's rule.

Two replays of a run's own pick with repair (src/equal_spend.py's re-pick), each list with `any` resolved by the
pick's order and certified on held-out targets by src/recertify.py (interval iteration on every check; inconclusive
within the band):
  * rounds: every chain cut to its first R rounds, R = 1 .. the run's rounds (as budget_transfer.chain_picks);
  * chains: the first k chains kept at full rounds, k = 1 .. K (the K-chains ablation).
Each row also carries the LLM calls, tokens and dollars of the rounds kept. R = the last round and k = K are the run's
own final list. Writes <run dir>/budget_ii_rounds.csv and budget_ii_chains.csv (one row per list and target).

Usage (from PRISM-Guided-Learning/):
    python src/budget_ii.py <run dir> --targets 16-35,76-95,136-155 [--axis rounds chains] [--workers 2]
        [--price 0.10 0.50]
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import pandas as pd

from budget_transfer import rules_hash
from config import load_run_config
from core.domain import load_domain
from core.planner import SymbolicPlanner
from core.rules import fixed_order
from core.training import training_sets
from recertify import certify_rules
from transfer import guard_sealed, parse_ids, training_instances

OUT = {"rounds": "budget_ii_rounds.csv", "chains": "budget_ii_chains.csv"}


def picks(run_dir: Path, axis: str) -> List[dict]:
    """Per training set and budget (rounds R, or chains k): the re-pick over what is kept, its frozen list, and the
    calls, tokens and chains of the kept rounds."""
    cfg = load_run_config(run_dir)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    sets = training_sets(domain.load_instances(cfg.domain.dataset), cfg.run.train_sets, cfg.run.hidden_sets,
                         cfg.run.pool_sets)[:cfg.run.limit]
    planner = SymbolicPlanner(domain, None, cfg)
    planner.select_chain = lambda target, results: {"chain": None, "members": []}   # the pick replaces it
    out = []
    for sid, target in enumerate(sets):
        its = json.loads((run_dir / "outputs" / f"sample_{sid:03d}.json").read_text(encoding="utf-8"))["iterations"]
        top = max(it["chain_iteration"] for it in its) if axis == "rounds" else cfg.planner.chains
        for budget in range(1, top + 1):
            if axis == "rounds":
                kept = [it for it in its if it["chain_iteration"] <= budget]
                chains = range(cfg.planner.chains)
            else:
                kept = [it for it in its if it["chain"] < budget]
                chains = range(budget)
            results = [{"iterations": [dict(it, iteration=it["chain_iteration"]) for it in kept if it["chain"] == c]}
                       for c in chains]
            res = planner.pick_candidates(target, results, log=lambda line: None)
            if "pick" not in res:
                continue
            chosen = res["pick"]["candidates"][res["pick"]["chosen"]]
            rules, order = [tuple(r) for r in chosen["rules"]], chosen["resolved_order"]
            frozen = [tuple(r) for r in (fixed_order(rules, order) if order else rules)]
            out.append({"source": sid, axis[:-1]: budget, "rules": frozen,
                        "picked_from": f"{chosen['from'][0]}/{chosen['from'][1]}", "repaired": chosen["repaired"],
                        "llm_calls": sum(it.get("llm_calls") or 0 for it in kept),
                        "prompt_tokens": sum(it.get("llm_prompt_tokens") or 0 for it in kept),
                        "output_tokens": sum(it.get("llm_output_tokens") or 0 for it in kept),
                        "solved": bool(res.get("success"))})
            print(f"[{run_dir.parent.name}/{run_dir.name}] set {sid}, {axis} {budget}: {out[-1]['llm_calls']} calls, "
                  f"picked {out[-1]['picked_from']}, training {'passes' if out[-1]['solved'] else 'fails'}", flush=True)
    return out


def table(run_dir: Path, axis: str, targets: str, price, workers: int) -> pd.DataFrame:
    cfg = load_run_config(run_dir)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(t.id): t for t in domain.load_instances(cfg.domain.dataset)}
    wanted = [i for i in parse_ids(targets) if i in instances]
    guard_sealed(cfg.domain.dataset, wanted)
    rows_of = picks(run_dir, axis)
    lists: Dict[str, list] = {rules_hash(p["rules"]): p["rules"] for p in rows_of}
    with ThreadPoolExecutor(workers) as pool:
        checks = [r for part in pool.map(lambda t: certify_rules(domain, cfg, instances[t], lists), wanted)
                  for r in part]
    by = {(c["source"], int(c["target"])): c for c in checks}
    trained = training_instances(run_dir)
    rows = []
    for p in rows_of:
        h = rules_hash(p["rules"])
        dollars = (p["prompt_tokens"] * price[0] + p["output_tokens"] * price[1]) / 1e6 if price else float("nan")
        for t in wanted:
            c = {k: v for k, v in by[(h, t)].items() if k not in ("source", "target")}
            rows.append({**{k: v for k, v in p.items() if k != "rules"}, "rules_hash": h, "num_rules": len(p["rules"]),
                         "dollars": dollars, "target": t, "trained": t in trained.get(p["source"], []), **c})
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / OUT[axis], index=False)
    print(f"{run_dir / OUT[axis]}: {len(lists)} lists, {len(df)} rows, {df.outcome.value_counts().to_dict()}",
          flush=True)
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--axis", nargs="+", choices=list(OUT), default=list(OUT))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--price", type=float, nargs=2, default=None, metavar=("IN", "OUT"))
    args = parser.parse_args(argv)
    for axis in args.axis:
        table(args.run_dir, axis, args.targets, args.price, args.workers)


if __name__ == "__main__":
    main()
