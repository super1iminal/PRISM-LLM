"""Budget and chain-count curves of finished multi-chain runs (no LLM calls), certified by the paper's rule.

Two replays of a run's own pick with repair, each list with `any` resolved by the
pick's order and certified on held-out targets by src/recertify.py (interval iteration on every check; inconclusive
within the band):
  * rounds: every chain cut to its first R rounds, R = 1 .. the run's rounds (as budget_transfer.chain_picks);
  * chains: the first k chains kept at full rounds, k = 1 .. K (the K-chains ablation).
Each row also carries the LLM calls, tokens and dollars of the rounds kept. R = the last round and k = K are the run's
own final list. Writes <run dir>/budget_ii_rounds.csv and budget_ii_chains.csv (one row per list and target).

Usage (from artifact/):
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
from recertify import certify_rules, own_targets, with_prism
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


def table(run_dir: Path, axis: str, targets: str, price, workers: int, source_targets=None, unseal: bool = False,
          name: str = "budget_ii", prism=()) -> pd.DataFrame:
    """One axis's replay of `run_dir`, certified on `targets`; written to <run dir>/<name>_<axis>.csv."""
    cfg = with_prism(load_run_config(run_dir), list(prism))
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(t.id): t for t in domain.load_instances(cfg.domain.dataset)}
    wanted = [i for i in parse_ids(targets) if i in instances]
    guard_sealed(cfg.domain.dataset, wanted, unseal)
    rows_of = picks(run_dir, axis)
    lists: Dict[str, list] = {rules_hash(p["rules"]): p["rules"] for p in rows_of}
    own = own_targets(source_targets)       # by training set: a hash is checked where any of its sets' targets are
    sets_of: Dict[str, set] = {}
    for p in rows_of:
        sets_of.setdefault(rules_hash(p["rules"]), set()).add(str(p["source"]))

    def todo(t):
        return {h: r for h, r in lists.items() if any(own({s: r}, instances[t]) for s in sets_of[h])}
    with ThreadPoolExecutor(workers) as pool:
        checks = [r for part in pool.map(lambda t: certify_rules(domain, cfg, instances[t], todo(t)), wanted)
                  for r in part]
    by = {(c["source"], int(c["target"])): c for c in checks}
    trained = training_instances(run_dir)
    rows = []
    for p in rows_of:
        h = rules_hash(p["rules"])
        dollars = (p["prompt_tokens"] * price[0] + p["output_tokens"] * price[1]) / 1e6 if price else float("nan")
        for t in wanted:
            if (h, t) not in by or not own({str(p["source"]): None}, instances[t]):
                continue
            c = {k: v for k, v in by[(h, t)].items() if k not in ("source", "target")}
            rows.append({**{k: v for k, v in p.items() if k != "rules"}, "rules_hash": h, "num_rules": len(p["rules"]),
                         "dollars": dollars, "target": t, "trained": t in trained.get(p["source"], []), **c})
    df = pd.DataFrame(rows)
    out = run_dir / f"{name}_{axis}.csv"
    df.to_csv(out, index=False)
    print(f"{out}: {len(lists)} lists, {len(df)} rows, {df.outcome.value_counts().to_dict()}",
          flush=True)
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--axis", nargs="+", choices=list(OUT), default=list(OUT))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--price", type=float, nargs=2, default=None, metavar=("IN", "OUT"))
    parser.add_argument("--source-targets", default=None, help='JSON, training set -> its own target ids')
    parser.add_argument("--unseal", action="store_true", help="Allow sealed test targets (the test pass only)")
    parser.add_argument("--name", default="budget_ii", help="Output name: <run dir>/<name>_<axis>.csv "
                        "(budget_ii_sealed for the test pass)")
    parser.add_argument("--prism", action="append", default=[], help="PRISM setting override key=value")
    args = parser.parse_args(argv)
    for axis in args.axis:
        table(args.run_dir, axis, args.targets, args.price, args.workers, args.source_targets, args.unseal,
              args.name, args.prism)


if __name__ == "__main__":
    main()
