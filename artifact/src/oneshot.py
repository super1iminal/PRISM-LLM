"""OneShot: the LLM's first answer, frozen and certified, with no selection or repair (no LLM calls).

Per training set of a finished `planner.chains` run: the first valid answer of the first chain (chain 0, round 1; the
round's rules are the first answer that parsed, after any re-asks), its `any` rules resolved as the pick resolves them
(`core.pick.resolved`: the best action order on the training members by the domain's fast checker), certified on
held-out targets by src/recertify.py's rule. Writes <run dir>/oneshot.json (per sample id: rules, order, where the
answer came from, its LLM calls) and <run dir>/<prefix>_<dataset stem>_oneshot.csv.

Usage (from artifact/):
    python src/oneshot.py <run dir> --targets 16-35,76-95,136-155 [--source-targets JSON] [--workers 2]
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import load_run_config
from core.domain import load_domain
from core.pick import evaluator, resolved
from core.rules import fixed_order
from core.training import training_sets
from recertify import certify_rules, own_targets, prefix_of, with_prism, write
from transfer import guard_sealed, parse_ids, training_instances


def first_answers(run_dir: Path) -> dict:
    """Per sample id: the first valid answer of chain 0, with `any` resolved by the pick's rule."""
    cfg = load_run_config(run_dir)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    sets = training_sets(domain.load_instances(cfg.domain.dataset), cfg.run.train_sets, cfg.run.hidden_sets,
                         cfg.run.pool_sets)[:cfg.run.limit]
    out = {}
    for sid, target in enumerate(sets):
        record = json.loads((run_dir / "outputs" / f"sample_{sid:03d}.json").read_text(encoding="utf-8"))
        rounds = sorted((it for it in record["iterations"] if it["chain"] == 0 and it.get("rules")),
                        key=lambda it: it["chain_iteration"])
        first = rounds[0]
        rules = [(r["condition"], r["action"]) for r in first["rules"]]
        members = list(target.members)
        specs = [domain.spec(i) for i in members]
        vocabs = [s.vocabulary(cfg.rules) for s in specs]
        checkers = [domain.fast_checker(i, cfg, v) for i, v in zip(members, vocabs)]
        if not all(c is not None for c in checkers):
            raise SystemExit(f"{cfg.domain.name} has no fast checker: the pick's resolution does not apply")
        _, order = resolved(rules, evaluator(checkers), [s.requirements for s in specs], list(specs[0].actions))
        frozen = fixed_order(rules, order) if order else rules
        out[str(sid)] = {"rules": [list(r) for r in frozen], "order": list(order) if order else None,
                         "from": [first["chain"], first["chain_iteration"]],
                         "invalid_answers": len(first.get("invalid_answers") or []),
                         "llm_calls": first.get("llm_calls"), "source": "oneshot: chain 0, first valid answer"}
        print(f"[{run_dir.parent.name}/{run_dir.name}] set {sid}: {len(frozen)} rules, order {order}", flush=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--data", default=None, help="Target dataset (default: the run's own)")
    parser.add_argument("--source-targets", default=None, help="JSON, sample id -> its own target ids")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefix", default=None, help="Output file prefix (transfer_sealed for the test pass)")
    parser.add_argument("--unseal", action="store_true", help="Allow sealed test targets (the test pass only)")
    parser.add_argument("--prism", action="append", default=[], help="PRISM setting override key=value")
    args = parser.parse_args(argv)
    lists = first_answers(args.run_dir)
    (args.run_dir / "oneshot.json").write_text(json.dumps(lists, indent=1), encoding="utf-8")
    cfg = with_prism(load_run_config(args.run_dir), args.prism)
    if cfg.rules.early_narrow:          # parsed as src/transfer.py parses the run's lists (places with numbers)
        cfg = replace(cfg, rules=replace(cfg.rules, shared_literals=True))
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    dataset = args.data or cfg.domain.dataset
    targets = [t for t in domain.load_instances(dataset) if int(t.id) in set(parse_ids(args.targets))]
    guard_sealed(dataset, [int(t.id) for t in targets], args.unseal)
    sources = {s: [tuple(r) for r in e["rules"]] for s, e in lists.items()}
    only = own_targets(args.source_targets)
    with ThreadPoolExecutor(args.workers) as pool:
        rows = [r for part in pool.map(lambda t: certify_rules(domain, cfg, t, only(sources, t)), targets) for r in part]
    df = pd.DataFrame(rows)
    trained = training_instances(args.run_dir) if Path(dataset).name == Path(cfg.domain.dataset).name else {}
    df["trained"] = [t in trained.get(int(s), []) for s, t in zip(df.source, df.target)]
    write(df, args.run_dir / f"{args.prefix or prefix_of(cfg)}_{Path(dataset).stem}_oneshot.csv")


if __name__ == "__main__":
    main()
