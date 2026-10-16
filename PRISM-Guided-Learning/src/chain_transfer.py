"""Certify every chain's kept rules of a `planner.chains` run (not only the chosen chain's) on held-out targets, to
read the choice among chains against a single chain within the same runs.

Each chain's kept rules are those of its last improved round (core/planner.py keeps the best so far); its margin is
the run's own (`chains` in the sample record, exact values).

Usage (from PRISM-Guided-Learning/):
    python src/chain_transfer.py <run dir> [--data <dataset>] [--targets 16-35] [--workers N]
Writes <run dir>/transfer_<dataset stem>_chains.csv: the transfer CSV columns plus `chain`, `margin` and `chosen`.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import load_file, load_run_config
from core.domain import load_domain
from core.prism import PrismRunner
from results_io import sample_records
from transfer import DEFAULT_CONFIG, TransferConfig, certify, guard_sealed, parse_ids, training_instances


def chain_rules(record: dict) -> Dict[int, List[Tuple[str, str]]]:
    """Chain index -> its kept rules (the rules of its last improved round)."""
    kept = {}
    for it in record["iterations"]:
        if it.get("improved") or it["chain"] not in kept:
            kept[it["chain"]] = [(r["condition"], r["action"]) for r in it["rules"]]
    return kept


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data", default=None, help="Target dataset (default: the run's own)")
    parser.add_argument("--targets", default=None, help='Target ids, e.g. "12-19" (default: all)')
    parser.add_argument("--workers", type=int, default=2, help="Targets checked at once")
    args = parser.parse_args(argv)
    cfg = load_run_config(args.run_dir)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    dataset = args.data or cfg.domain.dataset
    targets = domain.load_instances(dataset)
    if args.targets:
        wanted = set(parse_ids(args.targets))
        targets = [t for t in targets if int(t.id) in wanted]
    guard_sealed(dataset, [int(t.id) for t in targets])
    records = sample_records(args.run_dir)
    sources, meta = {}, {}
    for sid, record in records.items():
        if "chains" not in record:
            continue
        margins = {c["chain"]: c["margin"] for c in record["chains"]}
        for chain, rules in chain_rules(record).items():
            sources[f"{sid}:{chain}"] = rules
            meta[f"{sid}:{chain}"] = (sid, chain, margins.get(chain), chain == record["chain"])
    runner = PrismRunner(cfg.prism)
    exact_near = load_file(DEFAULT_CONFIG, TransferConfig).exact_near
    with ThreadPoolExecutor(args.workers) as pool:
        rows = [row for part in pool.map(lambda t: certify(domain, cfg, runner, t, sources, exact_near), targets)
                for row in part]
    df = pd.DataFrame(rows)
    df[["source", "chain", "margin", "chosen"]] = [meta[s] for s in df.source]
    trained = training_instances(args.run_dir) if Path(dataset).name == Path(cfg.domain.dataset).name else {}
    df["trained"] = [t in trained.get(s, []) for s, t in zip(df.source, df.target)]
    out = args.run_dir / f"transfer_{Path(dataset).stem}_chains.csv"
    df.sort_values(["source", "chain", "target"]).to_csv(out, index=False)
    held = df[~df.trained].assign(certified=df.certified.fillna(False).astype(bool))
    print(held.groupby(["source", "chain"]).agg(certified=("certified", "sum"), margin=("margin", "first"),
                                                chosen=("chosen", "first")).to_string())
    print(out)


if __name__ == "__main__":
    main()
