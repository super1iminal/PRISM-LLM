"""Freeze each instance's (or training set's) final rules from a run and certify them, unchanged, on every
instance of a dataset.

With the general rule vocabulary (`rules.general`), conditions only name instance constants and domain
features, so a rule set written for one instance means something on every other. This checks whether it
still meets every requirement there, in the worst case, by exact model checking per target instance.
Rules that do not parse on a target (e.g. a run without the general vocabulary) are reported as such. Column
`trained` marks the targets a rule set was written for (its own instance, or its training set's members) when
the targets are the run's own dataset.

Usage (from PRISM-Guided-Learning/):
    python src/transfer.py out/results/ablations/<condition>/<seed dir> [--data <dataset>] [--workers 4]
Writes <run dir>/transfer_<dataset stem>.csv (one row per source and target) and prints a summary.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import load_run_config
from core.domain import load_domain
from core.prism import PrismRunner
from core.rules import RuleError, SymbolicPolicy
from core.verifier import PolicyVerifier


def _records(run_dir: Path) -> Dict[int, dict]:
    return {int(path.stem.split("_")[1]): json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((run_dir / "outputs").glob("sample_*.json"))}


def training_instances(run_dir: Path) -> Dict[int, List[int]]:
    """Source sample id -> the ids of the instances its rules were written for: a training set's members, or
    the sample's own instance."""
    return {sid: [int(m["instance"]) for m in record["members"]] if "members" in record else [sid]
            for sid, record in _records(run_dir).items()}


def final_rules(run_dir: Path) -> Dict[int, List[Tuple[str, str]]]:
    """Source sample id -> its final rules, from the run's outputs/sample_*.json. Instances that ended with an
    error (no final rules) are skipped."""
    return {sid: [(r["condition"], r["action"]) for r in record["final_rules"]]
            for sid, record in _records(run_dir).items() if "final_rules" in record}


def certify(domain, cfg, runner, target, sources) -> List[dict]:
    """Every source rule set, checked on one target instance."""
    verifier = PolicyVerifier(domain, target, cfg, runner)
    spec, vocab = verifier.spec, verifier.spec.vocabulary(cfg.rules)
    rows = []
    for source, rules in sources.items():
        row = {"source": source, "target": int(target.id)}
        try:
            policy = SymbolicPolicy.from_raw(spec.variables, list(spec.actions), rules, vocab)
        except RuleError as e:
            rows.append({**row, "parsed": False, "error": str(e).splitlines()[0]})
            continue
        v = verifier.verify(policy, analysis=False)
        met = [r.name for r in spec.requirements if r.satisfied(v.worst[r.name])]
        rows.append({**row, "parsed": True, "met": len(met), "requirements": len(spec.requirements),
                     "certified": len(met) == len(spec.requirements), "uncovered": v.uncovered_situations,
                     **{f"worst_{r.name}": v.worst[r.name] for r in spec.requirements}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data", default=None, help="Target dataset (default: the run's own)")
    parser.add_argument("--workers", type=int, default=4, help="Target instances checked in parallel")
    args = parser.parse_args()

    cfg = load_run_config(args.run_dir)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    dataset = args.data or cfg.domain.dataset
    targets = domain.load_instances(dataset)
    sources = final_rules(args.run_dir)
    runner = PrismRunner(cfg.prism)
    with ThreadPoolExecutor(args.workers) as pool:
        rows = [row for part in pool.map(lambda t: certify(domain, cfg, runner, t, sources), targets) for row in part]

    df = pd.DataFrame(rows).sort_values(["source", "target"])
    trained = training_instances(args.run_dir) if Path(dataset).name == Path(cfg.domain.dataset).name else {}
    df["trained"] = [t in trained.get(s, []) for s, t in zip(df.source, df.target)]
    out = args.run_dir / f"transfer_{Path(dataset).stem}.csv"
    df.to_csv(out, index=False)
    certified = df.get("certified", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    per_source = df.assign(certified=certified).groupby("source").certified.sum()
    print(f"{len(sources)} rule sets x {len(targets)} targets ({dataset}); certified targets per source rule set:")
    print("  " + ", ".join(f"{s}: {int(n)}" for s, n in per_source.items()))
    covered = df[certified].target.nunique()
    held_out = df[certified & ~df.trained].groupby("source").size().reindex(per_source.index, fill_value=0)
    print(f"best single rule set: source {per_source.idxmax()} certifies {int(per_source.max())}/{len(targets)}; "
          f"targets certified by at least one rule set: {covered}/{len(targets)}")
    print("held-out targets certified per source rule set: " + ", ".join(f"{s}: {int(n)}" for s, n in held_out.items()))
    print(out)


if __name__ == "__main__":
    main()
