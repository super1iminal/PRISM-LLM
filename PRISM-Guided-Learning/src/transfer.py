"""Freeze each instance's (or training set's) final rules from a run and certify them, unchanged, on every
instance of a dataset.

With the general rule vocabulary (`rules.general`), conditions only name instance constants and domain
features, so a rule set written for one instance means something on every other. This checks whether it
still meets every requirement there, in the worst case, by exact model checking per target instance.
Rules that do not parse on a target (e.g. a run without the general vocabulary) are reported as such. Column
`trained` marks the targets a rule set was written for (its own instance, or its training set's members) when
the targets are the run's own dataset.

Features built from definitions (`rules.vocabulary` definitions or llm) are frozen with the rules: each source
rule set's definitions (the domain's definitions.yaml, or the ones its run wrote) are evaluated again on every
target's own model.

Usage (from PRISM-Guided-Learning/):
    python src/transfer.py out/results/ablations/<condition>/<seed dir> [--data <dataset>] [--workers 4]
                           [--targets 12-19]
Writes <run dir>/transfer_<dataset stem>[_targets<ids>].csv (one row per source and target) and prints a summary.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import load_run_config
from core.definitions import Definition, DefinitionError, Evaluator, domain_definitions
from core.domain import load_domain
from core.prism import PrismRunner
from core.rules import RuleError, SymbolicPolicy, Vocabulary
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


def final_definitions(run_dir: Path) -> Dict[int, List[Definition]]:
    """Source sample id -> the definitions its rules may use (none for a run with hand-written features)."""
    return {sid: [Definition.from_dict(d) for d in record.get("definitions", [])]
            for sid, record in _records(run_dir).items()}


def parse_targets(text: Optional[str]) -> Optional[List[int]]:
    """'12-19' or '12,14,15' (or a mix) -> instance ids; None -> every instance."""
    if not text:
        return None
    ids = []
    for part in text.split(","):
        lo, _, hi = part.strip().partition("-")
        ids += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    return ids


def certify(domain, cfg, runner, target, sources, definitions=None) -> List[dict]:
    """Every source rule set, checked on one target instance. `definitions` maps a source to the definitions
    its features come from (rules.vocabulary other than domain)."""
    verifier = PolicyVerifier(domain, target, cfg, runner)
    spec, base = verifier.spec, verifier.spec.vocabulary(cfg.rules)
    evaluator = None
    if cfg.rules.extended and cfg.rules.vocabulary != "domain":
        evaluator = Evaluator(verifier, cfg.rules.definition_level, cfg.rules.general, cfg.rules.definition_penalty)
    rows = []
    for source, rules in sources.items():
        row = {"source": source, "target": int(target.id)}
        try:
            vocab = base
            if evaluator is not None:
                features = evaluator.features((definitions or {}).get(source, []))
                vocab = Vocabulary(base.constants, base.features + features, base.allow_any, base.general)
            policy = SymbolicPolicy.from_raw(spec.variables, list(spec.actions), rules, vocab)
        except (RuleError, DefinitionError) as e:
            rows.append({**row, "parsed": False, "error": str(e).splitlines()[0]})
            continue
        v = verifier.verify(policy, analysis=False)
        met = [r.name for r in spec.requirements if r.satisfied(v.worst[r.name])]
        rows.append({**row, "parsed": True, "met": len(met), "requirements": len(spec.requirements),
                     "certified": len(met) == len(spec.requirements), "uncovered": v.uncovered_situations,
                     **{f"worst_{r.name}": v.worst[r.name] for r in spec.requirements}})
    return rows


def transfer(run_dir: Path, dataset: Optional[str] = None, workers: int = 4,
             targets: Optional[List[int]] = None) -> Tuple[pd.DataFrame, Path]:
    """Certify every source rule set of `run_dir` on `dataset`'s instances (only `targets`, if given); writes
    and returns the table and its path."""
    run_dir = Path(run_dir)
    cfg = load_run_config(run_dir)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    dataset = dataset or cfg.domain.dataset
    instances = domain.load_instances(dataset)
    if targets is not None:
        instances = [i for i in instances if int(i.id) in set(targets)]
    sources = final_rules(run_dir)
    definitions = final_definitions(run_dir)
    if cfg.rules.extended and cfg.rules.vocabulary == "definitions":
        names = [v.name for v in PolicyVerifier(domain, instances[0], cfg).spec.variables]
        definitions = {sid: domain_definitions(domain, cfg.rules, names) for sid in sources}
    runner = PrismRunner(cfg.prism)
    with ThreadPoolExecutor(workers) as pool:
        rows = [row for part in pool.map(lambda t: certify(domain, cfg, runner, t, sources, definitions), instances)
                for row in part]

    df = pd.DataFrame(rows).sort_values(["source", "target"])
    trained = training_instances(run_dir) if Path(dataset).name == Path(cfg.domain.dataset).name else {}
    df["trained"] = [t in trained.get(s, []) for s, t in zip(df.source, df.target)]
    suffix = f"_targets{min(targets)}-{max(targets)}" if targets else ""
    out = run_dir / f"transfer_{Path(dataset).stem}{suffix}.csv"
    df.to_csv(out, index=False)
    return df, out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data", default=None, help="Target dataset (default: the run's own)")
    parser.add_argument("--workers", type=int, default=4, help="Target instances checked in parallel")
    parser.add_argument("--targets", default=None, help="Only these target instance ids, e.g. 12-19 (the fixed test set)")
    args = parser.parse_args()

    df, out = transfer(args.run_dir, args.data, args.workers, parse_targets(args.targets))
    sources, targets = df.source.nunique(), df.target.nunique()
    certified = df.get("certified", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    per_source = df.assign(certified=certified).groupby("source").certified.sum()
    print(f"{sources} rule sets x {targets} targets; certified targets per source rule set:")
    print("  " + ", ".join(f"{s}: {int(n)}" for s, n in per_source.items()))
    covered = df[certified].target.nunique()
    held_out = df[certified & ~df.trained].groupby("source").size().reindex(per_source.index, fill_value=0)
    print(f"best single rule set: source {per_source.idxmax()} certifies {int(per_source.max())}/{targets}; "
          f"targets certified by at least one rule set: {covered}/{targets}")
    print("held-out targets certified per source rule set: " + ", ".join(f"{s}: {int(n)}" for s, n in held_out.items()))
    print(out)


if __name__ == "__main__":
    main()
