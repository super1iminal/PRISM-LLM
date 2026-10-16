"""Freeze each instance's (or training set's) final rules from a run and certify them, unchanged, on every
instance of a dataset.

With the general rule vocabulary (`rules.general`), conditions only name instance constants and domain
features, so a rule set written for one instance means something on every other. This checks whether it
still meets every requirement there, in the worst case, by exact model checking per target instance.
Rules that do not parse on a target (e.g. a run without the general vocabulary) are reported as such. Column
`trained` marks the targets a rule set was written for (its own instance, or its training set's members) when
the targets are the run's own dataset.

Usage (from PRISM-Guided-Learning/):
    python src/transfer.py out/results/ablations/<condition>/<seed dir> [--data <dataset>] [--targets 12-19] [--workers N]
Settings: configs/transfer/default.yaml (`--config FILE`, `--set key=value`; `--workers` is a shortcut).
Writes <run dir>/transfer_<dataset stem>.csv (one row per source and target) and prints a summary of the targets
checked. With `--targets`, rows of an existing CSV for the other targets are kept.
"""
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import CONFIG_DIR, load_file, load_run_config
from core.domain import load_domain
from core.prism import PrismError, PrismRunner
from core.rules import RuleError, open_ties
from core.verifier import PolicyVerifier
from results_io import sample_records

DEFAULT_CONFIG = CONFIG_DIR / "transfer" / "default.yaml"


@dataclass
class TransferConfig:
    """configs/transfer/default.yaml documents each key."""
    workers: int
    exact_near: float
    sealed: Dict[str, str]          # dataset file name -> its sealed test target ids ("*": all)


def guard_sealed(dataset: str, ids: List[int], unseal: bool = False) -> None:
    """Refuse to certify sealed test targets (configs/transfer/default.yaml, `sealed`) unless `unseal`: the test set
    stays unread until Asher asks for the test pass."""
    spec = load_file(DEFAULT_CONFIG, TransferConfig).sealed.get(Path(dataset).name)
    hit = [] if spec is None else (list(ids) if spec == "*" else sorted(set(ids) & set(parse_ids(spec))))
    if hit and not unseal:
        raise SystemExit(f"refusing to check sealed test targets of {dataset} ({len(hit)}, e.g. {hit[:3]}); "
                         f"pass --unseal only for the test pass Asher asked for")


def training_instances(run_dir: Path) -> Dict[int, List[int]]:
    """Source sample id -> the ids of the instances its rules were written for: a training set's members, or
    the sample's own instance."""
    return {sid: [int(m["instance"]) for m in record["members"]] if "members" in record else [sid]
            for sid, record in sample_records(run_dir).items()}


def final_rules(run_dir: Path) -> Dict[int, List[Tuple[str, str]]]:
    """Source sample id -> its final rules, from the run's outputs/sample_*.json. Instances that ended with an
    error (no final rules) are skipped."""
    return {sid: [(r["condition"], r["action"]) for r in record["final_rules"]]
            for sid, record in sample_records(run_dir).items() if "final_rules" in record}


def determinized_rules(run_dir: Path) -> Dict[int, List[Tuple[str, str]]]:
    """Source sample id -> its rules with `any` resolved into the order chosen on its training instances
    (src/determinize.py's <run dir>/determinized.json)."""
    data = json.loads((run_dir / "determinized.json").read_text(encoding="utf-8"))
    return {int(sid): [tuple(r) for r in entry["rules"]] for sid, entry in data.items()}


def polished_rules(run_dir: Path) -> Dict[int, List[Tuple[str, str]]]:
    """Source sample id -> its rules after src/polish.py's deletions (<run dir>/polished.json)."""
    data = json.loads((run_dir / "polished.json").read_text(encoding="utf-8"))
    return {int(sid): [tuple(r) for r in entry["rules"]] for sid, entry in data.items()}


def norepair_rules(run_dir: Path, key: str = "rules") -> Dict[int, List[Tuple[str, str]]]:
    """Source sample id -> the pick's list without repair (`planner.pick: candidates` with `planner.repair`;
    <run dir>/norepair.json), as chosen or (key "determinized") with `any` resolved by its order."""
    data = json.loads((run_dir / "norepair.json").read_text(encoding="utf-8"))
    return {int(sid): [tuple(r) for r in entry[key]] for sid, entry in data.items()}


RULES = {"final": (final_rules, ""), "determinized": (determinized_rules, "_determinized"),
         "polished": (polished_rules, "_polished"), "open": (final_rules, "_open"),
         "norepair": (norepair_rules, "_norepair"),
         "norepair_determinized": (lambda run_dir: norepair_rules(run_dir, "determinized"), "_norepair_determinized")}


def csv_path(run_dir: Path, dataset: str, rules: str = "final") -> Path:
    """Where the transfer check of `rules` (final or determinized) on `dataset` is written."""
    return run_dir / f"transfer_{Path(dataset).stem}{RULES[rules][1]}.csv"


def parse_ids(spec: str) -> List[int]:
    """Instance ids from a spec such as "12-19" or "0,4,8" or "0-3,12": ranges are inclusive."""
    ids = []
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        ids.extend(range(int(lo), int(hi or lo) + 1))
    return ids


def merge(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """`new` rows, plus the rows of an earlier transfer CSV at `path` for targets `new` does not cover."""
    if not path.exists():
        return new
    old = pd.read_csv(path)
    return pd.concat([old[~old.target.isin(new.target.unique())], new], ignore_index=True)


def certify(domain, cfg, runner, target, sources, exact_near: float = 0.0, deciding_only: bool = False) -> List[dict]:
    """Every source rule set, checked on one target instance. A worst case within `exact_near` of its threshold is
    re-checked with the exact solvers (`PolicyVerifier.verify_exact`), so the verdict holds (column `exact`). With
    `deciding_only`, only when those near values decide the verdict: a requirement that fails by more than
    `exact_near` fails either way, on the same assumption that values outside the band are accurate enough."""
    verifier = PolicyVerifier(domain, target, cfg, runner)
    spec, vocab = verifier.spec, verifier.spec.vocabulary(cfg.rules)
    rows = []
    for source, rules in sources.items():
        row = {"source": source, "target": int(target.id)}
        try:
            policy = spec.policy(rules, vocab)
        except RuleError as e:
            rows.append({**row, "parsed": False, "error": str(e).splitlines()[0]})
            continue
        try:
            v = verifier.verify(policy, analysis=False)
        except (PrismError, subprocess.TimeoutExpired) as e:   # undecided: counts as not certified
            rows.append({**row, "parsed": True, "certified": False, "requirements": len(spec.requirements),
                         "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"})
            continue
        exact = False
        near = [abs(v.worst[r.name] - r.threshold) < exact_near for r in spec.requirements]
        decides = not deciding_only or all(n or r.satisfied(v.worst[r.name]) for n, r in zip(near, spec.requirements))
        if exact_near and any(near) and decides:
            v_exact, _ = verifier.verify_exact(policy)
            v, exact = (v_exact, True) if v_exact else (v, False)
        met = [r.name for r in spec.requirements if r.satisfied(v.worst[r.name])]
        rows.append({**row, "parsed": True, "met": len(met), "requirements": len(spec.requirements),
                     "certified": len(met) == len(spec.requirements), "uncovered": v.uncovered_situations, "exact": exact,
                     **{f"worst_{r.name}": v.worst[r.name] for r in spec.requirements}})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data", default=None, help="Target dataset (default: the run's own)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Transfer config file")
    parser.add_argument("--set", action="append", default=[], help="Override key=value (repeatable)")
    parser.add_argument("--workers", type=int, default=None, help="Shortcut for --set workers=N")
    parser.add_argument("--rules", choices=list(RULES), default="final",
                        help="The run's final rules; those of src/determinize.py or src/polish.py (written to "
                             "*_determinized.csv, *_polished.csv); or the final rules with their action orders left "
                             "open (core.rules.open_ties, *_open.csv: certified however ties are broken); or the pick "
                             "without repair, as chosen or resolved (norepair.json, *_norepair*.csv)")
    parser.add_argument("--targets", default=None,
                        help='Only these target ids, e.g. "12-19" (rows for other targets in an existing CSV are kept)')
    parser.add_argument("--unseal", action="store_true", help="Allow sealed test targets (the test pass only)")
    args = parser.parse_args(argv)
    settings = load_file(args.config, TransferConfig,
                         args.set + ([f"workers={args.workers}"] if args.workers is not None else []))

    cfg = load_run_config(args.run_dir)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    dataset = args.data or cfg.domain.dataset
    targets = domain.load_instances(dataset)
    if args.targets:
        wanted = set(parse_ids(args.targets))
        targets = [t for t in targets if int(t.id) in wanted]
    guard_sealed(dataset, [int(t.id) for t in targets], args.unseal)
    sources = RULES[args.rules][0](args.run_dir)
    if args.rules == "open":      # ties between the actions a rule allows left open: certified for every tie-break
        actions = list(domain.spec(targets[0]).actions)
        sources = {s: open_ties(r, actions) for s, r in sources.items()}
    runner = PrismRunner(cfg.prism)
    with ThreadPoolExecutor(settings.workers) as pool:
        rows = [row for part in pool.map(lambda t: certify(domain, cfg, runner, t, sources, settings.exact_near),
                                                  targets) for row in part]

    df = pd.DataFrame(rows).sort_values(["source", "target"])
    trained = training_instances(args.run_dir) if Path(dataset).name == Path(cfg.domain.dataset).name else {}
    df["trained"] = [t in trained.get(s, []) for s, t in zip(df.source, df.target)]
    out = csv_path(args.run_dir, dataset, args.rules)
    merge(df, out).sort_values(["source", "target"]).to_csv(out, index=False)
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
