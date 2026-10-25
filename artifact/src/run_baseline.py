"""Fit a baseline on training instances, freeze it, and certify its policies on target instances.

Usage (from artifact/):
    python src/run_baseline.py configs/baselines/<name>.yaml [--set key=value ...]
Settings: the baseline config (BaselineConfig below). Writes
out/results/baselines/<name>/transfer_<dataset stem>.csv in src/transfer.py's format (one row per source policy and
target, `trained` marking the instances that source was fitted on) plus `method`, `fit.json` for methods that
report what their fit produced (e.g. a tree's size and rules), and prints certified targets per source. Methods
whose policies are per-state (`Baseline.explicit`) are certified through each target's explicit model
(core/explicit.py), the others through their rules. Rows
of an existing CSV for targets this run does not check are kept. `--rules open` certifies the same lists with their
action orders left open (core.rules.open_ties), into transfer_<dataset stem>_open.csv: certified however ties are
broken.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from baselines import load_method
from core.explicit import ExplicitMDP
from config import load_config, load_file
from core.domain import load_domain
from core.prism import PrismRunner
from core.rules import open_ties
from settings import RESULTS_PATH
from transfer import certify, guard_sealed, merge, parse_ids

BASELINES_PATH = RESULTS_PATH / "baselines"


@dataclass
class BaselineConfig:
    """A configs/baselines/*.yaml file."""
    method: str
    condition: Optional[str]
    dataset: str
    target_dataset: Optional[str]
    train: str
    targets: str
    workers: int
    exact_near: float
    params: Dict[str, Any]


def run(config_path: Path, overrides=(), out_root: Path = BASELINES_PATH, rules: str = "final",
        unseal: bool = False) -> pd.DataFrame:
    """Fit, certify and write the results of the baseline in `config_path` (`rules`: final, or open: with the
    action orders of its lists left open); returns them."""
    settings = load_file(config_path, BaselineConfig, overrides)
    guard_sealed(settings.target_dataset or settings.dataset, parse_ids(settings.targets), unseal)
    cfg = load_config(settings.condition)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(i.id): i for i in domain.load_instances(settings.dataset)}
    train = parse_ids(settings.train) if settings.train else []
    baseline = load_method(settings.method)(domain, settings.params)
    baseline.config = cfg
    baseline.fit([instances[i] for i in train])

    target_dataset = settings.target_dataset or settings.dataset
    same = Path(target_dataset).name == Path(settings.dataset).name
    pool_of = instances if same else {int(i.id): i for i in domain.load_instances(target_dataset)}
    targets = [pool_of[i] for i in parse_ids(settings.targets)]
    if baseline.explicit:
        if rules == "open":
            raise SystemExit("--rules open: per-state policies have no ties to leave open")
        rows = certify_explicit(domain, cfg, baseline, targets)
    else:
        runner = PrismRunner(cfg.prism)
        actions = list(domain.spec(targets[0]).actions)
        lists = {int(t.id): {s: (open_ties(r, actions) if rules == "open" else r) for s, r in baseline.rules(t).items()}
                 for t in targets}
        with ThreadPoolExecutor(settings.workers) as pool:
            parts = pool.map(lambda t: certify(domain, cfg, runner, t, lists[int(t.id)], settings.exact_near), targets)
            rows = [row for part in parts for row in part]
    df = pd.DataFrame(rows).sort_values(["source", "target"], key=lambda c: c.astype(str))
    df["trained"] = [same and t in (baseline.training(s) or train) for s, t in zip(df.source, df.target)]
    df["method"] = settings.method
    suffix = "_open" if rules == "open" else ""
    out = out_root / config_path.stem / f"transfer_{Path(target_dataset).stem}{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    merge(df, out).sort_values(["source", "target"], key=lambda c: c.astype(str)).to_csv(out, index=False)
    if baseline.summary():
        (out.parent / "fit.json").write_text(json.dumps(baseline.summary(), indent=1), encoding="utf-8")
    return df


def certify_explicit(domain, cfg, baseline, targets) -> List[dict]:
    """Per-state policies certified through each target's explicit model (core/explicit.py), one target at
    a time, since a model of millions of states takes gigabytes."""
    rows = []
    for t in targets:
        mdp = ExplicitMDP(domain, t, cfg)
        rows += mdp.certify(baseline.policies(mdp))
        print(f"  certified on {t.id} ({mdp.S} states)", flush=True)
        del mdp
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="Baseline config, e.g. configs/baselines/template.yaml")
    parser.add_argument("--set", action="append", default=[], help="Override key=value (repeatable)")
    parser.add_argument("--rules", choices=["final", "open"], default="final",
                        help="open: certify the lists with their action orders left open (*_open.csv)")
    parser.add_argument("--unseal", action="store_true", help="Allow sealed test targets (the test pass only)")
    args = parser.parse_args(argv)
    df = run(args.config, args.set, rules=args.rules, unseal=args.unseal)
    certified = df.get("certified", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    per_source = df.assign(certified=certified)[~df.trained].groupby("source").certified.sum()
    print(f"held-out targets certified per source ({df[~df.trained].target.nunique()} held-out targets):")
    for source, n in per_source.sort_values(ascending=False).items():
        print(f"  {source}: {int(n)}")


if __name__ == "__main__":
    main()
