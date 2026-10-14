"""Fit a baseline on training instances, freeze it, and certify its policies on target instances.

Usage (from PRISM-Guided-Learning/):
    python src/run_baseline.py configs/baselines/<name>.yaml [--set key=value ...]
Settings: the baseline config (BaselineConfig below; configs/baselines/README.md documents each key). Writes
out/results/baselines/<name>/transfer_<dataset stem>.csv in src/transfer.py's format (one row per source policy and
target, `trained` marking the training instances) plus `method`, and prints certified targets per source.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from baselines import load_method
from config import load_config, load_file
from core.domain import load_domain
from core.prism import PrismRunner
from settings import RESULTS_PATH
from transfer import certify, parse_ids

BASELINES_PATH = RESULTS_PATH / "baselines"


@dataclass
class BaselineConfig:
    """A configs/baselines/*.yaml file (configs/baselines/README.md documents each key)."""
    method: str
    condition: Optional[str]
    dataset: str
    train: str
    targets: str
    workers: int
    exact_near: float
    params: Dict[str, Any]


def run(config_path: Path, overrides=(), out_root: Path = BASELINES_PATH) -> pd.DataFrame:
    """Fit, certify and write the results of the baseline in `config_path`; returns them."""
    settings = load_file(config_path, BaselineConfig, overrides)
    cfg = load_config(settings.condition)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(i.id): i for i in domain.load_instances(settings.dataset)}
    train = parse_ids(settings.train) if settings.train else []
    baseline = load_method(settings.method)(domain, settings.params)
    baseline.fit([instances[i] for i in train])

    runner = PrismRunner(cfg.prism)
    targets = [instances[i] for i in parse_ids(settings.targets)]
    with ThreadPoolExecutor(settings.workers) as pool:
        parts = pool.map(lambda t: certify(domain, cfg, runner, t, baseline.rules(t), settings.exact_near), targets)
        rows = [row for part in parts for row in part]
    df = pd.DataFrame(rows).sort_values(["source", "target"], key=lambda c: c.astype(str))
    df["trained"] = df.target.isin(train)
    df["method"] = settings.method
    out = out_root / config_path.stem / f"transfer_{Path(settings.dataset).stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="Baseline config, e.g. configs/baselines/template.yaml")
    parser.add_argument("--set", action="append", default=[], help="Override key=value (repeatable)")
    args = parser.parse_args(argv)
    df = run(args.config, args.set)
    certified = df.get("certified", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    per_source = df.assign(certified=certified)[~df.trained].groupby("source").certified.sum()
    print(f"held-out targets certified per source ({df[~df.trained].target.nunique()} held-out targets):")
    for source, n in per_source.sort_values(ascending=False).items():
        print(f"  {source}: {int(n)}")


if __name__ == "__main__":
    main()
