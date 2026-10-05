"""Per-instance ceilings: what any controller could achieve, independent of the LLM.

For each instance of a dataset, on the bare MDP (no policy, full state observed):
  * the optimum of each requirement on its own (Pmax for >=, Pmin for <=);
  * whether the thresholds are achievable *jointly* by one controller (PRISM multi-objective
    query; randomized/history-dependent controllers allowed, so this is an upper bound).
A failure on an instance whose thresholds are not jointly achievable is not the LLM's fault.

Usage: python src/ceilings.py --domain gridworld --data grid_20_balanced.csv
Writes out/results/ceilings/<domain>_<dataset>.csv and .md.
"""
import argparse
import os
from pathlib import Path

import pandas as pd

from config import load_config
from core.domain import load_domain
from core.prism import PrismRunner
from core.verifier import PolicyVerifier
from settings import RESULTS_PATH

CEILINGS_PATH = RESULTS_PATH / "ceilings"


def ceilings(domain_name: str, dataset: str) -> pd.DataFrame:
    """One row per instance: each requirement's optimum and whether all thresholds are jointly achievable."""
    cfg = load_config()
    domain = load_domain(domain_name)
    runner = PrismRunner(cfg.prism)
    rows = []
    for idx, instance in enumerate(domain.load_instances(dataset)):
        verifier = PolicyVerifier(domain, instance, cfg, runner)
        reqs = verifier.spec.requirements
        optimum = runner.run(verifier.model, [f"{r.best_op()}=? [ {r.formula} ]" for r in reqs]).initial_values
        joint = verifier.jointly_feasible()   # None: PRISM could not decide (e.g. step-bounded objectives)
        row = {"sample_id": idx, "instance": instance.id, "jointly_feasible": joint}
        for r, value in zip(reqs, optimum):
            row[f"optimum_{r.name}"] = value
            row[f"threshold_{r.name}"] = r.threshold
            row[f"achievable_{r.name}"] = r.satisfied(value)
        row["individually_feasible"] = all(row[f"achievable_{r.name}"] for r in reqs)
        rows.append(row)
        print(f"instance {instance.id}: individually {row['individually_feasible']}, jointly {joint}", flush=True)
    return pd.DataFrame(rows).set_index("sample_id")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="gridworld")
    parser.add_argument("--data", default="grid_20_balanced.csv")
    args = parser.parse_args()
    df = ceilings(args.domain, args.data)
    os.makedirs(CEILINGS_PATH, exist_ok=True)
    stem = CEILINGS_PATH / f"{args.domain}_{Path(args.data).stem}"
    df.to_csv(stem.with_suffix(".csv"))

    reqs = [c[len("optimum_"):] for c in df.columns if c.startswith("optimum_")]
    lines = [f"# Ceilings: {args.domain} / {args.data}", "",
             "Bare MDP, full state observed. Optimum per requirement on its own; joint = one controller "
             "meets every threshold at once (PRISM multi-objective, an upper bound).", "",
             f"- Instances: {len(df)}",
             f"- Every threshold achievable on its own: {int(df.individually_feasible.sum())}/{len(df)}",
             f"- All thresholds achievable jointly: {int((df.jointly_feasible == True).sum())}/{len(df)}"  # noqa: E712
             + (f" ({int(df.jointly_feasible.isna().sum())} undecided)" if df.jointly_feasible.isna().any() else ""),
             "", "| instance | jointly | " + " | ".join(reqs) + " |", "|---|---|" + "---|" * len(reqs)]
    for _, row in df.iterrows():
        cells = [f"{row[f'optimum_{r}']:.3f}{'' if row[f'achievable_{r}'] else ' ✗'}" for r in reqs]
        lines.append(f"| {row['instance']} | {row['jointly_feasible']} | " + " | ".join(cells) + " |")
    lines += ["", "✗ = the optimum is below the threshold."]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))
    print(stem.with_suffix(".md"))


if __name__ == "__main__":
    main()
