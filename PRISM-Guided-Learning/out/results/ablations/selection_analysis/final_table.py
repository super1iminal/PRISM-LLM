"""The summary table: per benchmark, certified per list for the current choice, best of K by each selection rule
(post hoc; mean and 10-90% range of the chosen list over random K-subsets) and the chain runs (mean, range over
runs). Usage: python final_table.py <lists.csv> <open_table.csv>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXCLUDE = ("haiku_fam_order_six", "haiku_fam_hidden", "haiku_fam_grow", "haiku_ts_hidden", "haiku_ts_grow")
BASELINE = {"families/rooms": "19 PPO", "families/crossing": "19 trees", "families/adjacent": "19 PPO",
            "grids": "6.00 template", "grids_noclose": "6.00 template", "pacman": "4 trees", "mer": "6 trees",
            "mer_n": "6 trees"}
CORE = {"families/rooms": "8.4", "families/crossing": "4.6", "families/adjacent": "16.0", "grids": "6.25 (5.50-7.00)",
        "grids_noclose": "3.40 (3.00-3.75)", "pacman": "2.40 (1-3)", "mer": "2.40 (0-3)", "mer_n": "2.40 (0-6)"}


def best_of(pool, k, key, col, rng, trials=4000):
    recs = list(pool.itertuples())
    keys = [key(r) for r in recs]
    cert = np.array([getattr(r, col) for r in recs], dtype=float)
    picks = []
    for _ in range(trials):
        idx = rng.choice(len(recs), size=min(k, len(recs)), replace=False)
        best = min(keys[i] for i in idx)
        picks.append(cert[[i for i in idx if keys[i] == best]].mean())
    return np.mean(picks), np.percentile(picks, 10), np.percentile(picks, 90)


def fmt(m, lo, hi):
    return f"{m:.2f} ({lo:.1f}-{hi:.1f})"


def main(lists_csv, open_csv):
    res = pd.read_csv(lists_csv)
    res = res[~res.run.str.split("/").str[1].isin(EXCLUDE)]
    opn = pd.read_csv(open_csv)
    opn = opn[opn.any_order.isin(["none", "report"])].dropna(subset=["cert_open"])
    rng = np.random.default_rng(0)
    rules = [("best of 5, keep-best score", res, lambda r: (r.fails, round(r.shortfall, 6)), "certified", 5),
             ("best of 5, margin", res, lambda r: (-r.min_margin,), "certified", 5),
             ("best of 5, largest-member margin", res, lambda r: (-r.large_margin,), "certified", 5),
             ("best of 10, margin", res, lambda r: (-r.min_margin,), "certified", 10),
             ("ties open: best of 6, members passing > shortfall > margin", opn,
              lambda r: (-r.open_passed, round(r.open_short, 6), -r.open_margin), "cert_open", 6)]
    print("| benchmark | rule | lists | certified per list | core (current) | best baseline |")
    print("|---|---|---|---|---|---|")
    for bench in BASELINE:
        pool = res[res.bench == bench]
        if not len(pool):
            continue
        print(f"| {bench} | one list (resolved) | {len(pool)} | {pool.certified.mean():.2f} | {CORE[bench]} | "
              f"{BASELINE[bench]} |")
        for name, df, key, col, k in rules:
            p = df[df.bench == bench]
            if len(p) < max(k, 10):
                continue
            print(f"| {bench} | {name} | {len(p)} | {fmt(*best_of(p, k, key, col, rng))} | | |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
