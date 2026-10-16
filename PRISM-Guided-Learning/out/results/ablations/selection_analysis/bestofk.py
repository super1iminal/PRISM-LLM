"""Best-of-K selection over independent lists of one benchmark (Monte Carlo over random K-subsets of the pool)."""
import sys

import numpy as np
import pandas as pd

KEYS = {
    "score": lambda r: (r.fails, round(r.shortfall, 6)),
    "score+min": lambda r: (r.fails, round(r.shortfall, 6), -r.min_margin),
    "score+large": lambda r: (r.fails, round(r.shortfall, 6), -r.large_margin),
    "score+sum": lambda r: (r.fails, round(r.shortfall, 6), -r.sum_margin),
    "score+rules": lambda r: (r.fails, round(r.shortfall, 6), r.num_rules),
    "min": lambda r: (-r.min_margin,),
}


def simulate(pool: pd.DataFrame, k: int, key, rng, trials=20000):
    recs = list(pool.itertuples())
    keys = [key(r) for r in recs]
    cert = np.array([r.certified for r in recs], dtype=float)
    n = len(recs)
    if k == 1:
        return cert.mean()
    total = 0.0
    for _ in range(trials):
        idx = rng.choice(n, size=k, replace=False)
        best = min(keys[i] for i in idx)
        tied = [i for i in idx if keys[i] == best]
        total += cert[tied].mean()
    return total / trials


def main(path, conds=None):
    df = pd.read_csv(path)
    if conds:
        df = df[df.condition.isin(conds.split(","))]
    rng = np.random.default_rng(0)
    print("| benchmark | pool | K | " + " | ".join(KEYS) + " |")
    print("|---|---|---|" + "---|" * len(KEYS))
    for bench, pool in df.groupby("bench"):
        for k in (1, 3, 5):
            if len(pool) < k:
                continue
            vals = [simulate(pool, k, key, rng, 4000) for key in KEYS.values()]
            print(f"| {bench} | {len(pool)} | {k} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
