"""Best of K lists per benchmark by selection rules on training values, with ties open or resolved (open_table.py's
CSV). Each rule pairs a training key with the held-out metric of the list it freezes: ties open (cert_open, the
worst case over the allowed actions) or the resolved order (cert_resolved). Monte Carlo over K-subsets.
Usage: python open_eval.py <open_table.csv> [resolving | native | all]   (default: runs that resolve `any`;
native: runs that keep `any`; three-member conditions only)"""
import sys

import numpy as np
import pandas as pd

RULES = {   # name: (key to minimise, held-out column)
    "open: passed, shortfall, margin": (lambda r: (-r.open_passed, round(r.open_short, 6), -r.open_margin), "cert_open"),
    "open: margin": (lambda r: (-r.open_margin,), "cert_open"),
    "open: largest-member margin": (lambda r: (-r.open_large,), "cert_open"),
    "resolved: keep-best score": (lambda r: (-r.res_passed, round(r.res_short, 6)), "cert_resolved"),
    "resolved: margin": (lambda r: (-r.res_margin,), "cert_resolved"),
    "resolved: largest-member margin": (lambda r: (-r.res_large,), "cert_resolved"),
    "open margin, resolved list": (lambda r: (-r.open_margin,), "cert_resolved"),
}
EXCLUDE = ("haiku_fam_order_six", "haiku_fam_hidden", "haiku_fam_grow", "haiku_ts_hidden", "haiku_ts_grow")


def best_of(pool, k, key, col, rng, trials=4000):
    recs = list(pool.itertuples())
    keys = [key(r) for r in recs]
    cert = np.array([getattr(r, col) for r in recs], dtype=float)
    if k >= len(recs):
        best = min(keys)
        return cert[[i for i, x in enumerate(keys) if x == best]].mean()
    total = 0.0
    for _ in range(trials):
        idx = rng.choice(len(recs), size=k, replace=False)
        best = min(keys[i] for i in idx)
        total += cert[[i for i in idx if keys[i] == best]].mean()
    return total / trials


def main(path, scope="resolving"):
    df = pd.read_csv(path).dropna(subset=["cert_open", "cert_resolved"])
    family = df.run.str.split("/").str[1]
    df = df[~family.isin(EXCLUDE)]
    if scope == "resolving":
        df = df[df.any_order.isin(["all", "rotations", "declared"])]
    elif scope == "native":                      # lists the loop wrote and verified with `any` kept
        df = df[df.any_order.isin(["none", "report"])]
    rng = np.random.default_rng(0)
    print("| benchmark | lists | K | one list, resolved | one list, ties open | " + " | ".join(RULES) + " |")
    print("|---|---|---|---|---|" + "---|" * len(RULES))
    for bench, pool in df.groupby("bench"):
        for k in (5, 6, 10):
            vals = [best_of(pool, k, key, col, rng) for key, col in RULES.values()]
            print(f"| {bench} | {len(pool)} | {k} | {pool.cert_resolved.mean():.2f} | {pool.cert_open.mean():.2f} | " +
                  " | ".join(f"{v:.2f}" for v in vals) + " |")
    print()
    print("Spearman correlation with held-out certification (ties open / resolved):")
    for bench, pool in df.groupby("bench"):
        o = pool[["cert_open", "open_passed", "open_short", "open_margin", "open_large"]].corr("spearman")["cert_open"]
        r = pool[["cert_resolved", "res_passed", "res_short", "res_margin", "res_large", "open_margin"]] \
            .corr("spearman")["cert_resolved"]
        print(f"  {bench} ({len(pool)}): open " + ", ".join(f"{k[5:]} {v:+.2f}" for k, v in o.items() if k != "cert_open") +
              "; resolved " + ", ".join(f"{k} {v:+.2f}" for k, v in r.items() if k != "cert_resolved"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "resolving")
