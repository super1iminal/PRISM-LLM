"""Selection among every round's candidate vs among kept lists, on cand_cache.jsonl (numpy certification)."""
import json
import sys
from collections import defaultdict

import numpy as np

GROUPS = {0: "rooms", 1: "crossing", 2: "adjacent"}


def bench(d):
    if "fam" in d["run"]:
        return "families/" + GROUPS[d["sid"]]
    return "grids_noclose" if "noclose" in d["run"] else "grids"


def feats(r):
    """(fails, shortfall, min margin, largest-member margin) of a round's candidate, or None."""
    if r.get("skipped") or r["cert"] is None or any(v is None for v in r["train"].values()):
        return None
    fails, short, margins, per = 0, 0.0, [], []
    for m, vals in r["train"].items():
        mm = []
        for name, (v, thr, bound) in vals.items():
            g = v - thr if bound == ">=" else thr - v
            mm.append(g)
            fails += g < 0
            short += max(0.0, -g)
        margins += mm
        per.append(min(mm))
    return fails, round(short, 6), min(margins), per[-1]


def kept(d):
    """Index of the kept candidate after the last round (last improved)."""
    k = None
    for i, r in enumerate(d["rounds"]):
        if r["improved"] or k is None:
            k = i
    return k


def main(path, trials=3000):
    rows = [json.loads(l) for l in open(path)]
    pools = defaultdict(list)
    for d in rows:
        cands = [(i, feats(r), r["cert"]) for i, r in enumerate(d["rounds"])]
        k = kept(d)
        kc = d["rounds"][k]["cert"]
        kf = feats(d["rounds"][k])
        if kc is None or kf is None:
            continue
        pools[bench(d)].append({"kept": (kf, kc), "cands": [(f, c) for _, f, c in cands if f is not None]})
    rng = np.random.default_rng(0)
    keys = {"score": lambda f: (f[0], f[1]), "min": lambda f: (-f[2],), "large": lambda f: (-f[3],)}
    print("| benchmark | lists | K | kept by score | " + " | ".join(f"kept by {k}" for k in keys) + " | " +
          " | ".join(f"all by {k}" for k in keys) + " |")
    for b, pool in sorted(pools.items()):
        n = len(pool)
        for K in (1, 3, 5):
            if n < K:
                continue
            res = defaultdict(float)
            for _ in range(trials):
                idx = rng.choice(n, size=K, replace=False)
                sub = [pool[i] for i in idx]
                res["kept"] += np.mean([s["kept"][1] for s in sub])
                for name, key in keys.items():
                    ks = [s["kept"] for s in sub]
                    best = min(key(f) for f, _ in ks)
                    res["k_" + name] += np.mean([c for f, c in ks if key(f) == best])
                    al = [fc for s in sub for fc in s["cands"]]
                    best = min(key(f) for f, _ in al)
                    res["a_" + name] += np.mean([c for f, c in al if key(f) == best])
            print(f"| {b} | {n} | {K} | {res['kept'] / trials:.2f} | " +
                  " | ".join(f"{res['k_' + k] / trials:.2f}" for k in keys) + " | " +
                  " | ".join(f"{res['a_' + k] / trials:.2f}" for k in keys) + " |")


if __name__ == "__main__":
    main(sys.argv[1])
