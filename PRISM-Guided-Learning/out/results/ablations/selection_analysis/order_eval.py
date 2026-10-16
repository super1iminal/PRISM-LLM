"""Selection rules for the action order, evaluated on order_cache.jsonl (numpy certification of every order)."""
import json
import sys
from collections import defaultdict

import numpy as np

GROUPS = {0: "rooms", 1: "crossing", 2: "adjacent"}


def margin(name, value, reqs):
    thr, bound = reqs[name]
    return value - thr if bound == ">=" else thr - value


def score_of(member_vals, reqs_by_member):
    """keep-best score summed over members (best = worst), or None if a member is left open."""
    f = s = 0.0
    for m, vals in member_vals.items():
        if vals is None:
            return None
        reqs = reqs_by_member[m]
        for name, v in vals.items():
            gap = -margin(name, v, reqs)
            if gap > 0:
                f += 1
                s += gap
    return (f, f, round(s, 9), round(s, 9))


def features(member_vals, reqs_by_member, trained):
    ms = {m: sorted(margin(n, v, reqs_by_member[m]) for n, v in vals.items()) for m, vals in member_vals.items()}
    rel = {m: sorted(margin(n, v, reqs_by_member[m]) / (1 - reqs_by_member[m][n][0]) for n, v in vals.items())
           for m, vals in member_vals.items()}
    allm = sorted(x for m in ms for x in ms[m])
    largest = str(trained[-1])
    return {"min": allm[0], "sum": sum(allm), "large": ms[largest][0], "leximin": tuple(allm),
            "rel": min(x for m in rel for x in rel[m]), "large_lex": tuple(ms[largest])}


RULES = ["first", "min", "sum", "large", "leximin", "rel", "large_lex"]


def choose(d):
    trained = d["trained"]
    reqs = {m: {n: (t, b) for n, t, b in r} for m, r in d["reqs"].items()}
    n = len(d["orders"])
    scores, feats = [], []
    for k in range(n):
        mv = {str(m): d["values"][str(m)][k] for m in trained}
        sc = score_of(mv, reqs)
        scores.append(sc)
        feats.append(None if sc is None else features(mv, reqs, trained))
    valid = [k for k in range(n) if scores[k] is not None]
    if not valid:
        return None
    best = min(scores[k] for k in valid)
    tied = [k for k in valid if scores[k] == best]
    out = {"first": tied[0], "tied": tied, "solved": best[0] == 0}
    for rule in RULES[1:]:
        key = rule
        out[rule] = max(tied, key=lambda k: (feats[k][key], -k))
    return out


def certified(d, k):
    return sum(bool(d["values"][str(t)][k]) for t in d["targets"])


def benchmark(d):
    if d["dataset"] == "families_v0.csv":
        return "families/" + GROUPS[d["sid"]]
    return "grids_noclose" if "closer" in d["hidden_features"] else "grids"


def main(path, core_only=False):
    rows = [json.loads(l) for l in open(path)]
    if core_only:
        rows = [d for d in rows if d["run"].startswith("core/")]
    agg = defaultdict(lambda: defaultdict(list))
    for d in rows:
        c = choose(d)
        if c is None:
            continue
        b = benchmark(d)
        fin = sum(bool(d["final"][str(t)]) for t in d["targets"])
        per_order = [certified(d, k) for k in range(len(d["orders"]))]
        rec = agg[b]
        rec["final"].append(fin)
        for rule in RULES:
            rec[rule].append(per_order[c[rule]])
        rec["tied_mean"].append(np.mean([per_order[k] for k in c["tied"]]))
        rec["oracle"].append(max(per_order))
        rec["all_mean"].append(np.mean(per_order))
        rec["n_tied"].append(len(c["tied"]))
        rec["solved"].append(c["solved"])
    cols = ["final", "first"] + RULES[1:] + ["tied_mean", "all_mean", "oracle"]
    print("| benchmark | lists | solved | tied orders | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * (3 + len(cols)))
    for b in sorted(agg):
        rec = agg[b]
        print(f"| {b} | {len(rec['final'])} | {np.mean(rec['solved']):.0%} | {np.mean(rec['n_tied']):.1f} | " +
              " | ".join(f"{np.mean(rec[c]):.2f}" for c in cols) + " |")


if __name__ == "__main__":
    main(sys.argv[1], core_only=len(sys.argv) > 2 and sys.argv[2] == "core")
