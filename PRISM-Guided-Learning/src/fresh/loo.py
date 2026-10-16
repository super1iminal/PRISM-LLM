"""Training-only transfer proxy: train on two of a family's training members, check the kept list on the third.

    python src/fresh/loo.py rooms --out out/fresh/loo/<name> [method options]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fresh.gw import FAMILIES, load  # noqa: E402
from fresh.method import run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("family")
ap.add_argument("--out", required=True)
ap.add_argument("--chains", type=int, default=6)
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--position", action="store_true")
ap.add_argument("--phases", action="store_true")
ap.add_argument("--repair-steps", type=int, default=40)
args = ap.parse_args()
ids = FAMILIES[args.family]
folds = [([ids[0], ids[1]], ids[2]), ([ids[0], ids[2]], ids[1]), ([ids[1], ids[2]], ids[0])]


def fold(train, test):
    out = Path(args.out) / f"test{test}"
    s = run(args.family, args.chains, args.rounds, out, args.position, train, 8, args.repair_steps, args.phases)
    m = load([test])[0]
    if args.phases:
        m.all_phases()
    rows = []
    for f in s["finals"]:
        r = m.check(f["rules"], best=False)
        rows.append({"from": f["from"], "train_score": f["score"], "test_certified": r.certified,
                     "test_margin": r.margin})
    best = s["best"]
    rb = m.check(best["rules"], best=False) if best else None
    return {"train": train, "test": test, "kept_train_score": best["score"] if best else None,
            "kept_test_certified": rb.certified if rb else False, "kept_test_margin": rb.margin if rb else None,
            "finals": rows}


with ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(lambda f: fold(*f), folds))
Path(args.out, "loo.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
for r in results:
    print(r["train"], "->", r["test"], "kept", r["kept_train_score"], "test certified", r["kept_test_certified"],
          f"margin {r['kept_test_margin']:+.3f}" if r["kept_test_margin"] is not None else "",
          "| finals certified on test:", sum(x["test_certified"] for x in r["finals"]), "/", len(r["finals"]))
