"""Several method.py runs side by side in one process (threads: the LLM calls overlap, the numpy work shares one
core), all with the same settings.

    python src/fresh/multi.py --runs rooms=0,4,8 crossing=60,64,68 adjacent=120,124,128 --resolve-in-loop --out <dir>
    python src/fresh/multi.py --runs set0=0,4,8 set1=1,5,9 set2=2,6,10 set3=3,7,11 --kind random \
        --data grid_20_large.csv --hide hazard --out <dir>
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fresh.method as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="label=ids")
ap.add_argument("--chains", type=int, default=8)
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--kind", choices=["family", "random"], default="family")
ap.add_argument("--data", default="families_v0.csv")
ap.add_argument("--hide", default="hazard")
ap.add_argument("--resolve-in-loop", action="store_true")
ap.add_argument("--no-action-lines", action="store_true")
ap.add_argument("--position", action="store_true", help="allow x, y and the grid constants in conditions")
ap.add_argument("--out", required=True)
args = ap.parse_args()
M.CFG.kind, M.CFG.data, M.CFG.hidden = args.kind, args.data, tuple(h for h in args.hide.split(",") if h)
M.CFG.resolve_in_loop, M.CFG.action_lines = args.resolve_in_loop, not args.no_action_lines
runs = [(label, [int(i) for i in ids.split(",")]) for label, ids in (r.split("=") for r in args.runs)]


def one(label, ids):
    s = M.run(label, args.chains, args.rounds, Path(args.out), args.position, ids)
    b = s["best"]
    return f"{label}: kept {b['from']} score {b['score']} det {b['det_score']} cost ${s['usage']['cost']:.3f}"


with ThreadPoolExecutor(max_workers=len(runs)) as pool:
    for line in pool.map(lambda r: one(*r), runs):
        print(line, flush=True)
