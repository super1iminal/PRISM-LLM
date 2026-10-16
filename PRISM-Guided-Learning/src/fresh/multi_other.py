"""Repeats of the MER or Pac-Man method side by side in one process (threads).

    python src/fresh/multi_other.py mer --dataset mer_transfer.csv --repeats 1 2 3 --out out/fresh/mer_x
    python src/fresh/multi_other.py pacman --uniform --repeats 1 2 3 --out out/fresh/pm_v2
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser()
ap.add_argument("method", choices=["mer", "pacman"])
ap.add_argument("--dataset", nargs="+", default=["mer_transfer.csv"])
ap.add_argument("--uniform", action="store_true")
ap.add_argument("--literals", action="store_true", help="Pac-Man v3: numbers allowed on positions and headings")
ap.add_argument("--optimal", action="store_true", help="MER v2: feedback from the best scheduler's visited states")
ap.add_argument("--early", type=int, default=None, help="Pac-Man v4: uniform mode over the first moves only")
ap.add_argument("--repeats", nargs="+", type=int, default=[1, 2, 3])
ap.add_argument("--chains", type=int, default=8)
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--out", required=True)
args = ap.parse_args()

if args.method == "mer":
    import fresh.mer_method as M
    M.OPTIMAL_FEEDBACK = args.optimal

    def one(job):
        ds, k = job
        out = Path(args.out) / ("x" if ds == "mer_transfer.csv" else "n") / f"run{k}"
        s = M.run(ds, args.chains, args.rounds, out)
        b = s["best"]
        if b is None:
            return f"{ds} run{k}: no valid list, cost ${s['usage']['cost']:.3f}"
        return f"{ds} run{k}: kept {b['from']} score {b['score']} det {b['det_score']} cost ${s['usage']['cost']:.3f}"
else:
    import fresh.pm_method as M
    M.UNIFORM = args.uniform
    M.LITERALS_ON_POSITIONS = args.literals
    M.EARLY_STEPS = args.early

    def one(k):
        s = M.run(args.chains, args.rounds, Path(args.out) / f"run{k}")
        b = s["best"]
        return f"run{k}: kept {b['from']} crash {b['crash']} det {b['det_score']} cost ${s['usage']['cost']:.3f}"

jobs = [(ds, k) for ds in args.dataset for k in args.repeats] if args.method == "mer" else args.repeats
with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
    for line in pool.map(one, jobs):
        print(line, flush=True)
