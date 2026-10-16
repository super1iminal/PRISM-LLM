"""Random grids (grid_20_large.csv): one list per training set, certified on ids 12-19 (8 targets per list).

    python src/fresh/certify_grids.py <run> --lists l0.json l1.json l2.json l3.json [--hide hazard,closer] [--run-transfer]
    python src/fresh/certify_grids.py <run>        (tally)
Each list file is a method.py summary (its kept list, `any` resolved) or {"rules": [[condition, action], ...]}.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fresh.certify import MAIN  # noqa: E402

TRAIN_SETS = [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
TARGETS = list(range(12, 20))


def read(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    rules = obj["best"]["det_rules"] if "best" in obj else obj["rules"]
    return [tuple(r) for r in rules]


def write(run: Path, lists, hidden):
    (run / "outputs").mkdir(parents=True, exist_ok=True)
    config = {"approach": "symbolic", "domain": {"name": "gridworld", "dataset": "grid_20_large.csv"},
              "rules": {"extended": True, "general": True, "hidden_features": list(hidden)}}
    (run / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    for i, (rules, ids) in enumerate(zip(lists, TRAIN_SETS)):
        sample = {"final_rules": [{"condition": c, "action": a} for c, a in rules],
                  "members": [{"instance": str(m)} for m in ids]}
        (run / "outputs" / f"sample_{i:03d}.json").write_text(json.dumps(sample, indent=1), encoding="utf-8")


def certify(run: Path):
    cmd = [str(MAIN.parent / ".venv" / "Scripts" / "python"), "src/transfer.py", str(run.resolve()),
           "--targets", "12-19", "--workers", "1"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with open(run / "transfer.log", "w") as log:
        return subprocess.run(cmd, cwd=MAIN, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def tally(run: Path):
    csv = next(run.glob("transfer_*.csv"))
    df = pd.read_csv(csv)
    per = [int(df[(df.source == i) & df.target.isin(TARGETS)].certified.sum()) for i in range(len(TRAIN_SETS))]
    return per, sum(per) / len(per)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--lists", nargs=4)
    ap.add_argument("--hide", default="hazard")
    ap.add_argument("--run-transfer", action="store_true")
    args = ap.parse_args()
    run = Path(args.run)
    if args.lists:
        write(run, [read(Path(p)) for p in args.lists], [h for h in args.hide.split(",") if h])
    if args.run_transfer:
        certify(run)
    if list(run.glob("transfer_*.csv")):
        per, mean = tally(run)
        print(f"{run}: certified per list {per} of 8, mean {mean:.2f}")
