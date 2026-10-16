"""Write three frozen family lists as a run directory and certify them with the shared transfer.py (main checkout).

    python src/fresh/certify.py out/fresh/<name> --lists rooms.json crossing.json adjacent.json [--run]
    python src/fresh/certify.py out/fresh/<name> --summary        (per-list counts from transfer_families_v0.csv)

Each list file is {"rules": [[condition, action], ...]} (or a method.py summary file: its best list is used).
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

MAIN = Path("C:/Users/asher/Documents/Coding/llmtrust/PRISM-LLM/PRISM-Guided-Learning")
FAMILY_IDS = {"rooms": ["0", "4", "8"], "crossing": ["60", "64", "68"], "adjacent": ["120", "124", "128"]}
TARGETS = {"rooms": range(16, 36), "crossing": range(76, 96), "adjacent": range(136, 156)}
CONFIG = {"approach": "symbolic", "domain": {"name": "gridworld", "dataset": "families_v0.csv"},
          "rules": {"extended": True, "general": True, "hidden_features": ["hazard"]}}


def read_rules(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "best" in obj:
        obj = obj["best"]
    return [tuple(r) if isinstance(r, list) else (r["condition"], r["action"]) for r in obj["rules"]]


def write_run(run: Path, lists):
    (run / "outputs").mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(CONFIG, indent=1), encoding="utf-8")
    for i, fam in enumerate(["rooms", "crossing", "adjacent"]):
        sample = {"final_rules": [{"condition": c, "action": a} for c, a in lists[fam]],
                  "members": [{"instance": m} for m in FAMILY_IDS[fam]]}
        (run / "outputs" / f"sample_{i:03d}.json").write_text(json.dumps(sample, indent=1), encoding="utf-8")


def certify(run: Path, workers: int = 2):
    cmd = [str(MAIN.parent / ".venv" / "Scripts" / "python"), "src/transfer.py", str(run.resolve()),
           "--targets", "16-35,76-95,136-155", "--workers", str(workers)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    return subprocess.run(cmd, cwd=MAIN, env=env).returncode


def summary(run: Path):
    df = pd.read_csv(run / "transfer_families_v0.csv")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--lists", nargs=3, help="rooms, crossing, adjacent list files")
    ap.add_argument("--run-transfer", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    run = Path(args.run)
    if args.lists:
        write_run(run, {fam: read_rules(Path(p)) for fam, p in zip(["rooms", "crossing", "adjacent"], args.lists)})
    if args.run_transfer:
        certify(run, args.workers)
    if (run / "transfer_families_v0.csv").exists():
        df = summary(run)
        print(df.columns.tolist())
        print(df.head())


if __name__ == "__main__":
    main()
