"""Certify single family lists on their own family's 20 validation members (the per-list metric), at most `--jobs`
transfer.py processes at a time, each with one PRISM worker.

    python src/fresh/certify_one.py queue.json --jobs 2
queue.json: [{"dir": "out/fresh/ablation/<name>", "family": "crossing", "rules": [[condition, action], ...]}, ...]
    python src/fresh/certify_one.py --tally out/fresh/ablation/<name> ...
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fresh.certify import CONFIG, FAMILY_IDS, MAIN, TARGETS  # noqa: E402

SPEC = {"rooms": "16-35", "crossing": "76-95", "adjacent": "136-155"}


def write(d: Path, family: str, rules):
    (d / "outputs").mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(CONFIG, indent=1), encoding="utf-8")
    sample = {"final_rules": [{"condition": c, "action": a} for c, a in rules],
              "members": [{"instance": m} for m in FAMILY_IDS[family]], "family": family}
    (d / "outputs" / "sample_000.json").write_text(json.dumps(sample, indent=1), encoding="utf-8")


def start(d: Path, family: str) -> subprocess.Popen:
    cmd = [str(MAIN.parent / ".venv" / "Scripts" / "python"), "src/transfer.py", str(d.resolve()),
           "--targets", SPEC[family], "--workers", "1"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    log = open(d / "transfer.log", "w")
    return subprocess.Popen(cmd, cwd=MAIN, env=env, stdout=log, stderr=subprocess.STDOUT)


def tally(d: Path) -> int:
    family = json.loads((d / "outputs" / "sample_000.json").read_text(encoding="utf-8"))["family"]
    df = pd.read_csv(d / "transfer_families_v0.csv")
    return int(df[(df.source == 0) & df.target.isin(list(TARGETS[family]))].certified.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("queue", nargs="?")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--tally", nargs="*")
    args = ap.parse_args()
    if args.tally:
        for d in args.tally:
            print(d, tally(Path(d)))
        sys.exit()
    queue = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    running = []
    while queue or running:
        while queue and len(running) < args.jobs:
            item = queue.pop(0)
            d = Path(item["dir"])
            if (d / "transfer_families_v0.csv").exists():
                continue
            write(d, item["family"], [tuple(r) for r in item["rules"]])
            running.append((d, start(d, item["family"]), time.time()))
        for entry in list(running):
            d, proc, t0 = entry
            if proc.poll() is not None:
                running.remove(entry)
                ok = (d / "transfer_families_v0.csv").exists()
                print(f"{d}: {tally(d) if ok else 'FAILED'}/20 ({time.time() - t0:.0f} s)", flush=True)
        time.sleep(5)
