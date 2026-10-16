"""Certify run directories with the shared transfer.py (main checkout), one PRISM worker each, at most `--jobs` at a
time, starting a job only while the machine's CPU load is below `--max-load` percent.

    python src/fresh/cert_queue.py queue.json [--jobs 2] [--max-load 80]
queue.json: [{"dir": ..., "config": {...}, "targets": "12-19",
              "samples": [{"rules": [[condition, action], ...], "members": ["0", "4", "8"]}, ...]}, ...]
Prints, per directory, the validation targets each sample certifies (targets that are not its training members).
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
from fresh.certify import MAIN  # noqa: E402


def load_percent() -> float:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average"],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def write(item: dict):
    d = Path(item["dir"])
    (d / "outputs").mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(item["config"], indent=1), encoding="utf-8")
    for i, smp in enumerate(item["samples"]):
        sample = {"final_rules": [{"condition": c, "action": a} for c, a in smp["rules"]],
                  "members": [{"instance": str(m)} for m in smp["members"]]}
        (d / "outputs" / f"sample_{i:03d}.json").write_text(json.dumps(sample, indent=1), encoding="utf-8")


def start(item: dict) -> subprocess.Popen:
    d = Path(item["dir"])
    cmd = [str(MAIN.parent / ".venv" / "Scripts" / "python"), "src/transfer.py", str(d.resolve()),
           "--targets", item["targets"], "--workers", "1"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    return subprocess.Popen(cmd, cwd=MAIN, env=env, stdout=open(d / "transfer.log", "w"), stderr=subprocess.STDOUT)


def tally(d: Path):
    csvs = list(d.glob("transfer_*.csv"))
    if not csvs:
        return None
    df = pd.read_csv(csvs[0])
    held = df[~df.trained.astype(bool)]
    return [int(held[held.source == s].certified.sum()) for s in sorted(df.source.unique())], \
           [int((held.source == s).sum()) for s in sorted(df.source.unique())]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--max-load", type=float, default=80.0)
    args = ap.parse_args()
    queue = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    running = []
    while queue or running:
        if queue and len(running) < args.jobs:
            if (Path(queue[0]["dir"]) / "transfer.log").exists() and tally(Path(queue[0]["dir"])) is not None:
                queue.pop(0)
                continue
            if load_percent() < args.max_load:
                item = queue.pop(0)
                write(item)
                running.append((item, start(item), time.time()))
                continue
        for entry in list(running):
            item, proc, t0 = entry
            if proc.poll() is not None:
                running.remove(entry)
                res = tally(Path(item["dir"]))
                print(f"{item['dir']}: certified {res[0] if res else 'FAILED'} of {res[1] if res else '?'} "
                      f"({time.time() - t0:.0f} s)", flush=True)
        time.sleep(20)
