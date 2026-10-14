"""Run conditions in parallel, then certify each run's frozen rules on held-out targets (the transfer search).

Usage (from PRISM-Guided-Learning/, with the OpenRouter key in the environment: `set -a && . ../.env && set +a`):
    python src/run_search.py <condition> ... [--unseeded 3] [--targets 12-19] [--data <dataset>]
        [--max-parallel 6] [--transfer-workers 2] [--set section.key=value ...] [--summary CONFIG] [--dry-run]

Each (condition, repeat) is one job, at most `--max-parallel` at once: `run_ablation.py <condition> --unseeded N
--repeat k` (skipped when the run has finished), then `transfer.py` on the run for `--targets` (skipped when its
transfer CSV already covers them). Output goes to out/results/ablations/<condition>/seed_none_<k>.log. Each run
keeps up to `run.workers` PRISM processes busy, so `--max-parallel` also caps PRISM's memory use. At the end the
summary page (viz/transfer_summary.py, `--summary`) is rebuilt, if given.
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import pandas as pd

from config import load_config
from results_io import RESULT_FILES
from run_ablation import ABLATIONS_PATH, run_dir_for
from settings import PROJECT_ROOT
from transfer import parse_ids

PYTHON = sys.executable


def transfer_done(run_dir: Path, dataset: str, targets: List[int]) -> bool:
    """Whether the run's transfer CSV for `dataset` already has every target in `targets`."""
    path = run_dir / f"transfer_{Path(dataset).stem}.csv"
    return path.exists() and set(targets) <= set(pd.read_csv(path).target)


def job(condition: str, repeat: int, args) -> str:
    """Run one repeat of a condition and its transfer check; returns a one-line status."""
    cfg = load_config(condition, args.set)
    run_dir = run_dir_for(condition, f"none_{repeat}")
    log = run_dir.parent / f"{run_dir.name}.log"
    dataset = args.data or cfg.domain.dataset
    targets = parse_ids(args.targets)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as out:
        if not (run_dir / RESULT_FILES[cfg.approach]).exists():
            cmd = [PYTHON, "src/run_ablation.py", condition, "--unseeded", str(args.unseeded), "--repeat", str(repeat)]
            cmd += [x for s in args.set for x in ("--set", s)]
            if subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=out, stderr=subprocess.STDOUT).returncode:
                return f"[{condition} {repeat}] run failed (see {log})"
        if not transfer_done(run_dir, dataset, targets):
            cmd = [PYTHON, "src/transfer.py", str(run_dir), "--targets", args.targets,
                   "--workers", str(args.transfer_workers)] + (["--data", args.data] if args.data else [])
            if subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=out, stderr=subprocess.STDOUT).returncode:
                return f"[{condition} {repeat}] transfer failed (see {log})"
    return f"[{condition} {repeat}] done"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("conditions", nargs="+", help="Condition names (files in configs/run/conditions/)")
    parser.add_argument("--unseeded", type=int, default=3, help="Repeats per condition (seed_none_1..N)")
    parser.add_argument("--targets", default="12-19", help='Held-out target ids for the transfer check')
    parser.add_argument("--data", default=None, help="Target dataset (default: each run's own)")
    parser.add_argument("--max-parallel", type=int, default=6, help="Jobs (runs with their transfer check) at once")
    parser.add_argument("--transfer-workers", type=int, default=2, help="Targets checked at once per transfer check")
    parser.add_argument("--set", action="append", default=[], help="Run-config override section.key=value")
    parser.add_argument("--summary", type=Path, default=None, help="Summary config to rebuild at the end")
    parser.add_argument("--dry-run", action="store_true", help="Only print the jobs")
    args = parser.parse_args(argv)

    for condition in args.conditions:
        load_config(condition, args.set)       # fail on a typo before anything runs
    jobs = [(c, k) for c in args.conditions for k in range(1, args.unseeded + 1)]
    for c, k in jobs:
        print(f"[{c} {k}] -> {run_dir_for(c, f'none_{k}', ABLATIONS_PATH)}", flush=True)
    if args.dry_run:
        return
    with ThreadPoolExecutor(args.max_parallel) as pool:
        for future in as_completed([pool.submit(job, c, k, args) for c, k in jobs]):
            print(future.result(), flush=True)
    if args.summary:
        subprocess.run([PYTHON, "viz/transfer_summary.py", str(args.summary)], cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
