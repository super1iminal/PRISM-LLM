"""Plan, run and report experiments (configs/experiments/*.yaml): RQ1 comparisons and RQ2 ablations over one store of
runs (src/experiments/).

Usage (from artifact/; for LLM runs, the OpenRouter key in the environment: `set -a && . ../.env && set +a`):
    python src/run_experiment.py configs/experiments/<name>.yaml ... [--run] [--check] [--baselines]
        [--max-parallel 6] [--transfer-workers 2] [--index]
    python src/run_experiment.py --index
`--index`: also write out/results/experiments/README.md, every experiment's cells with their runs, pending and
missing repeats (and the baselines' certification), and the RQ2 overview figure rq2_overview.png.
Without flags: print each experiment's plan (runs found, and under which names; pending; missing) and write its
report from what exists, to out/results/experiments/<name>/REPORT.md. `--check`: also certify existing runs that
miss a transfer CSV. `--run`: also launch missing repeats (and check them). `--baselines`: also fit and certify
missing baselines (comparison experiments). Several experiments share one job pool, and a configuration two of them
need runs once.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.base import experiment, jobs
from experiments.index import write_index
from experiments.store import RunStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiments", nargs="*", type=Path, help="configs/experiments/<name>.yaml")
    parser.add_argument("--index", action="store_true",
                        help="Write out/results/experiments/README.md: every experiment's cells and their status")
    parser.add_argument("--run", action="store_true", help="Launch missing repeats (LLM runs)")
    parser.add_argument("--check", action="store_true", help="Certify existing runs that miss a transfer CSV")
    parser.add_argument("--baselines", action="store_true", help="Fit and certify missing baselines")
    parser.add_argument("--max-parallel", type=int, default=6, help="Jobs at once (runs, checks, baselines)")
    parser.add_argument("--transfer-workers", type=int, default=2, help="Targets certified at once per check")
    args = parser.parse_args(argv)
    if not args.experiments and not args.index:
        parser.error("name an experiment file, or pass --index")

    store = RunStore()
    exps = [experiment(path, store) for path in args.experiments]
    for exp in exps:
        print(f"== {exp.name}: {exp.title}")
        for line in exp.plan():
            print("  " + line)
    if args.run or args.check or args.baselines:
        todo = jobs(exps, args.run, args.check, args.baselines)
        print(f"{len(todo)} jobs", flush=True)
        with ThreadPoolExecutor(args.max_parallel) as pool:
            futures = [pool.submit(job, args.transfer_workers) for job in todo]
            for future in as_completed(futures):
                try:
                    print(future.result(), flush=True)
                except Exception as e:     # one job's failure must not stop the others
                    print(f"job failed: {type(e).__name__}: {e}", flush=True)
        store.refresh()
        exps = [experiment(path, store) for path in args.experiments]
    for exp in exps:
        print(exp.write())
    if args.index:
        print(write_index(store))


if __name__ == "__main__":
    main()
