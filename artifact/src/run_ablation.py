"""Run named conditions (configs/run/conditions/<name>.yaml) for several seeds.

Usage: python src/run_ablation.py B2 R1 R2 [--seeds 1 2 | --unseeded N] [--set section.key=value ...] [--dry-run]

The seeds default to configs/ablation/default.yaml; `--set` overrides the run config of every run.
Each (condition, seed) goes to out/results/<core or ablations>/<condition>/seed_<k>/ (core: the runs of the
comparison experiments in configs/experiments/, experiments.base.core_names) with its resolved config.json; `--name`
uses another directory name than the condition's (an experiment cell's: the condition plus overrides). `--unseeded N` instead runs N repeats with no seed (for models whose endpoints take
none, e.g. Claude on OpenRouter) into seed_none_1/ .. seed_none_N/; `--repeat K ...` runs only those
repeats. Finished runs (results parquet present) are skipped, so an interrupted batch can be restarted
with the same command. Runs execute one after another, never concurrently (src/run_search.py runs
several of these processes in parallel).
"""
import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config import CONFIG_DIR, load_config, load_file
from experiments.store import ABLATIONS_PATH, CORE_PATH
from results_io import SYMBOLIC_RESULTS

DEFAULT_CONFIG = CONFIG_DIR / "ablation" / "default.yaml"


@dataclass
class AblationConfig:
    """configs/ablation/default.yaml documents each key."""
    seeds: List[int]


def core_conditions() -> List[str]:
    """The run names of the method's main runs: the cells of the comparison experiments (RQ1)."""
    from experiments.base import core_names
    return core_names()


def results_root(condition: str) -> Path:
    """Where a condition's runs go: out/results/core/ for core conditions, out/results/ablations/ otherwise."""
    return CORE_PATH if condition in core_conditions() else ABLATIONS_PATH


def existing_root(condition: str) -> Path:
    """Where a condition's runs are now: its `results_root`, or the other root while a run that started before the
    condition was listed in (or removed from) a comparison experiment has not been moved yet."""
    root = results_root(condition)
    other = ABLATIONS_PATH if root == CORE_PATH else CORE_PATH
    return other if not (root / condition).exists() and (other / condition).exists() else root


def run_dir_for(condition: str, seed: str, root: Optional[Path] = None) -> Path:
    return (root or results_root(condition)) / condition / f"seed_{seed}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("conditions", nargs="+", help="Condition names (files in configs/run/conditions/)")
    seeding = parser.add_mutually_exclusive_group()
    seeding.add_argument("--seeds", type=int, nargs="+", default=None, help="Default: the ablation config's seeds")
    seeding.add_argument("--unseeded", type=int, metavar="N", default=None,
                         help="N repeats without a seed, in seed_none_1/ .. seed_none_N/")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Ablation config file")
    parser.add_argument("--repeat", type=int, nargs="+", default=None,
                        help="With --unseeded N: only these repeats (1..N)")
    parser.add_argument("--set", action="append", default=[], help="Run-config override section.key=value")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would run")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Results root (default: out/results/core for the comparison experiments' runs, "
                             "out/results/ablations otherwise; use out/results/smoke for smoke tests)")
    parser.add_argument("--name", default=None, help="Directory name instead of the condition's (one condition)")
    args = parser.parse_args()
    if args.repeat and not args.unseeded:
        parser.error("--repeat needs --unseeded")
    if args.name and len(args.conditions) > 1:
        parser.error("--name needs a single condition")
    if args.unseeded:
        runs = [(f"none_{k}", []) for k in range(1, args.unseeded + 1) if not args.repeat or k in args.repeat]
    else:
        runs = [(str(s), [f"llm.seed={s}"]) for s in args.seeds or load_file(args.config, AblationConfig).seeds]

    # Resolve every config first, so a typo fails before any GPU time is spent
    plan = []
    for condition in args.conditions:
        for label, seeding_overrides in runs:
            cfg = load_config(condition, list(args.set) + seeding_overrides)
            plan.append((condition, label, cfg, run_dir_for(args.name or condition, label, args.out_root)))

    for condition, seed, cfg, run_dir in plan:
        done = (run_dir / SYMBOLIC_RESULTS).exists()
        status = "done, skipping" if done else ("would run" if args.dry_run else "running")
        print(f"[{condition} seed {seed}] {status} -> {run_dir}", flush=True)
        if done or args.dry_run:
            continue
        if run_dir.exists():
            shutil.rmtree(run_dir)   # a previous attempt that did not finish
        from run_symbolic import run
        run(cfg, str(run_dir))


if __name__ == "__main__":
    main()
