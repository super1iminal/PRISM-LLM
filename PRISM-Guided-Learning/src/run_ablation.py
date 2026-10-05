"""Run named conditions (configs/run/conditions/<name>.yaml) for several seeds.

Usage: python src/run_ablation.py B2 R1 R2 [--seeds 1 2] [--set section.key=value ...] [--dry-run]

The seeds default to configs/ablation/default.yaml; `--set` overrides the run config of every run.
Each (condition, seed) goes to out/results/ablations/<condition>/seed_<k>/ with its resolved
config.json. Finished runs (results parquet present) are skipped, so an interrupted batch can be
restarted with the same command. Runs execute one after another, never concurrently.
"""
import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

from config import CONFIG_DIR, load_config, load_file
from settings import RESULTS_PATH

ABLATIONS_PATH = RESULTS_PATH / "ablations"
RESULT_FILES = {"symbolic": "SYMBOLIC_results.parquet", "legacy": "LEGACY_FEEDBACK_SIMPLIFIED_results.parquet"}
DEFAULT_CONFIG = CONFIG_DIR / "ablation" / "default.yaml"


@dataclass
class AblationConfig:
    """configs/ablation/default.yaml documents each key."""
    seeds: List[int]


def run_dir_for(condition: str, seed: int, root: Path = ABLATIONS_PATH) -> Path:
    return root / condition / f"seed_{seed}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("conditions", nargs="+", help="Condition names (files in configs/run/conditions/)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Default: the ablation config's seeds")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Ablation config file")
    parser.add_argument("--set", action="append", default=[], help="Run-config override section.key=value")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would run")
    parser.add_argument("--out-root", type=Path, default=ABLATIONS_PATH,
                        help="Results root (default out/results/ablations; use out/results/smoke for smoke tests)")
    args = parser.parse_args()
    seeds = args.seeds or load_file(args.config, AblationConfig).seeds

    # Resolve every config first, so a typo fails before any GPU time is spent
    plan = []
    for condition in args.conditions:
        for seed in seeds:
            cfg = load_config(condition, list(args.set) + [f"llm.seed={seed}"])
            plan.append((condition, seed, cfg, run_dir_for(condition, seed, args.out_root)))

    for condition, seed, cfg, run_dir in plan:
        done = (run_dir / RESULT_FILES[cfg.approach]).exists()
        status = "done, skipping" if done else ("would run" if args.dry_run else "running")
        print(f"[{condition} seed {seed}] {cfg.approach}: {status} -> {run_dir}", flush=True)
        if done or args.dry_run:
            continue
        if run_dir.exists():
            shutil.rmtree(run_dir)   # a previous attempt that did not finish
        if cfg.approach == "legacy":
            from run_legacy import run
        else:
            from run_symbolic import run
        run(cfg, str(run_dir))


if __name__ == "__main__":
    main()
