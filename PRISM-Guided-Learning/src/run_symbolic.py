"""Run the symbolic-policy planner on a domain dataset.

Usage: python src/run_symbolic.py [--condition B2] [--set section.key=value ...] [--out DIR]
       (shortcuts: --domain, --data, --workers, --max-attempts, --limit)
Settings come from configs/ (see docs/config.md); the resolved config is saved as <run>/config.json.
"""
import argparse
import datetime
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
from typing import Any, Dict, List

import pandas as pd

from config import Config, load_config
from core.domain import Instance, load_domain
from core.llm import OllamaLLM
from core.planner import SymbolicPlanner
from logging_utils import close_logger, setup_logger
from results_io import SYMBOLIC_RESULTS
from settings import RESULTS_PATH

def cli_overrides(args) -> List[str]:
    """Map the old shortcut flags onto config overrides."""
    out = list(args.set or [])
    for flag, key in (("domain", "domain.name"), ("data", "domain.dataset"), ("workers", "run.workers"),
                      ("max_attempts", "planner.max_rounds"), ("limit", "run.limit")):
        if getattr(args, flag, None) is not None:
            out.append(f"{key}={getattr(args, flag)}")
    return out


def add_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--condition", default=None, help="Named condition: configs/run/conditions/<name>.yaml")
    parser.add_argument("--set", action="append", help="Config override section.key=value (repeatable)")
    parser.add_argument("--domain")
    parser.add_argument("--data")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default=None, help="Output directory (default: timestamped under out/results)")


def main():
    parser = argparse.ArgumentParser()
    add_cli(parser)
    args = parser.parse_args()
    cfg = load_config(args.condition, cli_overrides(args))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    run_dir = args.out or os.path.join(RESULTS_PATH, f"symbolic_{cfg.domain.name}_{timestamp}")
    print(run(cfg, run_dir))


def run(cfg: Config, run_dir: str) -> str:
    """Solve every instance of `cfg.domain` with the symbolic planner; write results to `run_dir`."""
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=1)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = domain.load_instances(cfg.domain.dataset)[:cfg.run.limit]
    main_logger = setup_logger("main", run_dir=run_dir, include_timestamp=False)

    llm = OllamaLLM(cfg.llm)
    planner = SymbolicPlanner(domain, llm, cfg)

    def solve(instance: Instance) -> Dict[str, Any]:
        logger = setup_logger(f"worker_{instance.id}", run_dir=run_dir, include_timestamp=False)
        llm.reset_usage()
        start = time()
        try:
            result = planner.solve(instance, logger)
        except Exception as e:
            logger.error(traceback.format_exc())
            result = {"success": False, "error": f"{type(e).__name__}: {e}", "iterations": []}
        finally:
            close_logger(logger)
        result["total_time"] = time() - start
        result["instance"] = instance.id
        return result

    results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=cfg.run.workers) as pool:
        futures = {pool.submit(solve, inst): inst for inst in instances}
        for future in as_completed(futures):
            inst = futures[future]
            results[inst.id] = future.result()
            r = results[inst.id]
            main_logger.info(f"Instance {inst.id}: success={r['success']} iterations={len(r['iterations'])} "
                             f"time={r['total_time']:.1f}s {r.get('error') or ''}")

    ordered = [results[inst.id] for inst in instances]
    save_results(ordered, instances, run_dir)
    main_logger.info(f"Results saved to: {run_dir}")
    close_logger(main_logger)
    return run_dir


def results_to_df(results: List[Dict[str, Any]], instances: List[Instance]) -> pd.DataFrame:
    """One row per (sample_id, iteration)."""
    rows = []
    for sample_idx, (result, inst) in enumerate(zip(results, instances)):
        iterations = result.get("iterations", [])
        totals = {
            "llm_calls": sum(it["llm_calls"] for it in iterations),
            "llm_output_tokens": sum(it["llm_output_tokens"] for it in iterations),
            "llm_prompt_tokens": sum(it["llm_prompt_tokens"] for it in iterations),
        }
        for it in iterations:
            rows.append({
                "sample_id": sample_idx,
                "iteration": it["iteration"],
                "instance": inst.id,
                "mode": it["mode"],
                "num_rules": it["num_rules"],
                "reachable_situations": it["reachable_situations"],
                "uncovered_situations": it["uncovered_situations"],
                "iteration_time": it.get("iteration_time", 0.0),
                "prism_time": it["prism_time"],
                "joint_time": it.get("joint_time", 0.0),
                "llm_time": it["llm_time"],
                "iter_llm_calls": it["llm_calls"],
                "iter_llm_output_tokens": it["llm_output_tokens"],
                "iter_llm_prompt_tokens": it["llm_prompt_tokens"],
                "invalid_answers": len(it["invalid_answers"]),
                "improved": it.get("improved"),
                "kept_joint_feasible": it.get("kept_joint_feasible"),
                "branch_disagreement": it.get("branch_disagreement", False),
                **{f"prob_best_{k}": p for k, p in it["best"].items()},
                **{f"prob_worst_{k}": p for k, p in it["worst"].items()},
                "best_case_success_iter": it["best_case_success"],
                "worst_case_success_iter": it["worst_case_success"],
                "is_final": it["iteration"] == len(iterations),
                "success": result["success"],
                "best_case_success": result.get("best_case_success", False),
                **{f"final_best_{k}": p for k, p in result.get("final_best", {}).items()},
                **{f"final_worst_{k}": p for k, p in result.get("final_worst", {}).items()},
                **{f"optimum_{k}": p for k, p in result.get("optimum", {}).items()},
                "final_num_rules": len(result.get("final_rules", [])),
                **totals,
                "total_time": result.get("total_time", 0.0),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index(["sample_id", "iteration"])
    return df


def save_results(results: List[Dict[str, Any]], instances: List[Instance], run_dir: str) -> None:
    results_to_df(results, instances).to_parquet(os.path.join(run_dir, SYMBOLIC_RESULTS))
    out_dir = os.path.join(run_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    for idx, result in enumerate(results):
        with open(os.path.join(out_dir, f"sample_{idx:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, default=str)


if __name__ == "__main__":
    main()
