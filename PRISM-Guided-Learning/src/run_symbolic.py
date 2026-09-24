"""Run the symbolic-policy planner on a domain dataset.

Usage: python src/run_symbolic.py --domain gridworld --data grid_20_balanced.csv [--workers 2]
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

from core.domain import Instance, load_domain
from core.llm import OllamaLLM
from core.planner import PlannerConfig, SymbolicPlanner
from logging_utils import setup_logger
from settings import RESULTS_PATH

APPROACH_NAME = "SYMBOLIC"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="gridworld")
    parser.add_argument("--data", default="grid_20_balanced.csv")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N instances")
    parser.add_argument("--out", default=None, help="Output directory (default: timestamped under out/results)")
    args = parser.parse_args()

    domain = load_domain(args.domain)
    instances = domain.load_instances(args.data)[:args.limit]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    run_dir = args.out or os.path.join(RESULTS_PATH, f"symbolic_{args.domain}_{len(instances)}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    main_logger = setup_logger("main", run_dir=run_dir, include_timestamp=False)

    llm = OllamaLLM()
    planner = SymbolicPlanner(domain, llm, PlannerConfig(max_attempts=args.max_attempts))

    def solve(instance: Instance) -> Dict[str, Any]:
        logger = setup_logger(f"worker_{instance.id}", run_dir=run_dir, include_timestamp=False)
        llm.reset_usage()
        start = time()
        try:
            result = planner.solve(instance, logger)
        except Exception as e:
            logger.error(traceback.format_exc())
            result = {"success": False, "error": f"{type(e).__name__}: {e}", "iterations": []}
        result["total_time"] = time() - start
        result["instance"] = instance.id
        return result

    results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
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
    print(run_dir)


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
                "llm_time": it["llm_time"],
                "iter_llm_calls": it["llm_calls"],
                "iter_llm_output_tokens": it["llm_output_tokens"],
                "invalid_answers": len(it["invalid_answers"]),
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
    results_to_df(results, instances).to_parquet(os.path.join(run_dir, f"{APPROACH_NAME}_results.parquet"))
    out_dir = os.path.join(run_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    for idx, result in enumerate(results):
        with open(os.path.join(out_dir, f"sample_{idx:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, default=str)


if __name__ == "__main__":
    main()
