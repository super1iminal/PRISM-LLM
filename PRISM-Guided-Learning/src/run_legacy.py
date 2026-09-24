"""Run the legacy (per-state policy, FeedbackSimplified) planner on a gridworld dataset.

Usage: python src/run_legacy.py [--data grid_20_balanced.csv] [--workers 2] [--max-attempts 5]
"""
import argparse
import datetime
import json
import os
from time import time
from typing import Dict, List

import pandas as pd

from core.llm import OllamaLLM
from legacy.data_loader import DataLoader
from legacy.planner import FeedbackSimplifiedLLMPlanner
from legacy.prompting import ActionPolicy
from logging_utils import setup_logger
from settings import DOMAINS_PATH, OLLAMA_MODEL, RESULTS_PATH

APPROACH_NAME = "LEGACY_FEEDBACK_SIMPLIFIED"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="grid_20_balanced.csv")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--out", default=None, help="Output directory (default: timestamped under out/results)")
    args = parser.parse_args()

    dataloader = DataLoader(str(DOMAINS_PATH / "gridworld" / "data" / args.data))
    dataloader.load_data()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    run_dir = args.out or os.path.join(RESULTS_PATH, f"legacy_{len(dataloader.data)}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logger("eval", run_dir=run_dir, include_timestamp=False)

    model = OllamaLLM(ActionPolicy)
    planner = FeedbackSimplifiedLLMPlanner(model=model, model_name=OLLAMA_MODEL.replace(":", "_"),
                                           max_attempts=args.max_attempts)
    start_time = time()
    results = planner.evaluate(dataloader, max_workers=args.workers, run_dir=run_dir)
    logger.info(f"{APPROACH_NAME} finished in {time() - start_time:.2f} seconds")
    for idx, result in enumerate(results):
        logger.info(f"  Gridworld {idx+1}: LTL Score = {result['LTL_Score']:.4f}, success = {result['Success']}")

    save_results(results, dataloader, APPROACH_NAME, run_dir)
    save_outputs(results, run_dir)
    logger.info(f"Results saved to: {run_dir}")
    print(run_dir)


def results_to_multiindex_df(results: List[Dict], dataloader: DataLoader) -> pd.DataFrame:
    """Convert results to a DataFrame indexed by (sample_id, iteration)."""
    rows = []

    for sample_idx, result in enumerate(results):
        gridworld = dataloader.data[sample_idx][0]
        expected_steps = dataloader.data[sample_idx][1]
        num_iterations = result.get("Iterations_Used", 1)

        def at(key, i, default):
            values = result.get(key, [])
            return values[i] if i < len(values) else default

        for iter_idx in range(num_iterations):
            prism_probs = at("Iteration_Prism_Probs", iter_idx, {})
            if not prism_probs and iter_idx == 0:
                prism_probs = result.get("Prism_Probabilities", {})

            rows.append({
                "sample_id": sample_idx,
                "iteration": iter_idx + 1,
                "size": gridworld.size,
                "goals": len(gridworld.goals),
                "obstacles": len(gridworld.static_obstacles),
                "complexity": expected_steps,
                "iteration_time": at("Iteration_Times", iter_idx, 0.0),
                "prism_time": at("Iteration_PRISM_Times", iter_idx, 0.0),
                "llm_time": at("Iteration_LLM_Times", iter_idx, 0.0),
                "mistakes": at("Iteration_Mistakes", iter_idx, 0),
                "cost": at("Iteration_Costs", iter_idx, 0.0),
                **{f"prob_{k}": v for k, v in prism_probs.items()},
                "is_final": (iter_idx + 1 == num_iterations),
                "final_ltl_score": result.get("LTL_Score", 0.0),
                "success": result.get("Success", False),
                "llm_calls": result.get("LLM_Calls", 0),
                "llm_output_tokens": result.get("LLM_Output_Tokens", 0),
                "llm_prompt_tokens": result.get("LLM_Prompt_Tokens", 0),
                "total_time": result.get("Evaluation_Time", 0.0),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index(["sample_id", "iteration"])
    return df


def save_results(results: List[Dict], dataloader: DataLoader, name: str, run_dir: str) -> pd.DataFrame:
    df = results_to_multiindex_df(results, dataloader)
    df.to_parquet(os.path.join(run_dir, f"{name}_results.parquet"))
    return df


def save_outputs(results: List[Dict], run_dir: str) -> None:
    """Save raw LLM outputs and every verified policy (JSON-safe), one file per sample."""
    out_dir = os.path.join(run_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    for idx, result in enumerate(results):
        record = {
            "sample_id": idx,
            "success": result.get("Success", False),
            "error": result.get("error"),
            "final_prism_probs": result.get("Prism_Probabilities", {}),  # of the kept (best) policy
            "iteration_prism_probs": result.get("Iteration_Prism_Probs", []),
            # Each policy is a list of [state, action] with state = [x, y, g1, ..., gN]
            "iteration_policies": [
                [[list(state), action] for state, action in policy.items()]
                for policy in result.get("Iteration_Policies", [])
            ],
            "llm_raw_outputs": result.get("LLM_Raw_Outputs", []),
        }
        with open(os.path.join(out_dir, f"sample_{idx:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1)


if __name__ == "__main__":
    main()
