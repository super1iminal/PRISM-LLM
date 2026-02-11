import datetime
from enum import Enum
import os
from time import time
from typing import Dict, List

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.DataLoader import DataLoader
from utils.Logging import setup_logger
from utils.LLMPrompting import ActionPolicy
from planners.VanillaLLMPlanner import VanillaLLMPlanner
from planners.VanillaPlusLLMPlanner import VanillaPlusLLMPlanner
from planners.UniformPlanner import BaselinePlanner
from planners.RLCounterfactual import LTLGuidedQLearningWithObstacle
from planners.FeedbackLLMPlanner import FeedbackLLMPlanner
from planners.FeedbackMinusLLMPlanner import FeedbackMinusLLMPlanner
from config.Settings import RESULTS_PATH


class EvalModel(Enum):
    # Vanilla LLM planners (single iteration, no feedback)
    LLM_VANILLA_GPT5_NANO = 1
    LLM_VANILLA_GPT5_MINI = 2
    LLM_VANILLA_GEMINI_PRO = 3
    # Feedback LLM planners (multiple iterations, full feedback)
    LLM_FEEDBACK_GPT5_NANO = 4
    LLM_FEEDBACK_GPT5_MINI = 5
    LLM_FEEDBACK_GEMINI_PRO = 6
    # Other planners
    RL = 7
    RANDOM = 8
    UNIFORM = 9
    # Vanilla Plus LLM planners (multiple iterations, no feedback - regenerate from scratch)
    LLM_VANILLA_PLUS_GPT5_NANO = 10
    LLM_VANILLA_PLUS_GPT5_MINI = 11
    LLM_VANILLA_PLUS_GEMINI_PRO = 12
    # Feedback Minus LLM planners (multiple iterations, probabilities only - no NL problems)
    LLM_FEEDBACK_MINUS_GPT5_NANO = 20
    LLM_FEEDBACK_MINUS_GPT5_MINI = 21
    LLM_FEEDBACK_MINUS_GEMINI_PRO = 22


def main():
    """Main entry point - runs evaluation and saves results to CSV."""
    dataloader = DataLoader("PRISM-Guided-Learning/data/grid_20_samples.csv")
    dataloader.load_data()

    models = [
        # EvalModel.RL, 
        # EvalModel.LLM_VANILLA_GPT5_NANO, 
        # EvalModel.LLM_VANILLA_GPT5_MINI,
        # EvalModel.LLM_VANILLA_GEMINI_PRO,
        # EvalModel.LLM_VANILLA_PLUS_GPT5_NANO,
        # EvalModel.LLM_VANILLA_PLUS_GPT5_MINI,
        # EvalModel.LLM_VANILLA_PLUS_GEMINI_PRO,
        # EvalModel.LLM_FEEDBACK_MINUS_GPT5_NANO,
        EvalModel.LLM_FEEDBACK_MINUS_GPT5_MINI,
        # EvalModel.LLM_FEEDBACK_MINUS_GEMINI_PRO,
        # EvalModel.LLM_FEEDBACK_GPT5_NANO,         
        EvalModel.LLM_FEEDBACK_GPT5_MINI,
        #EvalModel.LLM_FEEDBACK_GEMINI_PRO 
    ]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_PATH, f"{len(dataloader.data)}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger("eval", console_output=False, run_dir=run_dir, include_timestamp=False)

    model_dfs = evaluate_models(models, dataloader, run_dir, logger)

    logger.info(f"Results saved to: {run_dir}")
    return model_dfs, run_dir


def evaluate_models(
    models: List[EvalModel],
    dataloader: DataLoader,
    run_dir: str,
    logger
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate multiple models on the given data.

    Args:
        models: List of model types to evaluate
        dataloader: DataLoader with grid configurations
        run_dir: Directory to save results
        logger: Logger instance

    Returns:
        Dictionary mapping model names to their results DataFrames
    """
    model_dfs = {}

    for model_type in models:
        try:
            model = get_model(model_type)

            start_time = time()
            results = model.evaluate(dataloader, max_workers=20, run_dir=run_dir)
            end_time = time()
            delta_time = end_time - start_time

            logger.info(f"Results for {model_type.name} (in {delta_time:.2f} seconds):")
            for idx, result in enumerate(results):
                logger.info(f"  Gridworld {idx+1}: LTL Score = {result['LTL_Score']:.4f}")
            logger.info("\n" * 2)

            df = save_results(results, dataloader, model_type.name, run_dir)
            model_dfs[model_type.name] = df

        except Exception as e:
            logger.error(f"Model {model_type.name} FAILED: {type(e).__name__}: {e}")
            logger.exception("Full traceback:")
            # Continue to next model

    return model_dfs


def get_model(model_type: EvalModel):
    """Factory function to create model instances."""
    # Vanilla LLM planners (single iteration, no feedback)
    if model_type == EvalModel.LLM_VANILLA_GPT5_NANO:
        model_name = "gpt-5-nano-2025-08-07"
        model = ChatOpenAI(model_name=model_name, temperature=1).with_structured_output(ActionPolicy)
        return VanillaLLMPlanner(model=model, model_name=model_name)
    elif model_type == EvalModel.LLM_VANILLA_GPT5_MINI:
        model_name = "gpt-5-mini-2025-08-07"
        model = ChatOpenAI(model_name=model_name, temperature=1).with_structured_output(ActionPolicy)
        return VanillaLLMPlanner(model=model, model_name=model_name)
    elif model_type == EvalModel.LLM_VANILLA_GEMINI_PRO:
        model_name = "gemini-2.5-pro"
        model = ChatGoogleGenerativeAI(model=model_name, temperature=1).with_structured_output(ActionPolicy)
        return VanillaLLMPlanner(model=model, model_name=model_name)
    # Vanilla Plus LLM planners (multiple iterations, no feedback - regenerate from scratch)
    elif model_type == EvalModel.LLM_VANILLA_PLUS_GPT5_NANO:
        model_name = "gpt-5-nano-2025-08-07"
        model = ChatOpenAI(model_name=model_name, temperature=1).with_structured_output(ActionPolicy)
        return VanillaPlusLLMPlanner(model=model, model_name=model_name, max_attempts=3)
    elif model_type == EvalModel.LLM_VANILLA_PLUS_GPT5_MINI:
        model_name = "gpt-5-mini-2025-08-07"
        model = ChatOpenAI(model_name=model_name, temperature=1).with_structured_output(ActionPolicy)
        return VanillaPlusLLMPlanner(model=model, model_name=model_name, max_attempts=3)
    elif model_type == EvalModel.LLM_VANILLA_PLUS_GEMINI_PRO:
        model_name = "gemini-2.5-pro"
        model = ChatGoogleGenerativeAI(model=model_name, temperature=1).with_structured_output(ActionPolicy)
        return VanillaPlusLLMPlanner(model=model, model_name=model_name, max_attempts=3)
    # Feedback Minus LLM planners (multiple iterations, probabilities only - no NL problems)
    elif model_type == EvalModel.LLM_FEEDBACK_MINUS_GPT5_NANO:
        model_name = "gpt-5-nano-2025-08-07"
        model = ChatOpenAI(model_name=model_name, temperature=1).with_structured_output(ActionPolicy)
        return FeedbackMinusLLMPlanner(model=model, model_name=model_name, max_attempts=3)
    elif model_type == EvalModel.LLM_FEEDBACK_MINUS_GPT5_MINI:
        model_name = "gpt-5-mini-2025-08-07"
        model = ChatOpenAI(model_name=model_name, temperature=1).with_structured_output(ActionPolicy)
        return FeedbackMinusLLMPlanner(model=model, model_name=model_name, max_attempts=3)
    elif model_type == EvalModel.LLM_FEEDBACK_MINUS_GEMINI_PRO:
        model_name = "gemini-2.5-pro"
        model = ChatGoogleGenerativeAI(model=model_name, temperature=1).with_structured_output(ActionPolicy)
        return FeedbackMinusLLMPlanner(model=model, model_name=model_name, max_attempts=3)
    # Feedback LLM planners (multiple iterations, full feedback)
    elif model_type == EvalModel.LLM_FEEDBACK_GPT5_NANO:
        model_name = "gpt-5-nano-2025-08-07"
        model = ChatOpenAI(model_name=model_name, temperature=1).with_structured_output(ActionPolicy)
        return FeedbackLLMPlanner(model=model, model_name=model_name, max_attempts=3)
    elif model_type == EvalModel.LLM_FEEDBACK_GPT5_MINI:
        model_name = "gpt-5-mini-2025-08-07"
        model = ChatOpenAI(model_name=model_name, temperature=1).with_structured_output(ActionPolicy)
        return FeedbackLLMPlanner(model=model, model_name=model_name, max_attempts=3)
    elif model_type == EvalModel.LLM_FEEDBACK_GEMINI_PRO:
        model_name = "gemini-2.5-pro"
        model = ChatGoogleGenerativeAI(model=model_name, temperature=1).with_structured_output(ActionPolicy)
        return FeedbackLLMPlanner(model=model, model_name=model_name, max_attempts=3)
    # Other planners
    elif model_type == EvalModel.RL:
        return LTLGuidedQLearningWithObstacle()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def results_to_multiindex_df(
    results: List[Dict],
    dataloader: DataLoader
) -> pd.DataFrame:
    """
    Convert results to MultiIndex DataFrame with (sample_id, iteration) index.

    Args:
        results: List of result dictionaries from model evaluation
        dataloader: DataLoader with grid configurations

    Returns:
        DataFrame with MultiIndex (sample_id, iteration) containing per-iteration metrics
    """
    rows = []

    for sample_idx, result in enumerate(results):
        gridworld = dataloader.data[sample_idx][0]
        expected_steps = dataloader.data[sample_idx][1]
        num_iterations = result.get("Iterations_Used", 1)

        for iter_idx in range(num_iterations):
            # Get per-iteration data with safe indexing
            iteration_prism_probs = result.get("Iteration_Prism_Probs", [{}])
            prism_probs = iteration_prism_probs[iter_idx] if iter_idx < len(iteration_prism_probs) else {}

            # Fallback: if iteration prism probs are empty, use final Prism_Probabilities
            if not prism_probs and iter_idx == 0:
                prism_probs = result.get("Prism_Probabilities", {})

            iteration_times = result.get("Iteration_Times", [0.0])
            iteration_time = iteration_times[iter_idx] if iter_idx < len(iteration_times) else 0.0

            iteration_prism_times = result.get("Iteration_PRISM_Times", [0.0])
            prism_time = iteration_prism_times[iter_idx] if iter_idx < len(iteration_prism_times) else 0.0

            iteration_llm_times = result.get("Iteration_LLM_Times", [0.0])
            llm_time = iteration_llm_times[iter_idx] if iter_idx < len(iteration_llm_times) else 0.0

            iteration_mistakes = result.get("Iteration_Mistakes", [0])
            mistakes = iteration_mistakes[iter_idx] if iter_idx < len(iteration_mistakes) else 0

            iteration_costs = result.get("Iteration_Costs", [0.0])
            cost = iteration_costs[iter_idx] if iter_idx < len(iteration_costs) else 0.0

            row = {
                "sample_id": sample_idx,
                "iteration": iter_idx + 1,
                # Grid metadata
                "size": gridworld.size,
                "goals": len(gridworld.goals),
                "obstacles": len(gridworld.static_obstacles),
                "complexity": expected_steps,
                # Per-iteration metrics
                "iteration_time": iteration_time,
                "prism_time": prism_time,
                "llm_time": llm_time,
                "mistakes": mistakes,
                "cost": cost,
                # Flattened probabilities
                **{f"prob_{k}": v for k, v in prism_probs.items()},
                # Final iteration flag
                "is_final": (iter_idx + 1 == num_iterations),
                "final_ltl_score": result.get("LTL_Score", 0.0),
                "success": result.get("Success", False),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index(["sample_id", "iteration"])
    return df


def save_results(
    results: List[Dict],
    dataloader: DataLoader,
    model_name: str,
    run_dir: str
) -> pd.DataFrame:
    """
    Save evaluation results to Parquet with MultiIndex.

    Args:
        results: List of result dictionaries from model evaluation
        dataloader: DataLoader with grid configurations
        model_name: Name of the model for the output file
        run_dir: Directory to save the Parquet file

    Returns:
        DataFrame containing the results with MultiIndex (sample_id, iteration)
    """
    df = results_to_multiindex_df(results, dataloader)
    output_path = os.path.join(run_dir, f"{model_name}_results.parquet")
    df.to_parquet(output_path)

    return df


if __name__ == "__main__":
    main()
