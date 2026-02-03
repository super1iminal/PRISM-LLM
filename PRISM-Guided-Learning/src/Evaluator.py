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
from planners.UniformPlanner import BaselinePlanner
from planners.RLCounterfactual import LTLGuidedQLearningWithObstacle
from planners.FeedbackLLMPlanner import FeedbackLLMPlanner
from config.Settings import EVAL_PATH, RESULTS_PATH


class EvalModel(Enum):
    # Vanilla LLM planners
    LLM_VANILLA_GPT5_NANO = 1
    LLM_VANILLA_GPT5_MINI = 2
    LLM_VANILLA_GEMINI_FLASH = 3
    # Feedback LLM planners
    LLM_FEEDBACK_GPT5_NANO = 4
    LLM_FEEDBACK_GPT5_MINI = 5
    LLM_FEEDBACK_GEMINI_FLASH = 6
    # Other planners
    RL = 7
    RANDOM = 8
    UNIFORM = 9


def main():
    """Main entry point - runs evaluation and saves results to CSV."""
    logger = setup_logger("eval", console_output=False, log_path=EVAL_PATH)

    dataloader = DataLoader("PRISM-Guided-Learning/data/grid_20_samples.csv")
    dataloader.load_data()

    models = [EvalModel.LLM_FEEDBACK_GPT5_NANO]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_PATH, f"{len(dataloader.data)}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

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
        model = get_model(model_type)

        start_time = time()
        results = model.evaluate(dataloader, parallel=True)
        end_time = time()
        delta_time = end_time - start_time

        logger.info(f"Results for {model_type.name} (in {delta_time:.2f} seconds):")
        for idx, result in enumerate(results):
            logger.info(f"  Gridworld {idx+1}: LTL Score = {result['LTL_Score']:.4f}")
        logger.info("\n" * 2)

        df = save_results(results, dataloader, model_type.name, run_dir)
        model_dfs[model_type.name] = df

    return model_dfs


def get_model(model_type: EvalModel):
    """Factory function to create model instances."""
    # Vanilla LLM planners
    if model_type == EvalModel.LLM_VANILLA_GPT5_NANO:
        model_name = "gpt-5-nano-2025-08-07"
        model = ChatOpenAI(model_name=model_name, temperature=1).with_structured_output(ActionPolicy)
        return VanillaLLMPlanner(model=model, model_name=model_name)
    elif model_type == EvalModel.LLM_VANILLA_GPT5_MINI:
        model_name = "gpt-5-mini-2025-08-07"
        model = ChatOpenAI(model_name=model_name, temperature=1).with_structured_output(ActionPolicy)
        return VanillaLLMPlanner(model=model, model_name=model_name)
    elif model_type == EvalModel.LLM_VANILLA_GEMINI_FLASH:
        model_name = "gemini-1.5-flash"
        model = ChatGoogleGenerativeAI(model=model_name, temperature=1).with_structured_output(ActionPolicy)
        return VanillaLLMPlanner(model=model, model_name=model_name)
    # Feedback LLM planners
    elif model_type == EvalModel.LLM_FEEDBACK_GPT5_NANO:
        model_name = "gpt-5-nano-2025-08-07"
        model = ChatOpenAI(model_name=model_name, temperature=1).with_structured_output(ActionPolicy)
        return FeedbackLLMPlanner(model=model, model_name=model_name, max_attempts=10)
    elif model_type == EvalModel.LLM_FEEDBACK_GPT5_MINI:
        model_name = "gpt-5-mini-2025-08-07"
        model = ChatOpenAI(model_name=model_name, temperature=1).with_structured_output(ActionPolicy)
        return FeedbackLLMPlanner(model=model, model_name=model_name, max_attempts=10)
    elif model_type == EvalModel.LLM_FEEDBACK_GEMINI_FLASH:
        model_name = "gemini-1.5-flash"
        model = ChatGoogleGenerativeAI(model=model_name, temperature=1).with_structured_output(ActionPolicy)
        return FeedbackLLMPlanner(model=model, model_name=model_name, max_attempts=10)
    # Other planners
    elif model_type == EvalModel.RL:
        return LTLGuidedQLearningWithObstacle()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_results(
    results: List[Dict],
    dataloader: DataLoader,
    model_name: str,
    run_dir: str
) -> pd.DataFrame:
    """
    Save evaluation results to CSV.

    Args:
        results: List of result dictionaries from model evaluation
        dataloader: DataLoader with grid configurations
        model_name: Name of the model for the output file
        run_dir: Directory to save the CSV

    Returns:
        DataFrame containing the results
    """
    rows = []

    for idx, result in enumerate(results):
        gridworld = dataloader.data[idx][0]
        expected_steps = dataloader.data[idx][1]

        rows.append({
            'ltl_score': result['LTL_Score'],
            'prism_probabilities': result['Prism_Probabilities'],
            'evaluation_time': result['Evaluation_Time'],
            'size': gridworld.size,
            'goals': len(gridworld.goals),
            'obstacles': len(gridworld.static_obstacles),
            'complexity': expected_steps
        })

    df = pd.DataFrame(rows)
    output_path = os.path.join(run_dir, f"results_{model_name}.csv")
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    main()
