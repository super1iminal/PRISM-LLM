from enum import Enum
from time import time
from utils.DataLoader import DataLoader
from utils.Logging import setup_logger
from planners.VanillaLLMPlanner import VanillaLLMPlanner
from planners.UniformPlanner import BaselinePlanner
from planners.RLCounterfactual import LTLGuidedQLearningWithObstacle
from planners.FeedbackLLMPlanner import FeedbackLLMPlanner


class EvalModel(Enum):
    LLM_VANILLA = 1
    LLM_FEEDBACK = 2
    RL = 3
    RANDOM = 4
    UNIFORM = 5

def main():
    logger = setup_logger("eval/eval") 
    
    
    dataloader = DataLoader("PRISM-Guided-Learning/data/grid_20_samples.csv")
    dataloader.load_data()
    
    models = [EvalModel.RL, EvalModel.LLM_VANILLA, EvalModel.LLM_FEEDBACK]
    
    for modelType in models:
        model = get_model(modelType)
        
        start_time = time()
        results = model.evaluate(dataloader, parallel = True)
        end_time = time()
        delta_time = end_time - start_time
        logger.info(f"Results for {modelType.name} (in {delta_time:.2f} seconds):")
        for idx, result in enumerate(results):
            logger.info(f"  Gridworld {idx+1}: LTL Score = {result['LTL_Score']:.4f}")
        logger.info("\n"*2)
    

def get_model(model_type: EvalModel):
    if model_type == EvalModel.LLM_VANILLA:
        return VanillaLLMPlanner()
    elif model_type == EvalModel.LLM_FEEDBACK:
        return FeedbackLLMPlanner(10)
    elif model_type == EvalModel.RL:
        return LTLGuidedQLearningWithObstacle()
    else:
        raise ValueError("Unknown model type")


if __name__ == "__main__":
    main()