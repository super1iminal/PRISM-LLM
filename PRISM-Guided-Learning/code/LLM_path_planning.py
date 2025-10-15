from prismUtils.PrismVerifier import PrismVerifier
from prismUtils.SimplifiedVerifier import SimplifiedVerifier
from prismUtils.PrismModelGenerator import PrismModelGenerator
from prismUtils.GridWorldEnv import GridWorldEnv

import logging
import os
import numpy as np
from typing import List, Optional, Dict, Tuple

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from pydantic import BaseModel, Field

import time

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/debug.log"),
                        logging.StreamHandler()
                    ],
                    filemode='w')

logger = logging.getLogger(__name__)


GRID_SIZE = 4

# TODO: make model avoid future goals

PROMPT_TEXT = """You are an expert path planner working on formulating paths that meet formal requirements.

The grid world is {size} x {size}. (0, 0) is in the top left.

Your job is to determine the best policy to reach a goal from any starting state. To do this, you will set the "best move" direction to 100, and the other moves to 0. Index 0 is up, 1 is right, 2 is down, and 3 is left.

There are static obstacles at positions {s_obstacles}.

There are future goals at positions {f_goals}. You should *avoid* these as if they were obstacles.

There may be moving obstacles. If there are any, their current and future motions will be presented below in a nested list structure:
{k_obstacles}. They move when you move with 90% probability.

Your goal is {goal}.

You must avoid all obstacles, including moving obstacles, as best you can. If you hit an obstacle, you should do your best to exit the obstacle, meaning you can assign a non-zero value to more than one direction.

If the path is not correct or if you do not cover the entire grid, then you will be fired.
"""

PROMPT_TEMPLATE = PromptTemplate(
    template = PROMPT_TEXT,
    input_variables=["size", "s_obstacles", "f_goals", "k_obstacles", "goal"]
)

def get_prompt(size: int, s_obstacles: List[Tuple[int]], f_goals: List[Tuple[int]], k_obstacles: List[List[Tuple[int]]], goal: Tuple[int]):
    size_str = str(size)
    s_obs_str = str(s_obstacles)
    f_obs_str = str(f_goals)
    k_obs_str = str(k_obstacles)
    goal_str = str(goal)

    return PROMPT_TEMPLATE.format(size=size_str, s_obstacles=s_obs_str, f_obstacles=f_obs_str, k_obstacles=k_obs_str, goal=goal_str)


class StateQ(BaseModel):
    x: int = Field(..., description="x coordinate (0-indexed)")
    y: int = Field(..., description="y coordinate (0-indexed)")
    q_values: List[int] = Field(..., description="Action Q-values for this state. Higher is better.")
    
class QTables(BaseModel):
    """A set of Q-values for **each state**"""
    states: List[StateQ]


class BaselinePlanner:
    def __init__(self, size):
        self.size = size
        self.env: GridWorldEnv = GridWorldEnv(size)
        self.model_generator = PrismModelGenerator(size)
        self.prism_verifier = PrismVerifier(self.get_prism_path())
        self.simplified_verifier = SimplifiedVerifier(self.prism_verifier)
        self.action_space = 4
        self.q_table = self.initialize_q_table()
                
        logger.info("Baseline Planner initialized.")
        
    def get_prism_path(self):
        """Get PRISM executable path"""
        prism_path = "/home/moonstone/Documents/Coding/prism-llm/prism-4.9/bin/prism"  # Update this path for your system
        if os.path.exists(prism_path) and os.access(prism_path, os.X_OK):
            return prism_path
        else:
            raise FileNotFoundError(f"PRISM executable not found or not executable at {prism_path}")
        
    def initialize_q_table(self):
        """Initialize Q-table"""
        q_table = {}
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                for g1 in [False, True]:
                    for g2 in [False, True]:
                        for g3 in [False, True]:
                            state = (x, y, g1, g2, g3)
                            q_table[state] = np.ones(self.action_space) * 1.0
        logger.info("Q-table initialized.")
        return q_table
            
    def step(self):
        # after updated q-table
        model_str = self.model_generator.generate_prism_model(self.q_table)
        ltl_score = self.simplified_verifier.verify_policy(model_str)
        
        logger.info(f"LTL Score (BASELINE): {ltl_score}")
        return ltl_score
    









class LLMPlanner:
    def __init__(self, size):
        self.size = size
        self.env: GridWorldEnv = GridWorldEnv(size)
        self.model_generator = PrismModelGenerator(size)
        self.prism_verifier = PrismVerifier(self.get_prism_path())
        self.simplified_verifier = SimplifiedVerifier(self.prism_verifier)
        self.action_space = 4
        self.q_table = self.initialize_q_table()
        self.prism_probs = {
            'goal1': 0.0,
            'goal2': 0.0,
            'goal3': 0.0,
            'seq1': 0.0,
            'seq2': 0.0,
            'seq3': 0.0,
            'avoid_obstacle': 0.0,
            'path_exists': 0.0
        }
        
        self.model = ChatOpenAI(model_name="gpt-5-mini-2025-08-07", temperature=0).with_structured_output(QTables)
        
        logger.info("LLM Planner initialized.")
        
    def get_prism_path(self):
        """Get PRISM executable path"""
        prism_path = "/home/moonstone/Documents/Coding/prism-llm/prism-4.9/bin/prism"  # Update this path for your system
        if os.path.exists(prism_path) and os.access(prism_path, os.X_OK):
            return prism_path
        else:
            raise FileNotFoundError(f"PRISM executable not found or not executable at {prism_path}")
        
    def initialize_q_table(self):
        """Initialize Q-table"""
        q_table = {}
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                for g1 in [False, True]:
                    for g2 in [False, True]:
                        for g3 in [False, True]:
                            state = (x, y, g1, g2, g3)
                            q_table[state] = np.ones(self.action_space) * 1.0
        logger.info("Q-table initialized.")
        return q_table

    def _update_prism_probabilities(self, probabilities: List[float]):
        """Update PMC verification probabilities"""
        if len(probabilities) >= 8:
            self.prism_probs = {
                'goal1': probabilities[0],
                'goal2': probabilities[1],
                'goal3': probabilities[2],
                'seq1': probabilities[3],
                'seq2': probabilities[4],
                'seq3': probabilities[5],
                'avoid_obstacle': probabilities[6],
                'path_exists': probabilities[7]
            }
            
    def step(self):
        # separate out non-positional state (mutually independent states)
        for i in self.env.goals.keys():
            goal = self.env.goals[i]
            logger.info(f"Planning for goal {i} at position {goal}")
            response = self.model.invoke(
                get_prompt(
                    self.size,
                    self.env.static_obstacles,
                    self.env.goals[i + 1:],
                    self.env.moving_obstacle_positions,
                    self.env.goals[i]
                )
            )
            logger.info("LLM Response received.")
            
            logger.info(response.states)
            for stateQ in response.states:
                x, y = stateQ.x, stateQ.y
                if i == 1:
                    applicable_goals = [[False, False, False], [False, True, False], [False, False, True], [False, True, True]]
                if i == 2:
                    applicable_goals = [[True, False, False], [True, False, True]]
                if i == 3:
                    applicable_goals = [[True, True, False],]
                
                for applicable_goal in applicable_goals:
                    g1, g2, g3 = applicable_goal
                    state = (x, y, g1, g2, g3)
                    assert(len(stateQ.q_values) == self.action_space)
                    if state in self.q_table:
                        logger.info(f"Updating Q-values for state {state} with {stateQ.q_values}")
                        for q_idx in range(len(self.q_table[state])):
                            self.q_table[state][q_idx] = self.q_table[state][q_idx] * 0.3 + stateQ.q_values[q_idx] * 0.7 # TODO: change to moving average
                    else:
                        logger.warning(f"State {state} from LLM not in Q-table.")
                
        # after updated q-table
        model_str = self.model_generator.generate_prism_model(self.q_table)
        ltl_score = self.simplified_verifier.verify_policy(model_str)
        
        logger.info(f"LTL Score (LLM): {ltl_score}")
        return ltl_score
            
        
        

def main():
    logger.info("Initializing LLM Planner...")
    llmPlanner = LLMPlanner(GRID_SIZE)
    logger.info("LLM Planner ready. Stepping...")
    start_time = time.time()
    llm_ltl = llmPlanner.step()        
    end_time = time.time()
    logger.info("LLM Planner step complete.")
    logger.info("Computing baseline for comparison...")
    baselinePlanner = BaselinePlanner(GRID_SIZE)
    baseline_ltl = baselinePlanner.step()
    logger.info(f"Baseline finished")
    
    logger.info(f"LLM LTL Score: {llm_ltl}, Baseline LTL Score: {baseline_ltl}")
    if llm_ltl > baseline_ltl:
        logger.info("LLM Planner outperforms baseline.")
    else:
        logger.warning("LLM Planner does not outperform baseline.")
    logger.info(f"LLM Planner step took {end_time - start_time:.2f} seconds.")


logger.info("Starting LLM Planner...")
if __name__ == "__main__":
    main()