from prismUtils.PrismVerifier import PrismVerifier
from prismUtils.SimplifiedVerifier import SimplifiedVerifier
from prismUtils.PrismModelGenerator import PrismModelGenerator
from prismUtils.GridWorldEnv import GridWorldEnv

import logging
import os
import numpy as np
from typing import List, Optional, Dict, Tuple
from itertools import product

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from pydantic import BaseModel, Field

import time

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/debug.log"),
                        logging.StreamHandler()
                    ])

with open("logs/debug.log", "w") as f:
    f.write("")  # Clear the log file at the start of each run

logger = logging.getLogger(__name__)


GRID_SIZE = 2

GOALS = {
    1: (1, 1),
}
# Static obstacles
STATIC_OBSTACLES = [(1, 0)]

# Moving obstacle positions in sequence
MOVING_OBSTACLES = []


PROMPT_TEXT = """You are an expert path planner working on formulating paths that meet formal requirements.

The grid world is {size} x {size}. Here is the visual layout:

{grid_visual}

Legend:
- 'S' = Start position (0,0) - the initial state
- 'G' = Current goal position you must reach
- 'X' = Static obstacle (CANNOT enter - you will bounce back)
- 'F' = Future goal (treat as obstacle for now - avoid it)
- 'M' = Moving obstacle position (avoid if possible)
- '.' = Empty cell you can move through

COORDINATE SYSTEM:
- Position format: (row, col) where row is Y-axis, col is X-axis
- (0,0) is in the TOP-LEFT corner
- Row increases DOWNWARD (0 → 1 → 2...)
- Column increases RIGHTWARD (0 → 1 → 2...)

ACTIONS:
You must provide Q-values for 4 directional actions at each state:
- Index 0 = UP: Move to (row-1, col) - DECREASES row
- Index 1 = RIGHT: Move to (row, col+1) - INCREASES column  
- Index 2 = DOWN: Move to (row+1, col) - INCREASES row
- Index 3 = LEFT: Move to (row, col-1) - DECREASES column

ACTION EXAMPLES:
From (0,0): UP→boundary, RIGHT→(0,1), DOWN→(1,0), LEFT→boundary
From (0,1): UP→boundary, RIGHT→(0,2), DOWN→(1,1), LEFT→(0,0)
From (1,0): UP→(0,0), RIGHT→(1,1), DOWN→(2,0), LEFT→boundary

Q-VALUE ENCODING:
- Set the BEST action to 100
- Set other actions to 0
- If multiple good actions exist (e.g., escaping obstacle), assign non-zero to multiple
- If at obstacle, prioritize escaping back to valid cells

TASK DETAILS:
- Static obstacles: {s_obstacles} (marked as 'X')
- Future goals to avoid: {f_goals} (marked as 'F')
- Moving obstacles: {k_obstacles} (marked as 'M', move with 90% probability)
- Your current goal: {goal} (marked as 'G')

CRITICAL REQUIREMENTS:
1. You MUST provide Q-values for ALL {total_states} states in the {size}x{size} grid
2. Never plan a path through obstacles (X) or future goals (F)
3. The best action should create a path from ANY position to the goal
4. If a cell is an obstacle, provide escape Q-values (how to exit if accidentally there)

EXAMPLE for a 2x2 grid with goal at (1,1) and obstacle at (1,0):

Grid:
  0 1
0 S .
1 X G

Correct solution:
- State (0,0): [0, 100, 0, 0] → Go RIGHT toward goal
- State (0,1): [0, 0, 100, 0] → Go DOWN to reach goal  
- State (1,0): [100, 0, 0, 0] → Go UP to escape obstacle
- State (1,1): [0, 0, 0, 0] → At goal, any action is fine

Now provide Q-values for your grid.
"""

PROMPT_TEMPLATE = PromptTemplate(
    template=PROMPT_TEXT,
    input_variables=["size", "grid_visual", "s_obstacles", "f_goals", "k_obstacles", "goal", "total_states"]
)

def generate_grid_visual(size: int, goal: Tuple[int, int], s_obstacles: List[Tuple[int, int]], 
                         f_goals: List[Tuple[int, int]], k_obstacles: List[Tuple[int, int]]) -> str:
    """Generate ASCII visual representation of the grid"""
    # Initialize grid with empty cells
    grid = [['.' for _ in range(size)] for _ in range(size)]
    
    # Place elements (order matters - later placements override)
    # Start with moving obstacles
    for obs in k_obstacles:
        if 0 <= obs[0] < size and 0 <= obs[1] < size:
            grid[obs[0]][obs[1]] = 'M'
    
    # Future goals
    for fg in f_goals:
        if 0 <= fg[0] < size and 0 <= fg[1] < size:
            grid[fg[0]][fg[1]] = 'F'
    
    # Static obstacles (higher priority)
    for obs in s_obstacles:
        if 0 <= obs[0] < size and 0 <= obs[1] < size:
            grid[obs[0]][obs[1]] = 'X'
    
    # Current goal (even higher priority)
    if 0 <= goal[0] < size and 0 <= goal[1] < size:
        grid[goal[0]][goal[1]] = 'G'
    
    # Start position (highest priority - but don't override goal if they're the same)
    if grid[0][0] not in ['G']:
        grid[0][0] = 'S'
    
    # Format as string with column headers
    header = '  ' + ' '.join(str(i) for i in range(size))
    rows = [f"{i} " + ' '.join(row) for i, row in enumerate(grid)]
    visual = header + '\n' + '\n'.join(rows)
    
    return visual

def get_prompt(size: int, s_obstacles: List[Tuple[int, int]], f_goals: List[Tuple[int, int]], 
               k_obstacles: List[Tuple[int, int]], goal: Tuple[int, int]) -> str:
    """Generate the complete prompt with visual grid"""
    
    # Flatten k_obstacles if it's nested
    flat_k_obstacles = []
    if k_obstacles:
        for item in k_obstacles:
            if isinstance(item, list):
                flat_k_obstacles.extend(item)
            else:
                flat_k_obstacles.append(item)
    
    grid_visual = generate_grid_visual(size, goal, s_obstacles, f_goals, flat_k_obstacles)
    total_states = size * size
    
    return PROMPT_TEMPLATE.format(
        size=size,
        grid_visual=grid_visual,
        s_obstacles=str(s_obstacles),
        f_goals=str(f_goals),
        k_obstacles=str(k_obstacles),
        goal=str(goal),
        total_states=total_states
    )


class StateQ(BaseModel):
    x: int = Field(..., description="x coordinate (row, 0-indexed)")
    y: int = Field(..., description="y coordinate (column, 0-indexed)")
    q_values: List[int] = Field(..., description="Action Q-values for this state. Higher is better.")
    
class QTables(BaseModel):
    """A set of Q-values for **each state**"""
    states: List[StateQ]


# all q-values initialized to 1
class BaselinePlanner:
    def __init__(self, size, goals, static_obstacles, moving_obstacles):
        self.size = size
        self.env = GridWorldEnv(size, goals, static_obstacles, moving_obstacles)
        self.model_generator = PrismModelGenerator(self.env)
        self.prism_verifier = PrismVerifier(self.get_prism_path())
        self.simplified_verifier = SimplifiedVerifier(self.prism_verifier, self.env)
        self.action_space = 4
        self.num_goals = len(goals)
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
        """Initialize Q-table with variable number of goals"""
        q_table = {}
        
        # Generate all combinations of goal states
        goal_combinations = list(product([False, True], repeat=self.num_goals))
        
        for x in range(self.size):
            for y in range(self.size):
                for goal_combo in goal_combinations:
                    state = (x, y) + goal_combo
                    q_table[state] = np.ones(self.action_space) * 1.0
        
        logger.info(f"Q-table initialized with {len(q_table)} states.")
        return q_table
            
    def step(self):
        # after updated q-table
        model_str = self.model_generator.generate_prism_model(self.q_table)
        ltl_score = self.simplified_verifier.verify_policy(model_str)
        
        logger.info(f"LTL Score (BASELINE): {ltl_score}")
        return ltl_score
    


class LLMPlanner:
    def __init__(self, size, goals, static_obstacles, moving_obstacles):
        self.size = size
        self.env = GridWorldEnv(size, goals, static_obstacles, moving_obstacles)
        self.model_generator = PrismModelGenerator(self.env)
        self.prism_verifier = PrismVerifier(self.get_prism_path())
        self.simplified_verifier = SimplifiedVerifier(self.prism_verifier, self.env)
        self.action_space = 4
        self.num_goals = len(goals)
        self.q_table = self.initialize_q_table()
        self.prism_probs = {}  # Will be populated dynamically
        
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
        """Initialize Q-table with variable number of goals"""
        q_table = {}
        
        # Generate all combinations of goal states
        goal_combinations = list(product([False, True], repeat=self.num_goals))
        
        for x in range(self.size):
            for y in range(self.size):
                for goal_combo in goal_combinations:
                    state = (x, y) + goal_combo
                    q_table[state] = np.ones(self.action_space) * 1.0
        
        logger.info(f"Q-table initialized with {len(q_table)} states.")
        return q_table

    def _update_prism_probabilities(self, probabilities: List[float]):
        """Update PMC verification probabilities dynamically"""
        goal_nums = sorted(self.env.goals.keys())
        
        idx = 0
        # Goal reachability probabilities
        for goal_num in goal_nums:
            if idx < len(probabilities):
                self.prism_probs[f'goal{goal_num}'] = probabilities[idx]
                idx += 1
        
        # Sequence probabilities
        for i in range(len(goal_nums) - 1):
            if idx < len(probabilities):
                self.prism_probs[f'seq_{goal_nums[i]}_before_{goal_nums[i+1]}'] = probabilities[idx]
                idx += 1
        
        # Complete sequence
        if idx < len(probabilities):
            self.prism_probs['complete_sequence'] = probabilities[idx]
            idx += 1
        
        # Obstacle avoidance
        if idx < len(probabilities):
            self.prism_probs['avoid_obstacle'] = probabilities[idx]
            idx += 1
    
    def _get_applicable_goal_states(self, goal_idx: int) -> List[List[bool]]:
        """Get the goal state combinations applicable for planning to reach goal i
        
        For goal at index i (0-indexed):
        - Goals 0 to i-1: must all be True (already reached)
        - Goal i: must be False (trying to reach it)
        - Goals i+1 to N-1: must all be False (haven't reached them yet)
        
        Returns a list with single element since there's only one valid state
        """
        goal_state = []
        
        for j in range(self.num_goals):
            if j < goal_idx:
                # Previous goals must be reached
                goal_state.append(True)
            elif j == goal_idx:
                # Current goal not yet reached
                goal_state.append(False)
            else:
                # Future goals not yet reached
                goal_state.append(False)
        
        return [goal_state]
            
    def step(self):
        # Plan for each goal separately
        goal_nums = sorted(self.env.goals.keys())
        
        for idx, goal_num in enumerate(goal_nums):
            goal = self.env.goals[goal_num]
            logger.info(f"Planning for goal {goal_num} at position {goal}")
            
            # Get future goals to avoid
            future_goals = [self.env.goals[k] for k in goal_nums if k > goal_num]
            
            response = self.model.invoke(
                get_prompt(
                    self.size,
                    self.env.static_obstacles,
                    future_goals,
                    self.env.moving_obstacle_positions,
                    goal
                )
            )
            logger.info("LLM Response received.")
            logger.info(response.states)
            
            # Get applicable goal state combinations for this goal
            applicable_goals = self._get_applicable_goal_states(idx)
            
            for stateQ in response.states:
                x, y = stateQ.x, stateQ.y
                
                for goal_state_list in applicable_goals:
                    goal_state_tuple = tuple(goal_state_list)
                    state = (x, y) + goal_state_tuple
                    
                    assert len(stateQ.q_values) == self.action_space, \
                        f"Expected {self.action_space} Q-values, got {len(stateQ.q_values)}"
                    
                    if state in self.q_table:
                        logger.info(f"Updating Q-values for state {state} with {stateQ.q_values}")
                        for q_idx in range(len(self.q_table[state])):
                            # Exponential moving average
                            self.q_table[state][q_idx] = self.q_table[state][q_idx] * 0.3 + stateQ.q_values[q_idx] * 0.7
                    else:
                        logger.warning(f"State {state} from LLM not in Q-table.")
                
        # After updated q-table, verify the policy
        model_str = self.model_generator.generate_prism_model(self.q_table)
        ltl_score = self.simplified_verifier.verify_policy(model_str)
        
        # Update probabilities from verification
        if self.simplified_verifier.ltl_probabilities:
            self._update_prism_probabilities(self.simplified_verifier.ltl_probabilities[-1])
        
        logger.info(f"LTL Score (LLM): {ltl_score}")
        return ltl_score
            
        
        

def main():
    logger.info("Initializing LLM Planner...")
    llmPlanner = LLMPlanner(GRID_SIZE, GOALS, STATIC_OBSTACLES, MOVING_OBSTACLES)
    logger.info("LLM Planner ready. Stepping...")
    start_time = time.time()
    llm_ltl = llmPlanner.step()        
    end_time = time.time()
    logger.info("LLM Planner step complete.")
    logger.info("Computing baseline for comparison...")
    baselinePlanner = BaselinePlanner(GRID_SIZE, GOALS, STATIC_OBSTACLES, MOVING_OBSTACLES)
    baseline_ltl = baselinePlanner.step()
    logger.info(f"Baseline finished")
    
    logger.info(f"LLM LTL Score: {llm_ltl}, Baseline LTL Score: {baseline_ltl}")
    if llm_ltl > baseline_ltl:
        logger.info("LLM Planner outperforms baseline.")
    else:
        logger.warning("LLM Planner does not outperform baseline.")
    logger.info(f"LLM Planner step took {end_time - start_time:.2f} seconds.")

    llmPlanner.simplified_verifier.save_probabilities_to_file(filename="llm_probabilities_and_rewards.txt")
    baselinePlanner.simplified_verifier.save_probabilities_to_file(filename="baseline_probabilities_and_rewards.txt")

logger.info("Starting LLM Planner...")
if __name__ == "__main__":
    main()