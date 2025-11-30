from verification.PrismVerifier import PrismVerifier
from verification.SimplifiedVerifier import SimplifiedVerifier
from verification.PrismModelGenerator import PrismModelGenerator
from environment.GridWorld import GridWorld
from utils.LLMPrompting import get_prompt, QTables, StateQ, generate_grid_visual

import os
import numpy as np
from typing import List, Optional, Dict, Tuple
from itertools import product
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import traceback

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from time import time

from config.Settings import PRISM_PATH
from utils.Logging import setup_logger, create_run_directory


# Feedback prompt template for correction iterations
FEEDBACK_PROMPT_TEXT = """You are an expert path planner. Your previous plan did NOT achieve perfect probabilities.

PREVIOUS ATTEMPT RESULTS:
{probability_summary}

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
- Index 0 = UP: Move to (row-1, col) - DECREASES row
- Index 1 = RIGHT: Move to (row, col+1) - INCREASES column  
- Index 2 = DOWN: Move to (row+1, col) - INCREASES row
- Index 3 = LEFT: Move to (row, col-1) - DECREASES column

YOUR PREVIOUS Q-VALUES (that need improvement):
{previous_q_values}

PROBLEMS TO FIX:
{problems}

Q-VALUE ENCODING:
- Set the BEST action to 100
- Set other actions to 0
- If multiple good actions exist, assign non-zero to multiple

TASK DETAILS:
- Static obstacles: {s_obstacles} (marked as 'X')
- Future goals to avoid: {f_goals} (marked as 'F')
- Moving obstacles: {k_obstacles} (marked as 'M')
- Your current goal: {goal} (marked as 'G')

CRITICAL: Fix the issues identified above. Provide IMPROVED Q-values for ALL {total_states} states.
"""

FEEDBACK_TEMPLATE = PromptTemplate(
    template=FEEDBACK_PROMPT_TEXT,
    input_variables=["probability_summary", "size", "grid_visual", "s_obstacles", 
                     "f_goals", "k_obstacles", "goal", "total_states", 
                     "previous_q_values", "problems"]
)


def format_previous_q_values(states: List[StateQ]) -> str:
    """Format previous Q-values for feedback prompt"""
    lines = []
    for state in states[:20]:  # Limit to avoid token overflow
        lines.append(f"  ({state.x}, {state.y}): {state.q_values}")
    if len(states) > 20:
        lines.append(f"  ... and {len(states) - 20} more states")
    return "\n".join(lines)


def identify_problems(prism_probs: Dict[str, float], threshold: float = 1.0) -> str:
    """Identify problems based on probabilities that are below threshold"""
    problems = []
    
    for key, prob in prism_probs.items():
        if prob < threshold:
            if key.startswith('goal'):
                problems.append(f"- Goal reachability ({key}): {prob:.4f} < 1.0 - Path to goal may be blocked or suboptimal")
            elif key.startswith('seq_'):
                problems.append(f"- Sequence ordering ({key}): {prob:.4f} < 1.0 - Goals may be visited in wrong order")
            elif key == 'complete_sequence':
                problems.append(f"- Complete sequence: {prob:.4f} < 1.0 - Not all goals reached in correct order")
            elif key == 'avoid_obstacle':
                problems.append(f"- Obstacle avoidance: {prob:.4f} < 1.0 - Path may go through obstacles")
            else:
                problems.append(f"- {key}: {prob:.4f} < 1.0")
    
    if not problems:
        return "No specific problems identified, but overall LTL score is below 1.0"
    
    return "\n".join(problems)


def format_probability_summary(prism_probs: Dict[str, float]) -> str:
    """Format probability summary for feedback"""
    lines = ["Current verification probabilities:"]
    for key, prob in prism_probs.items():
        status = "✓" if prob >= 0.99 else "✗"
        lines.append(f"  {status} {key}: {prob:.4f}")
    return "\n".join(lines)


def _evaluate_single_gridworld_feedback(args: Tuple) -> Dict:
    """
    Standalone function to evaluate a single gridworld with Feedback LLM planner.
    This function is designed to be called in a separate process/thread.
    
    Args:
        args: Tuple of (idx, gridworld, expected_steps, config)
        
    Returns:
        Dictionary containing evaluation results
    """
    idx, gridworld, expected_steps, config = args
    
    # Get shared run directory from config
    run_dir = config.get('run_dir', None)
    
    # Create a new planner instance for this worker
    planner = FeedbackLLMPlanner(
        max_iterations=config.get('max_iterations', 3),
        target_threshold=config.get('target_threshold', 0.99)
    )
    
    # Setup logger for this worker in the shared directory
    planner.logger = setup_logger(f"worker_{idx}", run_dir=run_dir, include_timestamp=False)
    planner.prism_verifier = PrismVerifier(planner.get_prism_path(), planner.logger)
    
    start_time = time()
    planner.size = gridworld.size
    planner.env = gridworld
    planner.model_generator = PrismModelGenerator(gridworld, planner.logger)
    planner.simplified_verifier = SimplifiedVerifier(planner.prism_verifier, gridworld, planner.logger)
    planner.num_goals = len(gridworld.goals)
    planner.q_table = planner.initialize_q_table()
    planner.prism_probs = {}
    
    ltl_score, iterations_used = planner.step()
    planner.logger.info(f"Evaluation {idx+1}: LTL Score = {ltl_score} (iterations: {iterations_used})")
    end_time = time()
    delta_time = end_time - start_time
    
    return {
        "index": idx,
        "LTL_Score": ltl_score,
        "Prism_Probabilities": planner.prism_probs.copy(),
        "Evaluation_Time": delta_time,
        "Iterations_Used": iterations_used
    }


class FeedbackLLMPlanner:
    def __init__(self, max_iterations: int = 3, target_threshold: float = 0.99):
        self.action_space = 4
        self.prism_probs = {}
        self.max_iterations = max_iterations
        self.target_threshold = target_threshold
        
        self.model = ChatOpenAI(
            model_name="gpt-5-mini-2025-08-07", 
            temperature=1
        ).with_structured_output(QTables)
        
    def evaluate(self, dataloader, parallel: bool = False, max_workers: Optional[int] = None,
                 use_threads: bool = True):
        """
        Evaluate the Feedback LLM planner on all gridworlds in the dataloader.
        
        Args:
            dataloader: DataLoader instance containing GridWorld configurations
            parallel: If True, evaluate gridworlds in parallel
            max_workers: Maximum number of workers. If None, uses CPU count for processes
                        or min(32, CPU count + 4) for threads.
            use_threads: If True and parallel=True, use ThreadPoolExecutor instead of 
                        ProcessPoolExecutor. Threads are recommended for I/O-bound LLM calls.
            
        Returns:
            List of results containing LTL scores and evaluation statistics
        """
        if parallel:
            return self._evaluate_parallel(dataloader, max_workers, use_threads)
        else:
            return self._evaluate_sequential(dataloader)
    
    def _evaluate_sequential(self, dataloader) -> List[Dict]:
        """Sequential evaluation of all gridworlds"""
        # Create a run directory for this evaluation
        run_dir = create_run_directory("feedback_LLM_sequential")
        
        self.logger = setup_logger("main", run_dir=run_dir, include_timestamp=False)
        self.prism_verifier = PrismVerifier(self.get_prism_path(), self.logger)
        results_list = []
        
        self.logger.info(f"Logs will be saved to: {run_dir}")
        
        for idx, (gridworld, _) in enumerate(dataloader):
            start_time = time()
            self.size = gridworld.size
            self.env = gridworld
            self.model_generator = PrismModelGenerator(self.env, self.logger)
            self.simplified_verifier = SimplifiedVerifier(self.prism_verifier, self.env, self.logger)
            self.num_goals = len(gridworld.goals)
            self.q_table = self.initialize_q_table()
            self.prism_probs = {}
            
            ltl_score, iterations_used = self.step()
            self.logger.info(f"Evaluation {idx+1}: LTL Score = {ltl_score} (iterations: {iterations_used})")
            end_time = time()
            delta_time = end_time - start_time
            results_list.append({
                "LTL_Score": ltl_score,
                "Prism_Probabilities": self.prism_probs.copy(),
                "Evaluation_Time": delta_time,
                "Iterations_Used": iterations_used
            })
            
        return results_list
    
    def _evaluate_parallel(self, dataloader, max_workers: Optional[int] = None,
                          use_threads: bool = True) -> List[Dict]:
        """
        Parallel evaluation of all gridworlds.
        
        Args:
            dataloader: DataLoader instance containing GridWorld configurations
            max_workers: Maximum number of workers
            use_threads: If True, use ThreadPoolExecutor (better for I/O-bound LLM calls)
            
        Returns:
            List of results containing LTL scores and evaluation statistics
        """
        # Create a shared run directory for all workers
        run_dir = create_run_directory("feedback_LLM_parallel")
        
        # Setup main logger in the shared directory
        main_logger = setup_logger("main", run_dir=run_dir, include_timestamp=False)
        
        if max_workers is None:
            if use_threads:
                max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)
            else:
                max_workers = multiprocessing.cpu_count()
        
        executor_type = "threads" if use_threads else "processes"
        main_logger.info(f"Starting parallel evaluation with {max_workers} {executor_type}")
        main_logger.info(f"Logs will be saved to: {run_dir}")
        
        # Prepare arguments with shared run directory
        config = {
            'max_iterations': self.max_iterations,
            'target_threshold': self.target_threshold,
            'run_dir': run_dir  # Pass shared directory to workers
        }
        data_list = list(dataloader)
        args_list = [
            (idx, gridworld, expected_steps, config)
            for idx, (gridworld, expected_steps) in enumerate(data_list)
        ]
        
        results_dict = {}
        total_start_time = time()
        
        # Choose executor type - threads are better for I/O-bound LLM API calls
        ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        
        with ExecutorClass(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_evaluate_single_gridworld_feedback, args): args[0]
                for args in args_list
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results_dict[result["index"]] = result
                    main_logger.info(
                        f"Completed gridworld {idx+1}: LTL Score = {result['LTL_Score']:.4f} "
                        f"(iterations: {result['Iterations_Used']})"
                    )
                except Exception as e:
                    main_logger.error(f"Gridworld {idx+1} failed with error: {str(e)}")
                    main_logger.error(traceback.format_exc())
                    results_dict[idx] = {
                        "LTL_Score": 0.0,
                        "Prism_Probabilities": {},
                        "Evaluation_Time": 0.0,
                        "Iterations_Used": 0,
                        "error": str(e)
                    }
        
        total_time = time() - total_start_time
        main_logger.info(f"Total parallel evaluation time: {total_time:.2f} seconds")
        
        # Sort results by original index
        results_list = []
        for idx in range(len(args_list)):
            result = results_dict.get(idx, {
                "LTL_Score": 0.0,
                "Prism_Probabilities": {},
                "Evaluation_Time": 0.0,
                "Iterations_Used": 0
            })
            result.pop("index", None)
            results_list.append(result)
        
        return results_list

    def get_prism_path(self):
        """Get PRISM executable path"""
        prism_path = PRISM_PATH
        if os.path.exists(prism_path) and os.access(prism_path, os.X_OK):
            return prism_path
        else:
            raise FileNotFoundError(f"PRISM executable not found or not executable at {prism_path}")
        
    def initialize_q_table(self):
        """Initialize Q-table with variable number of goals"""
        q_table = {}
        goal_combinations = list(product([False, True], repeat=self.num_goals))
        
        for x in range(self.size):
            for y in range(self.size):
                for goal_combo in goal_combinations:
                    state = (x, y) + goal_combo
                    q_table[state] = np.ones(self.action_space) * 1.0
        
        self.logger.info(f"Q-table initialized with {len(q_table)} states.")
        return q_table

    def _update_prism_probabilities(self, probabilities: List[float]):
        """Update PMC verification probabilities dynamically"""
        goal_nums = sorted(self.env.goals.keys())
        
        idx = 0
        for goal_num in goal_nums:
            if idx < len(probabilities):
                self.prism_probs[f'goal{goal_num}'] = probabilities[idx]
                idx += 1
        
        for i in range(len(goal_nums) - 1):
            if idx < len(probabilities):
                self.prism_probs[f'seq_{goal_nums[i]}_before_{goal_nums[i+1]}'] = probabilities[idx]
                idx += 1
        
        if idx < len(probabilities):
            self.prism_probs['complete_sequence'] = probabilities[idx]
            idx += 1
        
        if idx < len(probabilities):
            self.prism_probs['avoid_obstacle'] = probabilities[idx]
            idx += 1
    
    def _get_applicable_goal_states(self, goal_idx: int) -> List[List[bool]]:
        """Get the goal state combinations applicable for planning to reach goal i"""
        goal_state = []
        
        for j in range(self.num_goals):
            if j < goal_idx:
                goal_state.append(True)
            elif j == goal_idx:
                goal_state.append(False)
            else:
                goal_state.append(False)
        
        return [goal_state]
    
    def _check_all_probabilities_meet_threshold(self) -> bool:
        """Check if all probabilities meet the target threshold"""
        if not self.prism_probs:
            return False
        return all(prob >= self.target_threshold for prob in self.prism_probs.values())
    
    def _get_feedback_prompt(self, goal_num: int, goal: Tuple[int, int], 
                              future_goals: List[Tuple[int, int]], 
                              previous_response: QTables) -> str:
        """Generate feedback prompt for correction iteration"""
        
        flat_k_obstacles = []
        if self.env.moving_obstacle_positions:
            for item in self.env.moving_obstacle_positions:
                if isinstance(item, list):
                    flat_k_obstacles.extend(item)
                else:
                    flat_k_obstacles.append(item)
        
        grid_visual = generate_grid_visual(
            self.size, goal, self.env.static_obstacles, 
            future_goals, flat_k_obstacles
        )
        
        probability_summary = format_probability_summary(self.prism_probs)
        previous_q_values = format_previous_q_values(previous_response.states)
        problems = identify_problems(self.prism_probs, self.target_threshold)
        
        return FEEDBACK_TEMPLATE.format(
            probability_summary=probability_summary,
            size=self.size,
            grid_visual=grid_visual,
            s_obstacles=str(self.env.static_obstacles),
            f_goals=str(future_goals),
            k_obstacles=str(self.env.moving_obstacle_positions),
            goal=str(goal),
            total_states=self.size * self.size,
            previous_q_values=previous_q_values,
            problems=problems
        )
    
    def _apply_response_to_q_table(self, response: QTables, goal_idx: int):
        """Apply LLM response to Q-table"""
        applicable_goals = self._get_applicable_goal_states(goal_idx)
        
        for stateQ in response.states:
            x, y = stateQ.x, stateQ.y
            
            for goal_state_list in applicable_goals:
                goal_state_tuple = tuple(goal_state_list)
                state = (x, y) + goal_state_tuple
                
                assert len(stateQ.q_values) == self.action_space, \
                    f"Expected {self.action_space} Q-values, got {len(stateQ.q_values)}"
                
                if state in self.q_table:
                    self.logger.info(f"Updating Q-values for state {state} with {stateQ.q_values}")
                    for q_idx in range(len(self.q_table[state])):
                        self.q_table[state][q_idx] = self.q_table[state][q_idx] * 0.3 + stateQ.q_values[q_idx] * 0.7
                else:
                    self.logger.warning(f"State {state} from LLM not in Q-table.")
    
    def _verify_current_policy(self) -> float:
        """Verify current policy and update probabilities"""
        model_str = self.model_generator.generate_prism_model(self.q_table)
        ltl_score = self.simplified_verifier.verify_policy(model_str)
        
        if self.simplified_verifier.ltl_probabilities:
            self._update_prism_probabilities(self.simplified_verifier.ltl_probabilities[-1])
        
        return ltl_score
            
    def step(self) -> Tuple[float, int]:
        """Main planning step with feedback loop"""
        goal_nums = sorted(self.env.goals.keys())
        
        # Store responses for potential feedback iterations
        goal_responses: Dict[int, QTables] = {}
        
        # Initial pass - get Q-values for each goal
        for idx, goal_num in enumerate(goal_nums):
            goal = self.env.goals[goal_num]
            self.logger.info(f"Planning for goal {goal_num} at position {goal}")
            
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
            self.logger.info("LLM Response received.")
            self.logger.info(response.states)
            
            goal_responses[goal_num] = response
            self._apply_response_to_q_table(response, idx)
        
        # Initial verification
        ltl_score = self._verify_current_policy()
        self.logger.info(f"Initial LTL Score: {ltl_score}")
        
        # Feedback loop
        iteration = 1
        while iteration < self.max_iterations and not self._check_all_probabilities_meet_threshold():
            self.logger.info(f"=== Feedback Iteration {iteration} ===")
            self.logger.info(f"Current probabilities: {self.prism_probs}")
            
            # Re-query for each goal with feedback
            for idx, goal_num in enumerate(goal_nums):
                goal = self.env.goals[goal_num]
                future_goals = [self.env.goals[k] for k in goal_nums if k > goal_num]
                
                self.logger.info(f"Re-planning goal {goal_num} with feedback")
                
                feedback_prompt = self._get_feedback_prompt(
                    goal_num, goal, future_goals, goal_responses[goal_num]
                )
                
                response = self.model.invoke(feedback_prompt)
                self.logger.info("Feedback LLM Response received.")
                
                goal_responses[goal_num] = response
                self._apply_response_to_q_table(response, idx)
            
            # Re-verify
            ltl_score = self._verify_current_policy()
            self.logger.info(f"LTL Score after iteration {iteration}: {ltl_score}")
            
            iteration += 1
        
        if self._check_all_probabilities_meet_threshold():
            self.logger.info(f"All probabilities meet threshold after {iteration} iterations!")
        else:
            self.logger.warning(f"Max iterations ({self.max_iterations}) reached. Final probabilities: {self.prism_probs}")
        
        self.logger.info(f"Final LTL Score (Feedback LLM): {ltl_score}")
        return ltl_score, iteration