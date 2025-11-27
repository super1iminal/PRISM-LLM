from verification.PrismVerifier import PrismVerifier
from verification.SimplifiedVerifier import SimplifiedVerifier
from verification.PrismModelGenerator import PrismModelGenerator
from environment.GridWorld import GridWorld
from utils.LLMPrompting import get_prompt, QTables

import os
import numpy as np
from typing import List, Optional, Dict, Tuple
from itertools import product
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import traceback

from langchain_openai import ChatOpenAI

from time import time

from config.Settings import PRISM_PATH
from utils.Logging import setup_logger, create_run_directory


def _evaluate_single_gridworld_vanilla(args: Tuple) -> Dict:
    """
    Standalone function to evaluate a single gridworld with Vanilla LLM planner.
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
    planner = VanillaLLMPlanner()
    
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
    
    ltl_score = planner.step()
    planner.logger.info(f"Evaluation {idx+1}: LTL Score = {ltl_score}")
    end_time = time()
    delta_time = end_time - start_time
    
    return {
        "index": idx,
        "LTL_Score": ltl_score,
        "Prism_Probabilities": planner.prism_probs.copy(),
        "Evaluation_Time": delta_time
    }


class VanillaLLMPlanner:
    def __init__(self):
        self.action_space = 4
        self.prism_probs = {}  # Will be populated dynamically
        
        self.model = ChatOpenAI(model_name="gpt-4o-mini-2024-08-07", temperature=1).with_structured_output(QTables)
        
        
    def evaluate(self, dataloader, parallel: bool = False, max_workers: Optional[int] = None,
                 use_threads: bool = True):
        """
        Evaluate the Vanilla LLM planner on all gridworlds in the dataloader.
        
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
        run_dir = create_run_directory("vanilla_LLM_sequential")
        
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
            
            ltl_score = self.step()
            self.logger.info(f"Evaluation {idx+1}: LTL Score = {ltl_score}")
            end_time = time()
            delta_time = end_time - start_time
            results_list.append({
                "LTL_Score": ltl_score,
                "Prism_Probabilities": self.prism_probs.copy(),
                "Evaluation_Time": delta_time
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
        run_dir = create_run_directory("vanilla_LLM_parallel")
        
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
                executor.submit(_evaluate_single_gridworld_vanilla, args): args[0]
                for args in args_list
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results_dict[result["index"]] = result
                    main_logger.info(
                        f"Completed gridworld {idx+1}: LTL Score = {result['LTL_Score']:.4f}"
                    )
                except Exception as e:
                    main_logger.error(f"Gridworld {idx+1} failed with error: {str(e)}")
                    main_logger.error(traceback.format_exc())
                    results_dict[idx] = {
                        "LTL_Score": 0.0,
                        "Prism_Probabilities": {},
                        "Evaluation_Time": 0.0,
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
                "Evaluation_Time": 0.0
            })
            result.pop("index", None)
            results_list.append(result)
        
        return results_list

        
    def get_prism_path(self):
        """Get PRISM executable path"""
        prism_path = PRISM_PATH  # Update this path for your system
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
        
        self.logger.info(f"Q-table initialized with {len(q_table)} states.")
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
            self.logger.info(f"Planning for goal {goal_num} at position {goal}")
            
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
            self.logger.info("LLM Response received.")
            self.logger.info(response.states)
            
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
                        self.logger.info(f"Updating Q-values for state {state} with {stateQ.q_values}")
                        for q_idx in range(len(self.q_table[state])):
                            # Exponential moving average
                            self.q_table[state][q_idx] = self.q_table[state][q_idx] * 0.3 + stateQ.q_values[q_idx] * 0.7
                    else:
                        self.logger.warning(f"State {state} from LLM not in Q-table.")
                
        # After updated q-table, verify the policy
        model_str = self.model_generator.generate_prism_model(self.q_table)
        ltl_score = self.simplified_verifier.verify_policy(model_str)
        
        # Update probabilities from verification
        if self.simplified_verifier.ltl_probabilities:
            self._update_prism_probabilities(self.simplified_verifier.ltl_probabilities[-1])
        
        self.logger.info(f"LTL Score (LLM): {ltl_score}")
        return ltl_score