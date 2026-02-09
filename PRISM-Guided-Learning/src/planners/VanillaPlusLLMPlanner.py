from verification.PrismVerifier import PrismVerifier
from verification.SimplifiedVerifier import SimplifiedVerifier
from verification.PrismModelGenerator import PrismModelGenerator
from environment.GridWorld import GridWorld
from utils.LLMPrompting import get_prompt, ActionPolicy

import os
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


def _evaluate_single_gridworld_vanilla_plus(args: Tuple) -> Dict:
    """
    Standalone function to evaluate a single gridworld with Vanilla Plus LLM planner.
    This function is designed to be called in a separate process/thread.

    Args:
        args: Tuple of (idx, gridworld, expected_steps, config)

    Returns:
        Dictionary containing evaluation results
    """
    idx, gridworld, expected_steps, config = args

    # Get shared run directory from config
    run_dir = config.get('run_dir', None)
    model = config.get('model')
    model_name = config.get('model_name')

    # Create a new planner instance for this worker
    planner = VanillaPlusLLMPlanner(
        model=model,
        model_name=model_name,
        max_attempts=config.get('max_attempts', 3),
        target_threshold=config.get('target_threshold', 0.9)
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

    step_result = planner.step()
    planner.logger.info(f"Evaluation {idx+1}: LTL Score = {step_result['ltl_score']} (iterations: {step_result['iterations']})")
    end_time = time()
    delta_time = end_time - start_time

    return {
        "index": idx,
        "LTL_Score": step_result["ltl_score"],
        "Prism_Probabilities": planner.prism_probs.copy(),
        "Evaluation_Time": delta_time,
        "Iterations_Used": step_result["iterations"],
        "Iteration_Times": step_result["iteration_times"],
        "Iteration_PRISM_Times": step_result["iteration_prism_times"],
        "Iteration_LLM_Times": step_result["iteration_llm_times"],
        "Iteration_Prism_Probs": step_result["iteration_prism_probs"],
        "Iteration_Mistakes": step_result["iteration_mistakes"],
        "Iteration_Costs": step_result["iteration_costs"],
        "Success": step_result["success"]
    }


class VanillaPlusLLMPlanner:
    """
    Vanilla Plus LLM Planner - Multiple iterations with NO feedback.

    Unlike VanillaLLMPlanner (single iteration), this planner retries from scratch
    multiple times until probabilities meet the threshold or max_attempts is reached.

    Key difference from FeedbackLLMPlanner: Uses the initial prompt for ALL iterations,
    never provides feedback about previous attempt results.
    """

    def __init__(self, model, model_name: str, max_attempts: int = 3, target_threshold: float = 0.9):
        self.action_space = 4
        self.prism_probs = {}  # Will be populated dynamically
        self.model = model
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.target_threshold = target_threshold


    def evaluate(self, dataloader, parallel: bool = False, max_workers: Optional[int] = None,
                 use_threads: bool = True):
        """
        Evaluate the Vanilla Plus LLM planner on all gridworlds in the dataloader.

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
        run_dir = create_run_directory(f"vanilla_plus_LLM_{self.model_name}_sequential")

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

            step_result = self.step()
            self.logger.info(f"Evaluation {idx+1}: LTL Score = {step_result['ltl_score']} (iterations: {step_result['iterations']})")
            end_time = time()
            delta_time = end_time - start_time
            results_list.append({
                "LTL_Score": step_result["ltl_score"],
                "Prism_Probabilities": self.prism_probs.copy(),
                "Evaluation_Time": delta_time,
                "Iterations_Used": step_result["iterations"],
                "Iteration_Times": step_result["iteration_times"],
                "Iteration_PRISM_Times": step_result["iteration_prism_times"],
                "Iteration_LLM_Times": step_result["iteration_llm_times"],
                "Iteration_Prism_Probs": step_result["iteration_prism_probs"],
                "Iteration_Mistakes": step_result["iteration_mistakes"],
                "Iteration_Costs": step_result["iteration_costs"],
                "Success": step_result["success"]
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
        run_dir = create_run_directory(f"vanilla_plus_LLM_{self.model_name}_parallel")

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
            'max_attempts': self.max_attempts,
            'target_threshold': self.target_threshold,
            'run_dir': run_dir,  # Pass shared directory to workers
            'model': self.model,
            'model_name': self.model_name
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
                executor.submit(_evaluate_single_gridworld_vanilla_plus, args): args[0]
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
                        "Iteration_Times": [],
                        "Iteration_PRISM_Times": [],
                        "Iteration_LLM_Times": [],
                        "Iteration_Prism_Probs": [],
                        "Iteration_Mistakes": [],
                        "Iteration_Costs": [],
                        "Success": False,
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
                "Iterations_Used": 0,
                "Iteration_Times": [],
                "Iteration_PRISM_Times": [],
                "Iteration_LLM_Times": [],
                "Iteration_Prism_Probs": [],
                "Iteration_Mistakes": [],
                "Iteration_Costs": [],
                "Success": False
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
        """Initialize policy table with default action (DOWN toward typical goal)"""
        policy = {}

        # Generate all combinations of goal states
        goal_combinations = list(product([False, True], repeat=self.num_goals))

        for x in range(self.size):
            for y in range(self.size):
                for goal_combo in goal_combinations:
                    state = (x, y) + goal_combo
                    policy[state] = 2  # Default: DOWN

        self.logger.info(f"Policy initialized with {len(policy)} states.")
        return policy

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

    def _check_all_probabilities_meet_threshold(self) -> bool:
        """Check if all probabilities meet the target threshold"""
        if not self.prism_probs:
            return False
        return all(prob >= self.target_threshold for prob in self.prism_probs.values())

    def _invoke_llm_for_goal(self, goal_num: int, idx: int) -> Tuple[int, int, ActionPolicy]:
        """Invoke LLM for a single goal. Returns (goal_num, idx, response)."""
        goal = self.env.goals[goal_num]
        goal_nums = sorted(self.env.goals.keys())
        future_goals = [self.env.goals[k] for k in goal_nums if k > goal_num]

        self.logger.info(f"LLM planning for goal {goal_num} at position {goal}")
        try:
            response = self.model.invoke(
                get_prompt(
                    self.size,
                    self.env.static_obstacles,
                    future_goals,
                    self.env.moving_obstacle_positions,
                    goal,
                    self.env.prob_forward,
                    self.env.prob_slip_left,
                    self.env.prob_slip_right
                )
            )
        except Exception as e:
            self.logger.error(f"LLM invoke failed for goal {goal_num}: {type(e).__name__}: {e}")
            raise
        self.logger.info(f"LLM Response received for goal {goal_num}.")
        return goal_num, idx, response

    def _apply_response(self, idx: int, response: ActionPolicy):
        """Apply LLM response to policy table."""
        applicable_goals = self._get_applicable_goal_states(idx)

        for state_action in response.states:
            x, y = state_action.x, state_action.y

            for goal_state_list in applicable_goals:
                goal_state_tuple = tuple(goal_state_list)
                state = (x, y) + goal_state_tuple

                if state in self.q_table:
                    self.q_table[state] = state_action.best_action
                else:
                    self.logger.warning(f"State {state} from LLM not in policy.")

    def step(self) -> Dict:
        """Plan for all goals with multiple attempts (no feedback).

        Each iteration resets the Q-table and calls the LLM fresh with the initial
        prompt. Unlike FeedbackLLMPlanner, no information about previous attempts
        is provided to the LLM.

        Returns:
            Dict containing:
            - ltl_score: Final LTL verification score
            - iterations: Number of iterations used
            - iteration_times: List of total time per iteration
            - iteration_prism_times: List of PRISM time per iteration
            - iteration_llm_times: List of LLM time per iteration
            - iteration_prism_probs: List of full probability dict per iteration
            - iteration_mistakes: List of mistake counts per iteration
            - iteration_costs: List of cost values per iteration
            - success: True if all probabilities meet target_threshold
        """
        goal_nums = sorted(self.env.goals.keys())

        # Per-iteration tracking
        iteration_times = []
        iteration_prism_times = []
        iteration_llm_times = []
        iteration_prism_probs = []
        iteration_mistakes = []
        iteration_costs = []

        ltl_score = 0.0
        attempt = 0

        while attempt < self.max_attempts:
            attempt += 1
            iteration_start = time()

            # Reset Q-table for fresh start each iteration
            self.q_table = self.initialize_q_table()
            self.prism_probs = {}

            self.logger.info(f"=== Attempt {attempt}/{self.max_attempts} (fresh start, no feedback) ===")

            # Process goals sequentially
            llm_start = time()
            for idx, goal_num in enumerate(goal_nums):
                goal_num, idx, response = self._invoke_llm_for_goal(goal_num, idx)
                self.logger.info(f"Applying response for goal {goal_num}")
                self._apply_response(idx, response)
            llm_time = time() - llm_start

            # Time PRISM verification
            prism_start = time()
            model_str = self.model_generator.generate_prism_model(self.q_table)
            ltl_score = self.simplified_verifier.verify_policy(model_str)
            prism_time = time() - prism_start

            # Update probabilities from verification
            if self.simplified_verifier.ltl_probabilities:
                self._update_prism_probabilities(self.simplified_verifier.ltl_probabilities[-1])

            # Compute metrics
            mistakes = sum(1 for p in self.prism_probs.values() if p < self.target_threshold)
            cost = sum(1.0 - p for p in self.prism_probs.values() if p < 1.0)
            iteration_time = time() - iteration_start

            # Track metrics for this iteration
            iteration_times.append(iteration_time)
            iteration_prism_times.append(prism_time)
            iteration_llm_times.append(llm_time)
            iteration_prism_probs.append(self.prism_probs.copy())
            iteration_mistakes.append(mistakes)
            iteration_costs.append(cost)

            self.logger.info(f"LTL Score after attempt {attempt}: {ltl_score}")
            self.logger.info(f"Current probabilities: {self.prism_probs}")

            # Early exit if all probabilities meet threshold
            if self._check_all_probabilities_meet_threshold():
                self.logger.info(f"All probabilities meet threshold after {attempt} attempts")
                break

        success = self._check_all_probabilities_meet_threshold()
        if not success:
            self.logger.warning(f"Max attempts ({self.max_attempts}) reached. Final probabilities: {self.prism_probs}")

        self.logger.info(f"Final LTL Score (Vanilla Plus LLM): {ltl_score}")
        self.logger.info(f"Total PRISM time: {sum(iteration_prism_times):.2f}s, Total LLM time: {sum(iteration_llm_times):.2f}s")

        return {
            "ltl_score": ltl_score,
            "iterations": attempt,
            "iteration_times": iteration_times,
            "iteration_prism_times": iteration_prism_times,
            "iteration_llm_times": iteration_llm_times,
            "iteration_prism_probs": iteration_prism_probs,
            "iteration_mistakes": iteration_mistakes,
            "iteration_costs": iteration_costs,
            "success": success
        }
