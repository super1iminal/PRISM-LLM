from verification.PrismVerifier import PrismVerifier
from verification.SimplifiedVerifier import SimplifiedVerifier
from verification.PrismModelGenerator import PrismModelGenerator
from utils.LLMPrompting import get_prompt, build_prompt, ActionPolicy, generate_policy_visual, identify_problems, format_probability_summary, extract_segment_probs

import os
from typing import List, Optional, Dict, Tuple
from itertools import product
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import traceback

from time import time

from config.Settings import PRISM_PATH, get_threshold_for_key
from utils.Logging import setup_logger, create_run_directory


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
    model = config.get('model')
    model_name = config.get('model_name')

    # Create a new planner instance for this worker
    planner = FeedbackLLMPlanner(
        model=model,
        model_name=model_name,
        max_attempts=config.get('max_attempts', 3)
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


class FeedbackLLMPlanner:
    def __init__(self, model, model_name: str, max_attempts: int = 3):
        self.action_space = 4
        self.prism_probs = {}
        self.max_attempts = max_attempts  # Total LLM attempts per goal (1 initial + N-1 feedback)
        self.model = model
        self.model_name = model_name
        
    def evaluate(self, dataloader, max_workers: Optional[int] = None, use_threads: bool = True, run_dir: Optional[str] = None):
        """
        Evaluate the Feedback LLM planner on all gridworlds in the dataloader.

        Args:
            dataloader: DataLoader instance containing GridWorld configurations
            max_workers: Maximum number of workers. If None, uses CPU count for processes
                        or min(32, CPU count + 4) for threads.
            use_threads: If True, use ThreadPoolExecutor instead of
                        ProcessPoolExecutor. Threads are recommended for I/O-bound LLM calls.
            run_dir: Optional shared parent directory for logs. If None, creates standalone directory.

        Returns:
            List of results containing LTL scores and evaluation statistics
        """
        return self._evaluate_parallel(dataloader, max_workers, use_threads, run_dir)

    def _evaluate_parallel(self, dataloader, max_workers: Optional[int] = None,
                          use_threads: bool = True, run_dir: Optional[str] = None) -> List[Dict]:
        """
        Parallel evaluation of all gridworlds.

        Args:
            dataloader: DataLoader instance containing GridWorld configurations
            max_workers: Maximum number of workers
            use_threads: If True, use ThreadPoolExecutor (better for I/O-bound LLM calls)
            run_dir: Optional shared parent directory for logs.

        Returns:
            List of results containing LTL scores and evaluation statistics
        """
        # Create log directory under shared run_dir, or standalone
        if run_dir:
            log_dir = os.path.join(run_dir, f"feedback_LLM_{self.model_name}")
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_dir = create_run_directory(f"feedback_LLM_{self.model_name}_parallel")

        # Setup main logger in the shared directory
        main_logger = setup_logger("main", run_dir=log_dir, include_timestamp=False)
        
        if max_workers is None:
            if use_threads:
                max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)
            else:
                max_workers = multiprocessing.cpu_count()
        
        executor_type = "threads" if use_threads else "processes"
        main_logger.info(f"Starting parallel evaluation with {max_workers} {executor_type}")
        main_logger.info(f"Logs will be saved to: {log_dir}")

        # Prepare arguments with shared run directory
        config = {
            'max_attempts': self.max_attempts,
            'run_dir': log_dir,  # Pass shared directory to workers
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
        prism_path = PRISM_PATH
        if os.path.exists(prism_path) and os.access(prism_path, os.X_OK):
            return prism_path
        else:
            raise FileNotFoundError(f"PRISM executable not found or not executable at {prism_path}")
        
    def initialize_q_table(self):
        """Initialize policy table with default action (DOWN toward typical goal)"""
        policy = {}
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
        for goal_num in goal_nums:
            if idx < len(probabilities):
                self.prism_probs[f'goal{goal_num}'] = probabilities[idx]
                idx += 1
        
        for i in range(len(goal_nums) - 1):
            if idx < len(probabilities):
                self.prism_probs[f'seq_{goal_nums[i]}_before_{goal_nums[i+1]}'] = probabilities[idx]
                idx += 1
        
        # Complete sequence (only emitted when multiple goals)
        if len(goal_nums) > 1 and idx < len(probabilities):
            self.prism_probs['complete_sequence'] = probabilities[idx]
            idx += 1

        # Per-segment moving obstacle avoidance
        if self.env.moving_obstacle_positions:
            for goal_num in goal_nums:
                if idx < len(probabilities):
                    self.prism_probs[f'avoid_moving_seg{goal_num}'] = probabilities[idx]
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
        """Check if all probabilities meet the per-key thresholds"""
        if not self.prism_probs:
            return False
        return all(p >= get_threshold_for_key(k) for k, p in self.prism_probs.items())

    def _goals_needing_requery(self) -> List[int]:
        """Return indices of goals whose segment probabilities are below threshold."""
        goal_nums = sorted(self.env.goals.keys())
        needs_requery = []
        for idx, goal_num in enumerate(goal_nums):
            segment_probs = extract_segment_probs(self.prism_probs, goal_num, goal_nums)
            if any(p < get_threshold_for_key(k) for k, p in segment_probs.items()):
                needs_requery.append(idx)
        return needs_requery

    def _get_feedback_prompt(self, goal_idx: int, goal: Tuple[int, int],
                              future_goals: List[Tuple[int, int]]) -> str:
        """Generate feedback prompt for correction iteration with per-segment feedback"""

        goal_nums = sorted(self.env.goals.keys())
        goal_num = goal_nums[goal_idx]

        # Get the applicable goal state for this goal index
        goal_state_list = self._get_applicable_goal_states(goal_idx)[0]
        goal_state = tuple(goal_state_list)

        policy_vis = generate_policy_visual(
            self.size, self.q_table, goal_state,
            goal, self.env.static_obstacles, future_goals
        )

        # Per-segment probs for this goal + global metrics
        segment_probs = extract_segment_probs(self.prism_probs, goal_num, goal_nums)
        if 'complete_sequence' in self.prism_probs:
            segment_probs['complete_sequence'] = self.prism_probs['complete_sequence']

        prob_summary = format_probability_summary(segment_probs)
        problems = identify_problems(segment_probs)

        return build_prompt(
            size=self.size,
            s_obstacles=self.env.static_obstacles,
            f_goals=future_goals,
            k_obstacles=self.env.moving_obstacle_positions,
            goal=goal,
            prob_forward=self.env.prob_forward,
            prob_slip_left=self.env.prob_slip_left,
            prob_slip_right=self.env.prob_slip_right,
            is_feedback=True,
            probability_summary=prob_summary,
            policy_visual=policy_vis,
            problems=problems
        )
    
    def _apply_response_to_q_table(self, response: ActionPolicy, goal_idx: int):
        """Apply LLM response to policy table"""
        applicable_goals = self._get_applicable_goal_states(goal_idx)

        for state_action in response.states:
            x, y = state_action.x, state_action.y

            if state_action.best_action not in (0, 1, 2, 3):
                self.logger.warning(f"Invalid action {state_action.best_action} at ({x},{y}), skipping.")
                continue

            for goal_state_list in applicable_goals:
                goal_state_tuple = tuple(goal_state_list)
                state = (x, y) + goal_state_tuple

                if state in self.q_table:
                    self.logger.info(f"Setting action for state {state} to {state_action.best_action}")
                    self.q_table[state] = state_action.best_action
                else:
                    self.logger.warning(f"State {state} from LLM not in policy.")
    
    def _verify_current_policy(self) -> float:
        """Verify current policy and update probabilities"""
        model_str = self.model_generator.generate_prism_model(self.q_table)
        ltl_score = self.simplified_verifier.verify_policy(model_str)

        if self.simplified_verifier.ltl_probabilities:
            self._update_prism_probabilities(self.simplified_verifier.ltl_probabilities[-1])

        return ltl_score

    def step(self) -> Dict:
        """Main planning step with feedback loop

        Returns:
            Dictionary containing:
            - ltl_score: Final LTL verification score
            - iterations: Number of iterations used (1 = initial only)
            - iteration_times: List of total time per iteration
            - iteration_prism_times: List of PRISM time per iteration
            - iteration_llm_times: List of LLM time per iteration
            - iteration_prism_probs: List of full probability dicts per iteration
            - iteration_mistakes: List of mistake counts per iteration
            - iteration_costs: List of cost values per iteration
            - success: Boolean indicating if all probabilities meet threshold
        """
        goal_nums = sorted(self.env.goals.keys())

        # Per-iteration tracking
        iteration_times = []        # Total time per iteration
        iteration_prism_times = []  # PRISM time per iteration
        iteration_llm_times = []    # LLM time per iteration
        iteration_prism_probs = []  # Full probability dict per iteration
        iteration_mistakes = []     # Mistakes per iteration
        iteration_costs = []        # Cost per iteration

        # Store responses for potential feedback iterations
        goal_responses: Dict[int, ActionPolicy] = {}

        # Initial pass - get Q-values for each goal (iteration 1)
        iteration_start = time()
        iter_llm_time = 0.0

        for idx, goal_num in enumerate(goal_nums):
            goal = self.env.goals[goal_num]
            self.logger.info(f"Planning for goal {goal_num} at position {goal}")

            future_goals = [self.env.goals[k] for k in goal_nums if k > goal_num]

            self.logger.info(f"Calling LLM for goal {goal_num}...")
            llm_start = time()
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
            iter_llm_time += time() - llm_start
            self.logger.info("LLM Response received.")
            self.logger.info(response.states)

            goal_responses[goal_num] = response
            self._apply_response_to_q_table(response, idx)

        # Initial verification
        prism_start = time()
        ltl_score = self._verify_current_policy()
        iter_prism_time = time() - prism_start
        self.logger.info(f"Initial LTL Score: {ltl_score}")

        # Track metrics for initial iteration
        mistakes = sum(1 for k, p in self.prism_probs.items() if p < get_threshold_for_key(k))
        cost = sum(1.0 - p for p in self.prism_probs.values() if p < 1.0)
        iteration_times.append(time() - iteration_start)
        iteration_prism_times.append(iter_prism_time)
        iteration_llm_times.append(iter_llm_time)
        iteration_prism_probs.append(self.prism_probs.copy())
        iteration_mistakes.append(mistakes)
        iteration_costs.append(cost)
        failed = {k: f"{p:.4f} < {get_threshold_for_key(k)}" for k, p in self.prism_probs.items() if p < get_threshold_for_key(k)}
        if failed:
            self.logger.info(f"Failed requirements ({mistakes}): {failed}")

        # Track best policy to prevent over-correction
        best_ltl_score = ltl_score
        best_q_table = self.q_table.copy()
        best_prism_probs = self.prism_probs.copy()

        # Feedback loop - attempt counts: 1 = initial, 2+ = feedback iterations
        attempt = 1  # We've completed attempt 1 (initial)
        while attempt < self.max_attempts and not self._check_all_probabilities_meet_threshold():
            attempt += 1
            iteration_start = time()
            iter_llm_time = 0.0
            self.logger.info(f"=== Attempt {attempt}/{self.max_attempts} (feedback) ===")
            self.logger.info(f"Current probabilities: {self.prism_probs}")

            # Selective re-query: only goals with below-threshold segments
            goals_to_requery = self._goals_needing_requery()
            self.logger.info(f"Goals needing re-query: {[sorted(self.env.goals.keys())[i] for i in goals_to_requery]}")

            for idx, goal_num in enumerate(goal_nums):
                if idx not in goals_to_requery:
                    self.logger.info(f"Skipping goal {goal_num} (all segment probs meet threshold)")
                    continue

                goal = self.env.goals[goal_num]
                future_goals = [self.env.goals[k] for k in goal_nums if k > goal_num]

                self.logger.info(f"Re-planning goal {goal_num} with feedback")

                feedback_prompt = self._get_feedback_prompt(
                    idx, goal, future_goals
                )

                llm_start = time()
                try:
                    response = self.model.invoke(feedback_prompt)
                except Exception as e:
                    self.logger.error(f"Feedback LLM invoke failed for goal {goal_num} (attempt {attempt}): {type(e).__name__}: {e}")
                    raise
                iter_llm_time += time() - llm_start
                self.logger.info("Feedback LLM Response received.")

                goal_responses[goal_num] = response
                self._apply_response_to_q_table(response, idx)

            # Re-verify
            prism_start = time()
            ltl_score = self._verify_current_policy()
            iter_prism_time = time() - prism_start
            self.logger.info(f"LTL Score after attempt {attempt}: {ltl_score}")

            # Track metrics for this iteration
            mistakes = sum(1 for k, p in self.prism_probs.items() if p < get_threshold_for_key(k))
            cost = sum(1.0 - p for p in self.prism_probs.values() if p < 1.0)
            iteration_times.append(time() - iteration_start)
            iteration_prism_times.append(iter_prism_time)
            iteration_llm_times.append(iter_llm_time)
            iteration_prism_probs.append(self.prism_probs.copy())
            iteration_mistakes.append(mistakes)
            iteration_costs.append(cost)
            failed = {k: f"{p:.4f} < {get_threshold_for_key(k)}" for k, p in self.prism_probs.items() if p < get_threshold_for_key(k)}
            if failed:
                self.logger.info(f"Failed requirements ({mistakes}): {failed}")

            # Keep-best with rollback: revert if score didn't improve
            if ltl_score > best_ltl_score:
                best_ltl_score = ltl_score
                best_q_table = self.q_table.copy()
                best_prism_probs = self.prism_probs.copy()
                self.logger.info(f"New best score: {best_ltl_score:.4f}")
            else:
                self.logger.info(f"Score {ltl_score:.4f} did not improve over best {best_ltl_score:.4f}, reverting policy")
                self.q_table = best_q_table.copy()
                self.prism_probs = best_prism_probs.copy()

        # Ensure we return the best results
        self.q_table = best_q_table
        self.prism_probs = best_prism_probs

        success = self._check_all_probabilities_meet_threshold()
        if success:
            self.logger.info(f"All probabilities meet threshold after {attempt} attempts")
        else:
            self.logger.warning(f"Max attempts ({self.max_attempts}) reached. Final probabilities: {self.prism_probs}")

        self.logger.info(f"Final LTL Score (Feedback LLM): {best_ltl_score}")
        self.logger.info(f"Total PRISM time: {sum(iteration_prism_times):.2f}s, Total LLM time: {sum(iteration_llm_times):.2f}s")

        return {
            "ltl_score": best_ltl_score,
            "iterations": attempt,
            "iteration_times": iteration_times,
            "iteration_prism_times": iteration_prism_times,
            "iteration_llm_times": iteration_llm_times,
            "iteration_prism_probs": iteration_prism_probs,
            "iteration_mistakes": iteration_mistakes,
            "iteration_costs": iteration_costs,
            "success": success
        }