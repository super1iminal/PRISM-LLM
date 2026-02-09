import numpy as np
import os
from typing import Dict, Tuple, List, Optional
from itertools import product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from verification.PrismVerifier import PrismVerifier
from verification.SimplifiedVerifier import SimplifiedVerifier
from verification.PrismModelGenerator import PrismModelGenerator
from environment.GridWorld import GridWorld

from environment.GridWorldStepper import GridWorldMover
from config.Settings import PRISM_PATH, RL_MAX_EPISODES, RL_CONVERGENCE_EPSILON, RL_MIN_EPISODES_BEFORE_CONVERGENCE, get_threshold_for_key
from utils.Logging import setup_logger, create_run_directory


def _evaluate_single_gridworld(args: Tuple) -> Dict:
    """
    Standalone function to evaluate a single gridworld.
    This function is designed to be called in a separate process.
    
    Args:
        args: Tuple of (idx, gridworld, expected_steps, config)
        
    Returns:
        Dictionary containing evaluation results
    """
    idx, gridworld, expected_steps, config = args
    
    # Get shared run directory from config
    run_dir = config.get('run_dir', None)
    
    # Create a new agent instance for this process
    agent = LTLGuidedQLearningWithObstacle()
    agent.max_episodes = config.get('max_episodes', RL_MAX_EPISODES)
    agent.convergence_epsilon = config.get('convergence_epsilon', RL_CONVERGENCE_EPSILON)
    agent.min_episodes_before_convergence = config.get('min_episodes_before_convergence', RL_MIN_EPISODES_BEFORE_CONVERGENCE)
    agent.verify_interval = config.get('verify_interval', 1)
    
    # Setup logger for this process in the shared directory
    agent.logger = setup_logger(f"worker_{idx}", run_dir=run_dir, include_timestamp=False)
    agent.prism_verifier = PrismVerifier(agent._get_prism_path(), agent.logger)
    
    agent.logger.info(f"\n{'='*50}")
    agent.logger.info(f"Evaluating gridworld {idx+1}")
    agent.logger.info(f"Size: {gridworld.size}, Goals: {gridworld.goals}")
    agent.logger.info(f"Static obstacles: {gridworld.static_obstacles}")
    agent.logger.info(f"Expected BFS steps: {expected_steps}")
    agent.logger.info(f"{'='*50}\n")
    
    start_time = time()
    
    # Initialize for this gridworld
    agent._init_for_gridworld(gridworld)
    
    # Train the agent
    best_policy, best_ltl_score = agent.train(
        verify_interval=agent.verify_interval
    )
    
    end_time = time()
    delta_time = end_time - start_time
    
    agent.logger.info(f"Evaluation {idx+1}: Best LTL Score = {best_ltl_score:.4f}")
    agent.logger.info(f"Training time: {delta_time:.2f} seconds")
    
    return {
        "index": idx,
        "LTL_Score": best_ltl_score,
        "Prism_Probabilities": agent.prism_probs.copy(),
        "Evaluation_Time": delta_time,
        "Iterations_Used": 1,
        "Iteration_Times": [delta_time],
        "Iteration_PRISM_Times": [0.0],
        "Iteration_LLM_Times": [0.0],
        "Iteration_Prism_Probs": [agent.prism_probs.copy()],
        "Iteration_Mistakes": [sum(1 for k, p in agent.prism_probs.items() if p < get_threshold_for_key(k))],
        "Iteration_Costs": [sum(1.0 - p for p in agent.prism_probs.values() if p < 1.0)],
        "Success": all(p >= get_threshold_for_key(k) for k, p in agent.prism_probs.items()),
        "Episode_Rewards": agent.episode_rewards.copy(),
        "Training_Stats": agent.training_stats.copy()
    }


class LTLGuidedQLearningWithObstacle:
    """
    Q-Learning agent guided by LTL verification through PRISM model checking.
    Uses counterfactual learning to improve sample efficiency.
    """
    
    def __init__(self):
        self.action_space = 4
        self.prism_probs = {}
        
        # Hyperparameters
        self.epsilon = 0.9
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.997
        self.alpha = 0.2
        self.alpha_min = 0.05
        self.gamma = 0.99
        self.K = 5  # Counterfactual depth
        
        # Training configuration
        self.max_episodes = RL_MAX_EPISODES  # 500
        self.convergence_epsilon = RL_CONVERGENCE_EPSILON  # 0.001
        self.min_episodes_before_convergence = RL_MIN_EPISODES_BEFORE_CONVERGENCE  # 50
        self.verify_interval = 1
        self.previous_probs = None
        
    def _get_prism_path(self):
        """Get PRISM executable path"""
        if os.path.exists(PRISM_PATH) and os.access(PRISM_PATH, os.X_OK):
            return PRISM_PATH
        else:
            raise FileNotFoundError(f"PRISM executable not found or not executable at {PRISM_PATH}")
    
    def evaluate(self, dataloader, max_workers: Optional[int] = None, run_dir: Optional[str] = None):
        """
        Evaluate the RL planner on all gridworlds in the dataloader.

        Args:
            dataloader: DataLoader instance containing GridWorld configurations
            max_workers: Maximum number of worker processes. If None, uses CPU count.
            run_dir: Optional shared parent directory for logs. If None, creates standalone directory.

        Returns:
            List of results containing LTL scores and training statistics
        """
        return self._evaluate_parallel(dataloader, max_workers, run_dir)

    def _evaluate_parallel(self, dataloader, max_workers: Optional[int] = None, run_dir: Optional[str] = None) -> List[Dict]:
        """
        Parallel evaluation of all gridworlds using ProcessPoolExecutor.

        Args:
            dataloader: DataLoader instance containing GridWorld configurations
            max_workers: Maximum number of worker processes. If None, uses CPU count.
            run_dir: Optional shared parent directory for logs.

        Returns:
            List of results containing LTL scores and training statistics
        """
        # Create log directory under shared run_dir, or standalone
        if run_dir:
            log_dir = os.path.join(run_dir, "RL_counterfactual")
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_dir = create_run_directory("RL_counterfactual_parallel")

        # Setup main logger in the shared directory
        main_logger = setup_logger("main", run_dir=log_dir, include_timestamp=False)
        
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        main_logger.info(f"Starting parallel evaluation with {max_workers} workers")
        main_logger.info(f"Logs will be saved to: {log_dir}")

        # Prepare arguments for parallel execution
        config = {
            'max_episodes': self.max_episodes,
            'convergence_epsilon': self.convergence_epsilon,
            'min_episodes_before_convergence': self.min_episodes_before_convergence,
            'verify_interval': self.verify_interval,
            'run_dir': log_dir  # Pass shared directory to workers
        }
        
        # Convert dataloader to list for indexing
        data_list = list(dataloader)
        args_list = [
            (idx, gridworld, expected_steps, config)
            for idx, (gridworld, expected_steps) in enumerate(data_list)
        ]
        
        results_dict = {}
        total_start_time = time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(_evaluate_single_gridworld, args): args[0]
                for args in args_list
            }
            
            # Collect results as they complete
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
                        "Episode_Rewards": [],
                        "Training_Stats": [],
                        "error": str(e)
                    }
        
        total_time = time() - total_start_time
        main_logger.info(f"Total parallel evaluation time: {total_time:.2f} seconds")
        
        # Sort results by original index and remove index field
        results_list = []
        for idx in range(len(args_list)):
            result = results_dict.get(idx, {
                "LTL_Score": 0.0,
                "Prism_Probabilities": {},
                "Evaluation_Time": 0.0,
                "Episode_Rewards": [],
                "Training_Stats": []
            })
            # Remove index from result
            result.pop("index", None)
            results_list.append(result)
        
        return results_list
    
    def _init_for_gridworld(self, gridworld: GridWorld):
        """Initialize all components for a specific gridworld"""
        self.env = gridworld
        self.size = gridworld.size
        self.num_goals = len(gridworld.goals)
        self.stepper = GridWorldMover(gridworld)
        self.model_generator = PrismModelGenerator(gridworld, self.logger)
        self.simplified_verifier = SimplifiedVerifier(self.prism_verifier, gridworld, self.logger)
        
        # Initialize Q-table
        self.q_table = self._initialize_q_table()
        
        # Reset hyperparameters for new training run
        self.epsilon = 0.9
        self.alpha = 0.2
        
        # Initialize PRISM probabilities
        self.prism_probs = self._init_prism_probs()
        self.previous_probs = None  # Reset for convergence tracking

        # Performance tracking
        self.episode_rewards = []
        self.ltl_scores = []
        self.best_policy = None
        self.best_ltl_score = 0.0
        self.training_stats = []
    
    def _init_prism_probs(self) -> Dict[str, float]:
        """Initialize PRISM probability tracking dict based on number of goals"""
        probs = {}
        for i in range(1, self.num_goals + 1):
            probs[f'goal{i}'] = 0.0

        # Sequence probabilities
        for i in range(1, self.num_goals):
            probs[f'seq_{i}_before_{i+1}'] = 0.0

        probs['complete_sequence'] = 0.0

        # Per-segment moving obstacle avoidance
        if self.env.moving_obstacle_positions:
            for i in range(1, self.num_goals + 1):
                probs[f'avoid_moving_seg{i}'] = 0.0

        return probs
    
    def _initialize_q_table(self) -> Dict:
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
        
        # Complete sequence (only emitted when multiple goals)
        if len(goal_nums) > 1 and idx < len(probabilities):
            self.prism_probs['complete_sequence'] = probabilities[idx]
            idx += 1

        # Per-segment moving obstacle avoidance
        if self.env.moving_obstacle_positions:
            goal_nums_local = sorted(self.env.goals.keys())
            for goal_num in goal_nums_local:
                if idx < len(probabilities):
                    self.prism_probs[f'avoid_moving_seg{goal_num}'] = probabilities[idx]
                    idx += 1

    def _calculate_sequence_weight(self, state: tuple, next_state: tuple) -> float:
        """Calculate weight based on goal sequence progress"""
        # Extract goal flags from states
        goal_flags = state[2:]
        next_goal_flags = next_state[2:]
        
        # Get sorted goal numbers to map indices to actual goal numbers
        goal_nums = sorted(self.env.goals.keys())
        
        weight = 1.0
        
        for i in range(self.num_goals):
            # Check if this goal was just reached
            if not goal_flags[i] and next_goal_flags[i]:
                # Verify previous goals were already reached (correct sequence)
                prev_reached = all(goal_flags[j] for j in range(i))
                
                if prev_reached:
                    if i < self.num_goals - 1:
                        # Not the final goal - use sequence probability
                        current_goal = goal_nums[i]
                        next_goal = goal_nums[i + 1]
                        seq_prob = self.prism_probs.get(f'seq_{current_goal}_before_{next_goal}', 0.0)
                        weight *= (1.5 + seq_prob)
                    else:
                        # Final goal reached - use complete sequence probability
                        weight *= (1.5 + self.prism_probs.get('complete_sequence', 0.0))
        
        return weight

    def _calculate_counterfactual_reward(self, state: tuple, next_state: tuple, k: int) -> float:
        """Calculate reward with counterfactual reasoning and PRISM guidance"""
        x, y = state[0], state[1]
        next_x, next_y = next_state[0], next_state[1]
        goal_flags = state[2:]
        next_goal_flags = next_state[2:]
        
        # Get sorted goal numbers
        goal_nums = sorted(self.env.goals.keys())
        
        base_reward = -0.05 * (1 + 0.05 * self.size / 6)
        
        sequence_weight = self._calculate_sequence_weight(state, next_state)
        k_bonus = 1.0 + (k / self.K) * sequence_weight
        # Use average of per-segment avoidance probs, or 0.0 if none
        avoid_probs = [v for k, v in self.prism_probs.items() if k.startswith('avoid_moving_seg')]
        obstacle_prob = sum(avoid_probs) / len(avoid_probs) if avoid_probs else 0.0
        
        # Check obstacle collision
        if (next_x, next_y) in self.env.static_obstacles:
            obstacle_penalty = -10.0 * (1.0 + 0.2 * obstacle_prob)
            base_reward += obstacle_penalty
        else:
            # Reward for staying away from obstacles
            if self.env.static_obstacles:
                min_dist = float('inf')
                for obs in self.env.static_obstacles:
                    dist = abs(next_x - obs[0]) + abs(next_y - obs[1])
                    min_dist = min(min_dist, dist)
                
                if min_dist < 3:
                    avoidance_bonus = 1.0 * obstacle_prob * (1 / max(1, min_dist))
                    base_reward += avoidance_bonus
        
        # Determine current goal based on progress (use actual goal numbers)
        current_goal_idx = sum(1 for g in goal_flags if g)
        if current_goal_idx < self.num_goals:
            current_goal_num = goal_nums[current_goal_idx]
            current_goal = self.env.goals.get(current_goal_num)
            if current_goal:
                dist_to_goal = abs(next_x - current_goal[0]) + abs(next_y - current_goal[1])
                progress_reward = -0.005 * dist_to_goal
                base_reward += progress_reward
        
        # Goal achievement rewards
        for i, goal_num in enumerate(goal_nums):
            goal_pos = self.env.goals[goal_num]
            # Check if we just reached this goal
            if not goal_flags[i] and next_goal_flags[i]:
                if next_x == goal_pos[0] and next_y == goal_pos[1]:
                    # Check if it's the correct order
                    prev_goals_reached = all(goal_flags[j] for j in range(i))
                    if prev_goals_reached:
                        goal_prob = self.prism_probs.get(f'goal{goal_num}', 0.0)
                        if i == self.num_goals - 1:
                            # Final goal - big reward
                            base_reward += 1000.0 * k_bonus * (1 + goal_prob)
                        else:
                            base_reward += 100.0 * k_bonus * (1 + goal_prob)
        
        return base_reward

    def _get_counterfactual_states(self, state: tuple, next_state: tuple) -> List[Tuple]:
        """Generate counterfactual state transitions for learning"""
        x, y = state[0], state[1]
        next_x, next_y = next_state[0], next_state[1]
        goal_flags = list(state[2:])
        next_goal_flags = list(next_state[2:])
        
        # Get sorted goal numbers
        goal_nums = sorted(self.env.goals.keys())
        
        counterfactuals = []
        
        # Only generate counterfactuals with some probability
        if np.random.random() > 0.3:
            return counterfactuals
        
        max_counterfactuals = 2
        
        # Find current goal index (first unreached goal)
        current_goal_idx = None
        for i in range(self.num_goals):
            if not goal_flags[i]:
                current_goal_idx = i
                break
        
        if current_goal_idx is None:
            return counterfactuals
        
        # Check if we're close to the current goal (use actual goal number)
        goal_num = goal_nums[current_goal_idx]
        goal_pos = self.env.goals.get(goal_num)
        if goal_pos:
            dist_to_goal = abs(next_x - goal_pos[0]) + abs(next_y - goal_pos[1])
            
            if dist_to_goal <= 2 and not next_goal_flags[current_goal_idx]:
                # Create counterfactual where we reached the goal
                cf_goal_flags = next_goal_flags.copy()
                cf_goal_flags[current_goal_idx] = True
                
                cf_state = state
                cf_next_state = (next_x, next_y) + tuple(cf_goal_flags)
                counterfactuals.append((cf_state, cf_next_state))
        
        return counterfactuals[:max_counterfactuals]

    def _step_with_counterfactuals(self, state: tuple, action: int, k: int) -> Tuple[tuple, list, bool, int]:
        """Take a step and generate counterfactual experiences"""
        next_state, reward, done = self.stepper.step(action)
        experiences = [(state, action, next_state, reward)]
        
        counterfactuals = self._get_counterfactual_states(state, next_state)
        for cf_state, cf_next_state in counterfactuals:
            cf_reward = self._calculate_counterfactual_reward(cf_state, cf_next_state, k)
            if cf_state != state or cf_next_state != next_state:
                experiences.append((cf_state, action, cf_next_state, cf_reward))
        
        next_k = min(k + 1, self.K - 1) if reward > 0 else k
        
        return next_state, experiences, done, next_k

    def _get_action(self, state: tuple) -> int:
        """Select action using epsilon-greedy with goal-directed exploration"""
        if np.random.random() < self.epsilon:
            x, y = state[0], state[1]
            goal_flags = state[2:]
            
            # Find current target goal
            target = None
            goal_nums = sorted(self.env.goals.keys())
            for i, goal_num in enumerate(goal_nums):
                if not goal_flags[i]:
                    target = self.env.goals[goal_num]
                    break
            
            if target is None:
                return np.random.randint(self.action_space)
            
            dx = target[0] - x
            dy = target[1] - y
            
            # Prefer moving towards goal
            if abs(dx) > abs(dy):
                preferred_action = 2 if dx > 0 else 0  # DOWN or UP
            else:
                preferred_action = 1 if dy > 0 else 3  # RIGHT or LEFT
            
            if np.random.random() < 0.7:
                return preferred_action
            else:
                return np.random.randint(self.action_space)
        
        return int(np.argmax(self.q_table[state]))

    def _adjust_parameters(self, ltl_score: float):
        """Adjust learning parameters based on LTL score"""
        if ltl_score > self.best_ltl_score:
            self.best_ltl_score = ltl_score
            self.best_policy = self.get_current_policy()
            
            # Reduce exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.997)
            self.alpha = max(self.alpha_min, self.alpha * 0.998)
        else:
            # Increase exploration if not improving
            self.epsilon = min(0.9, self.epsilon * 1.02)
        
        self.training_stats.append({
            'ltl_score': ltl_score,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'best_score': self.best_ltl_score
        })

    def get_current_policy(self) -> Dict[tuple, int]:
        """Extract current greedy policy from Q-table"""
        return {state: int(np.argmax(q_values))
                for state, q_values in self.q_table.items()}

    def _check_termination(self, episode: int) -> Optional[str]:
        """Check termination conditions. Returns reason string or None."""
        # Condition 1: All probabilities meet threshold (can trigger anytime)
        if self._all_probs_meet_threshold():
            return "all probabilities >= threshold"

        # Condition 2: Convergence (only after minimum episodes to allow learning)
        if episode >= self.min_episodes_before_convergence:
            if self._check_convergence():
                return "probabilities converged (change < epsilon)"
        else:
            # Still track previous probs for when we do start checking
            self._update_previous_probs()

        return None

    def _update_previous_probs(self):
        """Update previous_probs without checking convergence."""
        self.previous_probs = self.prism_probs.copy()

    def _all_probs_meet_threshold(self) -> bool:
        """Check if all probabilities >= their per-key thresholds."""
        if not self.prism_probs:
            return False
        return all(p >= get_threshold_for_key(k)
                   for k, p in self.prism_probs.items())

    def _check_convergence(self) -> bool:
        """Check if probability changes are below epsilon (single check)."""
        if self.previous_probs is None:
            self.previous_probs = self.prism_probs.copy()
            return False

        # Calculate max change across all probabilities
        max_change = max(
            abs(self.prism_probs.get(k, 0) - self.previous_probs.get(k, 0))
            for k in self.prism_probs.keys()
        )

        self.previous_probs = self.prism_probs.copy()
        return max_change < self.convergence_epsilon

    def train(self, verify_interval: int = 1):
        """
        Train the Q-learning agent with PRISM verification guidance.

        Termination conditions (checked after each verification):
        1. All probabilities >= threshold
        2. Probability changes < epsilon (convergence)
        3. Max episodes reached

        Args:
            verify_interval: Frequency of PRISM verification

        Returns:
            Tuple of (best_policy, best_ltl_score)
        """
        try:
            start_time = time()
            best_episode = 0
            episode = 0

            while episode < self.max_episodes:
                # Periodic verification (before episode training)
                if episode % verify_interval == 0:
                    try:
                        model_str = self.model_generator.generate_prism_model(self.get_current_policy())
                        ltl_score = self.simplified_verifier.verify_policy(model_str)

                        if self.simplified_verifier.ltl_probabilities:
                            latest_probs = self.simplified_verifier.ltl_probabilities[-1]
                            self._update_prism_probabilities(latest_probs)

                        self.ltl_scores.append(ltl_score)
                        self._adjust_parameters(ltl_score)

                        self.logger.info(
                            f"Episode {episode}: LTL Score = {ltl_score:.4f}, "
                            f"Epsilon = {self.epsilon:.3f}, "
                            f"Alpha = {self.alpha:.3f}, "
                            f"Best Score = {self.best_ltl_score:.4f}, "
                            f"Time = {(time() - start_time)/60:.1f}m"
                        )

                        if ltl_score > self.best_ltl_score:
                            best_episode = episode

                        # Check termination conditions after verification
                        termination_reason = self._check_termination(episode)
                        if termination_reason:
                            self.logger.info(f"Early termination at episode {episode}: {termination_reason}")
                            break

                    except Exception as e:
                        self.logger.error(f"Verification error in episode {episode}: {str(e)}")
                        episode += 1
                        continue

                # Episode training loop
                state = self.stepper.reset()
                total_reward = 0
                done = False
                k = 0
                steps = 0

                while not done:
                    action = self._get_action(state)
                    next_state, experiences, done, next_k = self._step_with_counterfactuals(
                        state, action, k)

                    # Update Q-values from all experiences
                    for exp_state, exp_action, exp_next_state, exp_reward in experiences:
                        if exp_state in self.q_table and exp_next_state in self.q_table:
                            current_q = self.q_table[exp_state][exp_action]
                            next_max_q = np.max(self.q_table[exp_next_state])
                            new_q = current_q + self.alpha * (
                                exp_reward + self.gamma * next_max_q - current_q
                            )
                            self.q_table[exp_state][exp_action] = new_q

                    state = next_state
                    k = next_k
                    total_reward += experiences[0][3]
                    steps += 1

                    if steps >= self.stepper.max_steps:
                        done = True

                self.episode_rewards.append(total_reward)

                if episode % 10 == 0:
                    self.logger.info(f"Episode {episode}: Steps = {steps}, "
                                    f"Total Reward = {total_reward:.2f}")

                episode += 1
            
            # Final verification
            try:
                model_str = self.model_generator.generate_prism_model(self.get_current_policy())
                final_score = self.simplified_verifier.verify_policy(model_str)
                self.logger.info(f"\nFinal LTL Score: {final_score:.4f}")
            except Exception as e:
                self.logger.error(f"Final verification error: {str(e)}")
            
            training_time = time() - start_time
            self.logger.info(f"\nTraining completed in {training_time/60:.1f} minutes")
            self.logger.info(f"Best LTL Score: {self.best_ltl_score:.4f} "
                            f"(Episode {best_episode})")
            
            return self.best_policy, self.best_ltl_score
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def plot_training_progress(self, save_dir: str = "results"):
        """Plot and save training progress graphs"""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(2, 1, 1)
        episodes = range(len(self.episode_rewards))
        plt.plot(episodes, self.episode_rewards, 'b-', alpha=0.3, label='Rewards')
        
        window = min(10, len(self.episode_rewards))
        if len(self.episode_rewards) >= window:
            smoothed = np.convolve(self.episode_rewards,
                                   np.ones(window)/window,
                                   mode='valid')
            plt.plot(range(window-1, len(self.episode_rewards)),
                    smoothed, 'r-', label=f'Moving Average (w={window})')
        
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)
        
        # Plot LTL scores and parameters
        plt.subplot(2, 1, 2)
        verify_episodes = range(len(self.training_stats))
        
        plt.plot(verify_episodes,
                [s['ltl_score'] for s in self.training_stats],
                'g-', label='LTL Score')
        plt.plot(verify_episodes,
                [s['epsilon'] for s in self.training_stats],
                'b--', label='Epsilon')
        plt.plot(verify_episodes,
                [s['alpha'] for s in self.training_stats],
                'r--', label='Alpha')
        
        plt.title('LTL Scores and Parameters')
        plt.xlabel('Verification Episode')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_progress.png'))
        plt.close()

    def save_training_stats(self, save_dir: str = "results"):
        """Save training statistics to file"""
        os.makedirs(save_dir, exist_ok=True)
        stats_file = os.path.join(save_dir, 'training_statistics.txt')
        
        try:
            with open(stats_file, 'w') as f:
                f.write("Episode,LTL_Score,Epsilon,Alpha,Best_Score\n")
                for episode, stats in enumerate(self.training_stats):
                    f.write(f"{episode},"
                           f"{stats['ltl_score']:.4f},"
                           f"{stats['epsilon']:.4f},"
                           f"{stats['alpha']:.4f},"
                           f"{stats['best_score']:.4f}\n")
                           
            self.logger.info(f"Training statistics saved to {stats_file}")
        except Exception as e:
            self.logger.error(f"Failed to save training statistics: {str(e)}")


def main():
    """Standalone execution for testing"""
    from utils.DataLoader import DataLoader
    
    try:
        os.makedirs("PRISM-Guided-Learning/out/logs", exist_ok=True)
        
        # Load data
        dataloader = DataLoader("PRISM-Guided-Learning/data/grid_20_samples.csv")
        dataloader.load_data()
        
        # Create and evaluate agent
        agent = LTLGuidedQLearningWithObstacle()
        
        results = agent.evaluate(dataloader, max_workers=4)
        
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        for idx, result in enumerate(results):
            print(f"\nGridworld {idx+1}:")
            print(f"  LTL Score: {result['LTL_Score']:.4f}")
            print(f"  Training Time: {result['Evaluation_Time']:.2f}s")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()