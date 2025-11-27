from . import PrismVerifier
import logging
from typing import Dict, Tuple, List, Any
import os
from environment.GridWorld import GridWorld

class SimplifiedVerifier:
    def __init__(self, prism_verifier: PrismVerifier, gridWorld: GridWorld, logger):
        self.prism_verifier = prism_verifier
        self.logger = logger
        self.ltl_probabilities = []
        self.verification_history = []  
        self.gridWorld = gridWorld
        self.num_goals = len(gridWorld.goals)
        self.logger.info(f"SimplifiedVerifier initialized with {self.num_goals} goals")

    def _generate_sequence_property(self, goal_nums: List[int], exclude_goal: int) -> str:
        """Generate a sequence property that ensures goals are reached in order.
        
        Args:
            goal_nums: List of goal numbers to reach in order (e.g., [1, 2])
            exclude_goal: Goal number that should not be reached until sequence is complete
        
        Returns:
            LTL property string
        """
        if len(goal_nums) == 1:
            # Base case: reach goal_nums[0] before exclude_goal
            return f'!\"at_goal{exclude_goal}\" U \"at_goal{goal_nums[0]}\"'
        else:
            # Recursive case: reach goal_nums[0], then the rest, all before exclude_goal
            rest_sequence = self._generate_sequence_property(goal_nums[1:], exclude_goal)
            return f'!\"at_goal{exclude_goal}\" U (\"at_goal{goal_nums[0]}\" & ({rest_sequence}))'

    def _generate_property_string(self) -> str:
        """Generate PRISM property string for variable number of goals"""
        properties = []
        
        # 1. Reachability properties for each goal
        for goal_num in self.gridWorld.goals:
            properties.append(f'P=? [ F \"at_goal{goal_num}\" ];')
        
        # 2. Sequential ordering properties (only if multiple goals)
        if self.num_goals > 1:
            goal_list = sorted(self.gridWorld.goals.keys())
            
            # Generate sequence properties: G1...Gi before G(i+1)
            for i in range(len(goal_list) - 1):
                goals_before = goal_list[:i+1]
                next_goal = goal_list[i+1]
                seq_property = self._generate_sequence_property(goals_before, next_goal)
                properties.append(f'P=? [ {seq_property} ];')
            
            # Complete sequence property (AND of all sequential constraints)
            all_sequences = []
            for i in range(len(goal_list) - 1):
                goals_before = goal_list[:i+1]
                next_goal = goal_list[i+1]
                all_sequences.append(self._generate_sequence_property(goals_before, next_goal))
            
            complete_seq = ' & '.join([f'({seq})' for seq in all_sequences])
            properties.append(f'P=? [ {complete_seq} ];')
        
        # 3. Obstacle avoidance
        properties.append('P=? [ G<=30 !\"at_obstacle\" ];')
        
        return '\n'.join(properties) + '\n'

    def verify_policy(self, model_str: str) -> float:
        property_str = self._generate_property_string()
        
        probabilities = self.prism_verifier.verify_property(model_str, property_str)
        self.ltl_probabilities.append(probabilities)
        
        score = self._calculate_score(probabilities)
        self.logger.info(f"Goal Sequence Probabilities: {probabilities}")
        self.logger.info(f"Combined Score: {score:.4f}")
        return score

    def _calculate_score(self, results: List[float]) -> float:
        """Calculate policy score with balanced weights
        
        Scoring breakdown:
        - For single goal (N=1): 3/4 for reachability, 1/4 for obstacle avoidance
        - For multiple goals (N>1):
          * 1/4 (0.25) split evenly among goal reachability
          * 2/4 (0.50) split evenly among sequence properties
          * 1/4 (0.25) for obstacle avoidance
        """
        if self.num_goals == 1:
            # Single goal case: only reachability + obstacle avoidance
            expected_len = 2  # 1 reachability + 1 obstacle
            if len(results) < expected_len:
                self.logger.warning(f"Single goal: Expected {expected_len} results, got {len(results)}")
                return 0.0
            
            goal_reach = results[0]
            avoid_obstacle = results[1]
            
            # For single goal: 75% reachability, 25% obstacle
            score = 0.75 * goal_reach + 0.25 * avoid_obstacle
            self.logger.info(f"Single goal score: 0.75*{goal_reach:.2f} + 0.25*{avoid_obstacle:.2f} = {score:.4f}")
            return score
        
        else:
            # Multiple goals case
            expected_len = 2 * self.num_goals + 1
            if len(results) < expected_len:
                self.logger.warning(f"Multiple goals: Expected {expected_len} results, got {len(results)}")
                return 0.0
            
            # Extract results
            goal_reach = results[:self.num_goals]  # First N results
            sequences = results[self.num_goals:2*self.num_goals]  # Next N results
            avoid_obstacle = results[-1]  # Last result
            
            # Calculate weights
            goal_weight = 0.25 / self.num_goals  # Split 1/4 evenly
            seq_weight = 0.50 / self.num_goals   # Split 2/4 evenly
            obstacle_weight = 0.25
            
            # Calculate score
            score = (
                sum(goal_weight * p for p in goal_reach) +
                sum(seq_weight * p for p in sequences) +
                obstacle_weight * avoid_obstacle
            )
            
            return score

    def _generate_header(self) -> str:
        """Generate CSV header for variable number of goals"""
        header_parts = ["Episode"]
        
        # Goal reachability columns
        for goal_num in sorted(self.gridWorld.goals.keys()):
            header_parts.append(f"Reach_G{goal_num}")
        
        # Sequence columns (only if multiple goals)
        if self.num_goals > 1:
            goal_list = sorted(self.gridWorld.goals.keys())
            for i in range(len(goal_list) - 1):
                goals_str = ''.join([f"G{g}" for g in goal_list[:i+1]])
                next_goal = goal_list[i+1]
                header_parts.append(f"{goals_str}_before_G{next_goal}")
            
            # Complete sequence column
            header_parts.append("Complete_Sequence")
        
        # Other columns
        header_parts.extend(["Avoid_Obstacle", "Episode_Reward", "Score"])
        
        return ','.join(header_parts)

    def save_probabilities_to_file(self, agent=None, save_dir: str = "logs", filename: str = "ltl_probabilities_and_rewards.txt"):
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, filename)

        try:
            with open(output_path, 'w') as f:
                # Write headers
                f.write(self._generate_header() + '\n')
                
                # Write data for each episode
                for episode, probs in enumerate(self.ltl_probabilities):
                    prob_str = ','.join(f"{p:.4f}" for p in probs)
                    reward = agent.episode_rewards[episode] if (agent and hasattr(agent, 'episode_rewards') and episode < len(agent.episode_rewards)) else 0.0
                    score = self._calculate_score(probs)
                    f.write(f"{episode},{prob_str},{reward:.4f},{score:.4f}\n")
                    
                self.logger.info(f"Saved LTL probabilities and rewards to {output_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving probabilities to file: {str(e)}")