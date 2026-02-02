from . import PrismVerifier
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional
import os
from environment.GridWorld import GridWorld


@dataclass
class Requirement:
    """Single source of truth for a verification requirement."""
    name: str           # Human-readable name for logging (e.g., "Reach G1")
    property: str       # PRISM property string (e.g., 'P=? [ F "at_goal1" ];')
    weight: float       # Weight for scoring
    category: str       # Category for grouping ("reachability", "sequence", "safety")
    csv_header: str     # Column name for CSV export


class SimplifiedVerifier:
    def __init__(self, prism_verifier: PrismVerifier, gridWorld: GridWorld, logger):
        self.prism_verifier = prism_verifier
        self.logger = logger
        self.ltl_probabilities = []
        self.verification_history = []
        self.gridWorld = gridWorld
        self.num_goals = len(gridWorld.goals)
        self.requirements = self._build_requirements()
        self.logger.info(f"SimplifiedVerifier initialized with {self.num_goals} goals, {len(self.requirements)} requirements")

    def _generate_sequence_property(self, goal_nums: List[int], exclude_goal: int) -> str:
        """Generate a sequence property that ensures goals are reached in order.

        Args:
            goal_nums: List of goal numbers to reach in order (e.g., [1, 2])
            exclude_goal: Goal number that should not be reached until sequence is complete

        Returns:
            LTL property string (inner part, without P=? wrapper)
        """
        if len(goal_nums) == 1:
            return f'!"at_goal{exclude_goal}" U "at_goal{goal_nums[0]}"'
        else:
            rest_sequence = self._generate_sequence_property(goal_nums[1:], exclude_goal)
            return f'!"at_goal{exclude_goal}" U ("at_goal{goal_nums[0]}" & ({rest_sequence}))'

    def _build_requirements(self) -> List[Requirement]:
        """Build all requirements - SINGLE SOURCE OF TRUTH.

        To add a new requirement, add it here. Everything else derives from this.

        Weight distribution:
        - Single goal: 75% reachability, 25% safety
        - Multiple goals: 25% reachability, 50% sequence, 25% safety
        """
        requirements = []
        goal_list = sorted(self.gridWorld.goals.keys())

        # Calculate weights based on number of goals
        # Weights should sum to 1.0: 0.25 reachability + 0.50 sequence + 0.25 safety
        if self.num_goals == 1:
            reachability_weight = 0.75
            sequence_weight = 0.0
            safety_weight = 0.25
        else:
            reachability_weight = 0.25 / self.num_goals
            # N goals have N-1 sequence requirements
            sequence_weight = 0.50 / (self.num_goals - 1)
            safety_weight = 0.25

        # 1. Reachability requirements for each goal
        for goal_num in goal_list:
            requirements.append(Requirement(
                name=f"Reach G{goal_num}",
                property=f'P=? [ F "at_goal{goal_num}" ];',
                weight=reachability_weight,
                category="reachability",
                csv_header=f"Reach_G{goal_num}"
            ))

        # 2. Sequential ordering requirements (only if multiple goals)
        if self.num_goals > 1:
            # Individual sequence properties: G1...Gi before G(i+1)
            for i in range(len(goal_list) - 1):
                goals_before = goal_list[:i+1]
                next_goal = goal_list[i+1]
                seq_property = self._generate_sequence_property(goals_before, next_goal)

                goals_str = ','.join([f"G{g}" for g in goals_before])
                requirements.append(Requirement(
                    name=f"{goals_str} before G{next_goal}",
                    property=f'P=? [ {seq_property} ];',
                    weight=sequence_weight,
                    category="sequence",
                    csv_header=f"{''.join([f'G{g}' for g in goals_before])}_before_G{next_goal}"
                ))

            # Complete sequence property (AND of all sequential constraints)
            all_sequences = []
            for i in range(len(goal_list) - 1):
                goals_before = goal_list[:i+1]
                next_goal = goal_list[i+1]
                all_sequences.append(self._generate_sequence_property(goals_before, next_goal))

            complete_seq = ' & '.join([f'({seq})' for seq in all_sequences])
            requirements.append(Requirement(
                name="Complete sequence",
                property=f'P=? [ {complete_seq} ];',
                weight=0.0,  # Informational only, not scored
                category="sequence",
                csv_header="Complete_Sequence"
            ))

        # 3. Safety: Obstacle avoidance
        requirements.append(Requirement(
            name="Avoid obstacles (30 steps)",
            property='P=? [ G<=30 !"at_obstacle" ];',
            weight=safety_weight,
            category="safety",
            csv_header="Avoid_Obstacle"
        ))

        return requirements

    def _generate_property_string(self) -> str:
        """Generate PRISM property string from requirements."""
        return '\n'.join(req.property for req in self.requirements) + '\n'

    def verify_policy(self, model_str: str) -> float:
        """Verify policy and return combined score."""
        property_str = self._generate_property_string()

        probabilities = self.prism_verifier.verify_property(model_str, property_str)
        self.ltl_probabilities.append(probabilities)

        # Log results with requirement names
        self._log_verification_results(probabilities)

        score = self._calculate_score(probabilities)
        self.logger.info(f"Combined Score: {score:.4f}")
        return score

    def _log_verification_results(self, probabilities: List[float]) -> None:
        """Log each probability with its corresponding requirement."""
        self.logger.info("PRISM Verification Results:")

        max_name_len = max(len(req.name) for req in self.requirements)

        for i, req in enumerate(self.requirements):
            if i < len(probabilities):
                prob = probabilities[i]
                weight_str = f"(weight: {req.weight:.3f})" if req.weight > 0 else "(info only)"
                self.logger.info(f"  {req.name:<{max_name_len}} = {prob:.4f} {weight_str}")
            else:
                self.logger.warning(f"  {req.name:<{max_name_len}} = MISSING")

    def _calculate_score(self, results: List[float]) -> float:
        """Calculate policy score using requirement weights."""
        if len(results) < len(self.requirements):
            self.logger.warning(f"Expected {len(self.requirements)} results, got {len(results)}")
            return 0.0

        score = sum(
            req.weight * results[i]
            for i, req in enumerate(self.requirements)
            if i < len(results)
        )
        return score

    def _generate_header(self) -> str:
        """Generate CSV header from requirements."""
        header_parts = ["Episode"]
        header_parts.extend(req.csv_header for req in self.requirements)
        header_parts.extend(["Episode_Reward", "Score"])
        return ','.join(header_parts)

    def save_probabilities_to_file(self, agent=None, save_dir: str = "logs", filename: str = "ltl_probabilities_and_rewards.txt"):
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, filename)

        try:
            with open(output_path, 'w') as f:
                f.write(self._generate_header() + '\n')

                for episode, probs in enumerate(self.ltl_probabilities):
                    prob_str = ','.join(f"{p:.4f}" for p in probs)
                    reward = agent.episode_rewards[episode] if (agent and hasattr(agent, 'episode_rewards') and episode < len(agent.episode_rewards)) else 0.0
                    score = self._calculate_score(probs)
                    f.write(f"{episode},{prob_str},{reward:.4f},{score:.4f}\n")

                self.logger.info(f"Saved LTL probabilities and rewards to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving probabilities to file: {str(e)}")
