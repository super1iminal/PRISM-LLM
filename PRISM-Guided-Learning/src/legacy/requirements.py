from dataclasses import dataclass
from typing import List
from legacy.prism_verifier import PrismVerifier
from legacy.gridworld import GridWorld

GOAL_REACHABILITY_THRESHOLD = 0.8
SEQUENCE_ORDERING_THRESHOLD = 0.8
OBSTACLE_AVOIDANCE_THRESHOLD = 0.7


def get_threshold_for_key(key: str) -> float:
    """Map a probability key to its per-requirement threshold."""
    if key.startswith('goal'):
        return GOAL_REACHABILITY_THRESHOLD
    elif key.startswith('seq_') or key == 'complete_sequence':
        return SEQUENCE_ORDERING_THRESHOLD
    elif key.startswith('avoid_moving'):
        return OBSTACLE_AVOIDANCE_THRESHOLD
    return GOAL_REACHABILITY_THRESHOLD  # default fallback


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
        - No moving obstacles:
          - Single goal: 100% reachability
          - Multiple goals: 33% reachability, 67% sequence
        - With moving obstacles:
          - Single goal: 75% reachability, 25% safety (split across segments)
          - Multiple goals: 25% reachability, 50% sequence, 25% safety (split across segments)
        """
        requirements = []
        goal_list = sorted(self.gridWorld.goals.keys())
        has_moving = bool(self.gridWorld.moving_obstacle_positions)

        # Calculate weights based on number of goals and whether moving obstacles exist
        if has_moving:
            if self.num_goals == 1:
                reachability_weight = 0.75
                sequence_weight = 0.0
                safety_budget = 0.25
            else:
                reachability_weight = 0.25 / self.num_goals
                sequence_weight = 0.50 / (self.num_goals - 1)
                safety_budget = 0.25
        else:
            # No moving obstacles — no safety requirements, redistribute weight
            if self.num_goals == 1:
                reachability_weight = 1.0
                sequence_weight = 0.0
            else:
                reachability_weight = 0.25 / self.num_goals
                sequence_weight = 0.50 / (self.num_goals - 1)
                # Redistribute old safety weight (0.25) into reachability
                # Total reachability becomes 0.25 + 0.25 = 0.50
                reachability_weight = 0.50 / self.num_goals
            safety_budget = 0.0

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

        # 3. Per-segment moving obstacle avoidance (replaces old avoid_obstacle)
        if has_moving:
            seg_weight = safety_budget / len(goal_list)
            for goal_num in goal_list:
                requirements.append(Requirement(
                    name=f"Avoid moving obs (seg {goal_num})",
                    property=f'P=? [ G ("in_seg{goal_num}" => !"at_moving_obs") ];',
                    weight=seg_weight,
                    category="segment_safety",
                    csv_header=f"Avoid_Moving_Seg{goal_num}"
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
