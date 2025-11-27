import logging
import numpy as np
from typing import Dict, Tuple
from itertools import product
from environment.GridWorld import GridWorld


class PrismModelGenerator:
    def __init__(self, gridWorld: GridWorld, logger):
        self.logger = logger
        self.gridWorld = gridWorld
        self.debug_transitions = set()
        self.num_goals = len(gridWorld.goals)
    
    def _generate_transitions(self, x: int, y: int, goal_states: tuple, 
                           q_values: np.ndarray) -> list:
        """Generate transitions with sequential moving obstacle
        
        Args:
            x, y: Current position
            goal_states: Tuple of booleans indicating which goals have been reached
            q_values: Q-values for actions from this state
        """
        
        # softmax on q values
        q_exp = np.exp((q_values - np.max(q_values)) / 0.1)
        probs = q_exp / np.sum(q_exp)
        
        int_probs = np.round(probs * 1000).astype(int)
        diff = 1000 - int_probs.sum()
        int_probs[np.argmax(probs)] += diff
        
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        transitions = []
        
        # Build guard with variable goals
        goal_guard_parts = " & ".join([f"g{i+1}={str(goal_states[i]).lower()}" 
                                       for i in range(self.num_goals)])
        guard = f"  [] (x={x} & y={y} & {goal_guard_parts}) ->"
        
        updates = []
        
        for i, (dx, dy) in enumerate(moves):
            if int_probs[i] > 0:
                new_x = max(0, min(self.gridWorld.size - 1, x + dx))
                new_y = max(0, min(self.gridWorld.size - 1, y + dy))

                # Handle moving obstacle with more precise sequence transitions
                for obs_idx, current_obs_pos in enumerate(self.gridWorld.moving_obstacle_positions):
                    next_obs_idx = (obs_idx + 1) % len(self.gridWorld.moving_obstacle_positions)
                    next_obs_pos = self.gridWorld.moving_obstacle_positions[next_obs_idx]

                    if (new_x, new_y) == current_obs_pos:
                        # Calculate transition probabilities based on sequence
                        transition_prob = 0.9  # 90% chance to move to next position
                        stay_prob = 0.1  # 10% chance to stay in current position
                        
                        # Build goal update string
                        goal_update = " & ".join([f"(g{i+1}'={str(goal_states[i]).lower()})" 
                                                  for i in range(self.num_goals)])
                        
                        # Stay in current position with small probability
                        updates.append(f"{(int_probs[i]/1000) * stay_prob}:(x'={x})"
                                    f" & (y'={y}) & {goal_update}")
                        
                        # Move to next position in sequence with higher probability
                        updates.append(f"{(int_probs[i]/1000) * transition_prob}:(x'={next_obs_pos[0]})"
                                    f" & (y'={next_obs_pos[1]}) & {goal_update}")
                        break
                else:
                    # TODO: don't override LLM decisions
                    # TODO: try to do this in PRISM instead
                    # Handle static obstacles
                    if (new_x, new_y) in self.gridWorld.static_obstacles:
                        goal_update = " & ".join([f"(g{i+1}'={str(goal_states[i]).lower()})" 
                                                  for i in range(self.num_goals)])
                        updates.append(f"{int_probs[i]/1000}:(x'={x})"
                                    f" & (y'={y}) & {goal_update}")
                    else:
                        # Regular movement - update goals sequentially
                        new_goal_states = list(goal_states)
                        
                        # Check each goal in sequence
                        for goal_idx in range(self.num_goals):
                            goal_num = goal_idx + 1  # Goals are numbered starting from 1
                            
                            if goal_idx == 0:
                                # First goal can be reached directly
                                if (new_x, new_y) == self.gridWorld.goals[goal_num]:
                                    new_goal_states[goal_idx] = True
                            else:
                                # Subsequent goals require previous goal to be reached
                                prev_goal_reached = new_goal_states[goal_idx - 1]
                                if prev_goal_reached and (new_x, new_y) == self.gridWorld.goals[goal_num]:
                                    new_goal_states[goal_idx] = True
                        
                        goal_update = " & ".join([f"(g{i+1}'={str(new_goal_states[i]).lower()})" 
                                                  for i in range(self.num_goals)])
                        
                        updates.append(f"{int_probs[i]/1000}:(x'={new_x})"
                                    f" & (y'={new_y}) & {goal_update}")
        
        if updates:
            transition = f"{guard} " + " + ".join(updates) + ";"
            transitions.append(transition)
        
        return transitions

    def generate_prism_model(self, q_table: Dict[Tuple, np.ndarray]) -> str:
        """Generate PRISM model string with moving obstacle labels
        
        Args:
            q_table: Dictionary mapping states to Q-values
                    State format: (x, y, g1, g2, ..., gN) where N is number of goals
        """
        self.debug_transitions = set()
        
        model = ["dtmc", "", f"const int N = {self.gridWorld.size};", ""]
        
        # Module definition
        model.append("module gridworld")
        model.append(f"  x : [0..{self.gridWorld.size-1}] init 0;")
        model.append(f"  y : [0..{self.gridWorld.size-1}] init 0;")
        for goal_num in self.gridWorld.goals:
            model.append(f"  g{goal_num} : bool init false;")
        model.append("")
        
        total_states = 0

        # Generate all combinations of goal states
        goal_combinations = list(product([False, True], repeat=self.num_goals))
        
        for x in range(self.gridWorld.size):
            for y in range(self.gridWorld.size):
                for goal_combo in goal_combinations:
                    state = (x, y) + goal_combo
                    if state in q_table:
                        total_states += 1
                        transitions = self._generate_transitions(
                            x, y, goal_combo, q_table[state]
                        )
                        model.extend(transitions)
        
        model.append("endmodule")
        model.append("")

        # Labels for properties
        model.append("// Labels for properties")
        for goal_num in self.gridWorld.goals:
            model.append(f'label "at_goal{goal_num}" = x={self.gridWorld.goals[goal_num][0]} & y={self.gridWorld.goals[goal_num][1]};')

        # Generate obstacle labels
        obstacle_positions = []
        
        # Add moving obstacles
        for x, y in self.gridWorld.moving_obstacle_positions:
            obstacle_positions.append(f"(x={x} & y={y})")
        
        # Add static obstacles
        for x, y in self.gridWorld.static_obstacles:
            obstacle_positions.append(f"(x={x} & y={y})")
        
        # Only add obstacle label if there are any obstacles
        if obstacle_positions:
            at_obstacle_label = " | ".join(obstacle_positions)
            model.append(f'label "at_obstacle" = {at_obstacle_label};')
        else:
            # No obstacles - use a label that's always false
            model.append('label "at_obstacle" = false;')

        model_str = "\n".join(model)
        return model_str