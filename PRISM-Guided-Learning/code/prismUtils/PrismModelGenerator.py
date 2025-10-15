import logging
import numpy as np
from typing import Dict, Tuple
from . import GridWorldEnv


class PrismModelGenerator:
    def __init__(self, gridWorld: GridWorldEnv):
        self.logger = logging.getLogger(__name__)
        self.gridWorld = gridWorld
        self.debug_transitions = set()
    def _generate_transitions(self, x: int, y: int, g1: bool, g2: bool, g3: bool, 
                           q_values: np.ndarray) -> list:
        """Generate transitions with sequential moving obstacle"""
        
        # softmax on q values
        q_exp = np.exp((q_values - np.max(q_values)) / 0.1)
        probs = q_exp / np.sum(q_exp)
        
        int_probs = np.round(probs * 1000).astype(int)
        diff = 1000 - int_probs.sum()
        int_probs[np.argmax(probs)] += diff
        
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        transitions = []
        
        guard = (f"  [] (x={x} & y={y}"
                f" & g1={str(g1).lower()}"
                f" & g2={str(g2).lower()}"
                f" & g3={str(g3).lower()}) ->")
        
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
                        # Higher probability for next position in sequence
                        transition_prob = 0.9  # 90% chance to move to next position
                        stay_prob = 0.1  # 10% chance to stay in current position
                        
                        # Stay in current position with small probability
                        updates.append(f"{(int_probs[i]/1000) * stay_prob}:(x'={x})"
                                    f" & (y'={y})"
                                    f" & (g1'={str(g1).lower()})"
                                    f" & (g2'={str(g2).lower()})"
                                    f" & (g3'={str(g3).lower()})")
                        
                        # Move to next position in sequence with higher probability
                        updates.append(f"{(int_probs[i]/1000) * transition_prob}:(x'={next_obs_pos[0]})"
                                    f" & (y'={next_obs_pos[1]})"
                                    f" & (g1'={str(g1).lower()})"
                                    f" & (g2'={str(g2).lower()})"
                                    f" & (g3'={str(g3).lower()})")
                        break
                else:
                    # Handle static obstacles
                    if (new_x, new_y) in self.static_obstacles:
                        updates.append(f"{int_probs[i]/1000}:(x'={x})"
                                    f" & (y'={y})"
                                    f" & (g1'={str(g1).lower()})"
                                    f" & (g2'={str(g2).lower()})"
                                    f" & (g3'={str(g3).lower()})")
                    else:
                        # Regular movement
                        new_g1 = g1 or ((new_x, new_y) == self.gridWorld.goals[1])
                        new_g2 = g2 or (g1 and (new_x, new_y) == self.gridWorld.goals[2])
                        new_g3 = g3 or (g2 and (new_x, new_y) == self.gridWorld.goals[3])

                        updates.append(f"{int_probs[i]/1000}:(x'={new_x})"
                                    f" & (y'={new_y})"
                                    f" & (g1'={str(new_g1).lower()})"
                                    f" & (g2'={str(new_g2).lower()})"
                                    f" & (g3'={str(new_g3).lower()})")
        
        if updates:
            transition = f"{guard} " + " + ".join(updates) + ";"
            transitions.append(transition)
        
        return transitions

    def generate_prism_model(self, q_table: Dict[Tuple[int, int, bool, bool, bool], np.ndarray]) -> str:
        """Generate PRISM model string with moving obstacle labels"""
        self.debug_transitions = set()
        
        model = ["dtmc", "", f"const int N = {self.gridWorld.size};", ""]
        
        # Module definition
        model.append("module gridworld")
        model.append(f"  x : [0..{self.gridWorld.size-1}] init 0;")
        model.append(f"  y : [0..{self.gridWorld.size-1}] init 0;")
        for goal in self.gridWorld.goals:
            model.append(f"  g{goal} : bool init false;")
        model.append("")
        
        total_states = 0

        for x in range(self.gridWorld.size):
            for y in range(self.gridWorld.size):
                for g1 in [False, True]:
                    for g2 in [False, True]:
                        for g3 in [False, True]:
                            state = (x, y, g1, g2, g3)
                            if state in q_table:
                                total_states += 1 # ASH: lol this will always be true and is unecessary, i think 
                                                    # they prob used it for logging
                                transitions = self._generate_transitions(
                                    x, y, g1, g2, g3, q_table[state]
                                )
                                model.extend(transitions)
        
        model.append("endmodule")
        model.append("")

        # Labels for properties
        model.append("// Labels for properties")
        for goal in self.gridWorld.goals:
            model.append(f'label "at_goal{goal}" = x={self.gridWorld.goals[goal][0]} & y={self.gridWorld.goals[goal][1]};')

        # Generate moving obstacle positions label
        moving_obstacle_label = " | ".join([f"(x={x} & y={y})" 
                                          for x, y in self.gridWorld.moving_obstacle_positions])
        static_obstacle_label = " | ".join([f"(x={x} & y={y})" 
                                          for x, y in self.gridWorld.static_obstacles])
        
        at_obstacle_label = f"({moving_obstacle_label}) | ({static_obstacle_label})" if moving_obstacle_label else static_obstacle_label

        model.append(f'label "at_obstacle" = {at_obstacle_label};')

        model_str = "\n".join(model)
        return model_str