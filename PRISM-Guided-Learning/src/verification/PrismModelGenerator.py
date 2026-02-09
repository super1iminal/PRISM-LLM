import logging
from typing import Dict, Tuple
from itertools import product
from environment.GridWorld import GridWorld


class PrismModelGenerator:
    def __init__(self, gridWorld: GridWorld, logger):
        self.logger = logger
        self.gridWorld = gridWorld
        self.debug_transitions = set()
        self.num_goals = len(gridWorld.goals)
        self.has_moving_obs = bool(gridWorld.moving_obstacle_positions)

    def _generate_transitions(self, x: int, y: int, goal_states: tuple,
                           best_action: int) -> list:
        """Generate transitions with stochastic slip model.

        Uses deterministic policy with stochastic execution:
        - 0.7 probability of moving in the intended direction
        - 0.15 probability of slipping left (relative to intended direction)
        - 0.15 probability of slipping right (relative to intended direction)

        Moving obstacles are passable — collision is tracked via PRISM labels only.

        Args:
            x, y: Current position
            goal_states: Tuple of booleans indicating which goals have been reached
            best_action: Best action (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        """

        # Actions: [0=UP, 1=RIGHT, 2=DOWN, 3=LEFT]
        left_action = {0: 3, 1: 0, 2: 1, 3: 2}
        right_action = {0: 1, 1: 2, 2: 3, 3: 0}

        # Stochastic transition probabilities (from GridWorld)
        prob_forward = self.gridWorld.prob_forward
        prob_left = self.gridWorld.prob_slip_left
        prob_right = self.gridWorld.prob_slip_right

        action_probs = {
            best_action: prob_forward,
            left_action[best_action]: prob_left,
            right_action[best_action]: prob_right
        }

        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # UP, RIGHT, DOWN, LEFT
        transitions = []

        # Build guard — use [step] label when moving obstacles exist for synchronization
        goal_guard_parts = " & ".join([f"g{i+1}={str(goal_states[i]).lower()}"
                                       for i in range(self.num_goals)])
        action_label = "[step]" if self.has_moving_obs else "[]"
        guard = f"  {action_label} (x={x} & y={y} & {goal_guard_parts}) ->"

        updates = []

        for action, prob in action_probs.items():
            dx, dy = moves[action]
            new_x = max(0, min(self.gridWorld.size - 1, x + dx))
            new_y = max(0, min(self.gridWorld.size - 1, y + dy))

            # Handle static obstacles — bounce back
            if (new_x, new_y) in self.gridWorld.static_obstacles:
                goal_update = " & ".join([f"(g{i+1}'={str(goal_states[i]).lower()})"
                                          for i in range(self.num_goals)])
                updates.append(f"{prob}:(x'={x})"
                            f" & (y'={y}) & {goal_update}")
            else:
                # Regular movement — moving obstacle cells are passable
                new_goal_states = list(goal_states)

                for goal_idx in range(self.num_goals):
                    goal_num = goal_idx + 1

                    if goal_idx == 0:
                        if (new_x, new_y) == self.gridWorld.goals[goal_num]:
                            new_goal_states[goal_idx] = True
                    else:
                        prev_goal_reached = new_goal_states[goal_idx - 1]
                        if prev_goal_reached and (new_x, new_y) == self.gridWorld.goals[goal_num]:
                            new_goal_states[goal_idx] = True

                goal_update = " & ".join([f"(g{i+1}'={str(new_goal_states[i]).lower()})"
                                          for i in range(self.num_goals)])

                updates.append(f"{prob}:(x'={new_x})"
                            f" & (y'={new_y}) & {goal_update}")

        if updates:
            transition = f"{guard} " + " + ".join(updates) + ";"
            transitions.append(transition)

        return transitions

    def generate_prism_model(self, policy: Dict[Tuple, int]) -> str:
        """Generate PRISM model string with synchronized moving obstacle module.

        Args:
            policy: Dictionary mapping states to best action (0-3)
                    State format: (x, y, g1, g2, ..., gN) where N is number of goals
        """
        self.debug_transitions = set()

        model = ["dtmc", "", f"const int N = {self.gridWorld.size};", ""]

        # Agent module
        model.append("module gridworld")
        model.append(f"  x : [0..{self.gridWorld.size-1}] init 0;")
        model.append(f"  y : [0..{self.gridWorld.size-1}] init 0;")
        for goal_num in self.gridWorld.goals:
            model.append(f"  g{goal_num} : bool init false;")
        model.append("")

        total_states = 0

        goal_combinations = list(product([False, True], repeat=self.num_goals))

        for x in range(self.gridWorld.size):
            for y in range(self.gridWorld.size):
                for goal_combo in goal_combinations:
                    state = (x, y) + goal_combo
                    if state in policy:
                        total_states += 1
                        transitions = self._generate_transitions(
                            x, y, goal_combo, policy[state]
                        )
                        model.extend(transitions)

        model.append("endmodule")
        model.append("")

        # Synchronized obstacle module (only when moving obstacles exist)
        if self.has_moving_obs:
            num_positions = self.gridWorld.num_obs_steps
            max_idx = num_positions - 1
            model.append("module obstacle")
            model.append(f"  obs_idx : [0..{max_idx}] init 0;")
            # Two guarded commands for wrapping
            if max_idx > 0:
                model.append(f"  [step] (obs_idx < {max_idx}) -> (obs_idx'=obs_idx+1);")
                model.append(f"  [step] (obs_idx = {max_idx}) -> (obs_idx'=0);")
            else:
                # Single position — stays at 0
                model.append("  [step] (obs_idx = 0) -> (obs_idx'=0);")
            model.append("endmodule")
            model.append("")

        # Labels for properties
        model.append("// Labels for properties")
        for goal_num in self.gridWorld.goals:
            gx, gy = self.gridWorld.goals[goal_num]
            model.append(f'label "at_goal{goal_num}" = x={gx} & y={gy};')

        # Moving obstacle collision label (position + time synchronized)
        if self.has_moving_obs:
            collision_parts = []
            for idx, (ox, oy) in enumerate(self.gridWorld.moving_obstacle_positions):
                collision_parts.append(f"(obs_idx={idx} & x={ox} & y={oy})")
            at_moving_obs = " | ".join(collision_parts)
            model.append(f'label "at_moving_obs" = {at_moving_obs};')

        # Segment labels (for per-segment queries)
        goal_list = sorted(self.gridWorld.goals.keys())
        if len(goal_list) > 0:
            for i, goal_num in enumerate(goal_list):
                if i == 0:
                    model.append(f'label "in_seg{goal_num}" = !g{goal_num};')
                else:
                    prev_goal = goal_list[i - 1]
                    model.append(f'label "in_seg{goal_num}" = g{prev_goal} & !g{goal_num};')

        model_str = "\n".join(model)
        return model_str
