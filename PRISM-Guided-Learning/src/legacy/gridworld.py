from typing import Tuple

class GridWorld:
    def __init__(self, size: int, goals: dict, static_obstacles: list, moving_obstacle_positions: list):
        self.size = size  # 4x4 grid
        self.goals = goals
        # Static obstacles
        self.static_obstacles = static_obstacles

        # Moving obstacle: expand raw path into repeating cycle
        self.moving_obstacle_path_raw = list(moving_obstacle_positions)
        self.moving_obstacle_positions = self._expand_path(moving_obstacle_positions)
        self.num_obs_steps = len(self.moving_obstacle_positions)

        # Stochastic transition probabilities (slip model)
        self.prob_forward = 0.7      # Probability of moving in intended direction
        self.prob_slip_left = 0.15   # Probability of slipping 90° left
        self.prob_slip_right = 0.15  # Probability of slipping 90° right

    @staticmethod
    def _expand_path(path: list) -> list:
        """Expand a moving obstacle path into a repeating cycle.

        - If path is empty: return []
        - If path[-1] == path[0]: already looped, strip duplicate endpoint
        - Else: back-and-forth -> path + path[-2:0:-1]
        """
        if not path:
            return []
        if len(path) == 1:
            return list(path)
        if path[-1] == path[0]:
            # Already looped – strip the duplicate endpoint
            return list(path[:-1])
        # Back-and-forth: e.g. [A,B,C] -> [A,B,C,B]
        return list(path) + list(path[-2:0:-1])

    def _update_moving_obstacle(self):
        """Update moving obstacle position sequentially"""
        self.current_moving_index = (self.current_moving_index + 1) % len(self.moving_obstacle_positions)
        self.current_moving_obstacle = self.moving_obstacle_positions[self.current_moving_index]
