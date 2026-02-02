from typing import Tuple

class GridWorld:
    def __init__(self, size: int, goals: dict, static_obstacles: list, moving_obstacle_positions: list):
        self.size = size  # 4x4 grid
        self.goals = goals
        # Static obstacles
        self.static_obstacles = static_obstacles

        # Moving obstacle positions in sequence
        self.moving_obstacle_positions = moving_obstacle_positions

        # Stochastic transition probabilities (slip model)
        self.prob_forward = 0.7      # Probability of moving in intended direction
        self.prob_slip_left = 0.15   # Probability of slipping 90° left
        self.prob_slip_right = 0.15  # Probability of slipping 90° right
        
    def _update_moving_obstacle(self):
        """Update moving obstacle position sequentially"""
        self.current_moving_index = (self.current_moving_index + 1) % len(self.moving_obstacle_positions)
        self.current_moving_obstacle = self.moving_obstacle_positions[self.current_moving_index]