from typing import Tuple

class GridWorldEnv:
    def __init__(self, size: int, goals: dict, static_obstacles: list, moving_obstacle_positions: list):
        self.size = size  # 4x4 grid
        self.goals = goals
        # Static obstacles
        self.static_obstacles = static_obstacles

        # Moving obstacle positions in sequence
        self.moving_obstacle_positions = moving_obstacle_positions
        
    def _update_moving_obstacle(self):
        """Update moving obstacle position sequentially"""
        self.current_moving_index = (self.current_moving_index + 1) % len(self.moving_obstacle_positions)
        self.current_moving_obstacle = self.moving_obstacle_positions[self.current_moving_index]