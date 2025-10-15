from typing import Tuple
from . import GridWorldEnv

class GridWorldMover:
    def __init__(self, gridWorld: GridWorldEnv):
        self.gridWorld = gridWorld

        self.current_moving_index = 0 if self.gridWorld.moving_obstacle_positions else -1  # Track current position in sequence
        self.current_moving_obstacle = self.gridWorld.moving_obstacle_positions[0] if self.gridWorld.moving_obstacle_positions else None

        self.max_steps = 100    # Reduced for smaller grid
        self.steps = 0
        self.current_pos = (0, 0)  # Initialize current position to (0,0)
        self.visited_goals = set()  # Initialize visited goals
        self.reset()
        
    def reset(self) -> Tuple[int, int, bool, bool, bool]:
        """Reset environment"""
        self.current_pos = (0, 0)  # Reset to starting position (0,0)
        self.visited_goals = set()
        self.steps = 0
        self.current_moving_index = 0 if self.gridWorld.moving_obstacle_positions else -1  # Track current position in sequence
        self.current_moving_obstacle = self.gridWorld.moving_obstacle_positions[0] if self.gridWorld.moving_obstacle_positions else None

        return self.get_state()

    def get_state(self) -> Tuple[int, int, bool, bool, bool]:
        """Get current state with goal flags"""
        x, y = self.current_pos
        g1_reached = 1 in self.visited_goals
        g2_reached = 2 in self.visited_goals
        g3_reached = 3 in self.visited_goals
        return (x, y, g1_reached, g2_reached, g3_reached)
    
    
    def step(self, action: int) -> Tuple[Tuple[int, int, bool, bool, bool], float, bool]:
        """Take action and return (state, reward, done)"""
        self.steps += 1
        
        # Update moving obstacle position every step
        self._update_moving_obstacle()
            
        x, y = self.current_pos
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  
        dx, dy = moves[action]

        new_x = max(0, min(self.gridWorld.size - 1, x + dx))
        new_y = max(0, min(self.gridWorld.size - 1, y + dy))

        # Check for collision with current obstacle position
        if ((new_x, new_y) in self.gridWorld.static_obstacles or
            (self.current_moving_obstacle != None and (new_x, new_y) == self.current_moving_obstacle)):
            new_x, new_y = x, y
            reward = -10.0
        else:
            self.current_pos = (new_x, new_y)
            reward = -0.1  

        if self.steps >= self.max_steps:
            return self.get_state(), reward, True

        done = False

        # Check goals in sequence
        current_goal = len(self.visited_goals) + 1
        if self.current_pos == self.gridWorld.goals.get(current_goal):
            self.visited_goals.add(current_goal)
            if current_goal == 3:  
                reward = 100.0
                done = True
            else:
                reward = 10.0
        elif self.current_pos in self.gridWorld.goals.values():
            reward = -5.0

        if not done and len(self.visited_goals) < 3:
            next_goal = self.gridWorld.goals[len(self.visited_goals) + 1]
            distance = abs(new_x - next_goal[0]) + abs(new_y - next_goal[1])
            reward -= 0.01 * distance  

        return self.get_state(), reward, done