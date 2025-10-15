from typing import Tuple

class GridWorldEnv:
    def __init__(self, size: int):
        self.size = size  # 4x4 grid
        self.goals = {
            1: (2, 2),   # First intermediate goal
            2: (3, 3),   # Second intermediate goal
            3: (0, 3)    # Final goal
        }
        # Static obstacles
        self.static_obstacles = [(1, 1), (2, 1)]
        
        # Moving obstacle positions in sequence
        self.moving_obstacle_positions = [
            (3, 0), (3, 1), (3, 2), (3, 3), (3, 2),  # Moving up and down on rightmost row
        ]
        self.current_moving_index = 0  # Track current position in sequence
        self.current_moving_obstacle = self.moving_obstacle_positions[0]
        
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
        self.current_moving_index = 0
        self.current_moving_obstacle = self.moving_obstacle_positions[self.current_moving_index]
        return self.get_state()

    def get_state(self) -> Tuple[int, int, bool, bool, bool]:
        """Get current state with goal flags"""
        x, y = self.current_pos
        g1_reached = 1 in self.visited_goals
        g2_reached = 2 in self.visited_goals
        g3_reached = 3 in self.visited_goals
        return (x, y, g1_reached, g2_reached, g3_reached)

    def _update_moving_obstacle(self):
        """Update moving obstacle position sequentially"""
        self.current_moving_index = (self.current_moving_index + 1) % len(self.moving_obstacle_positions)
        self.current_moving_obstacle = self.moving_obstacle_positions[self.current_moving_index]

    def step(self, action: int) -> Tuple[Tuple[int, int, bool, bool, bool], float, bool]:
        """Take action and return (state, reward, done)"""
        self.steps += 1
        
        # Update moving obstacle position every step
        self._update_moving_obstacle()
            
        x, y = self.current_pos
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  
        dx, dy = moves[action]

        new_x = max(0, min(self.size - 1, x + dx))
        new_y = max(0, min(self.size - 1, y + dy))
        
        # Check for collision with current obstacle position
        if ((new_x, new_y) in self.static_obstacles or 
            (new_x, new_y) == self.current_moving_obstacle):
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
        if self.current_pos == self.goals.get(current_goal):
            self.visited_goals.add(current_goal)
            if current_goal == 3:  
                reward = 100.0
                done = True
            else:
                reward = 10.0
        elif self.current_pos in self.goals.values():
            reward = -5.0

        if not done and len(self.visited_goals) < 3:
            next_goal = self.goals[len(self.visited_goals) + 1]
            distance = abs(new_x - next_goal[0]) + abs(new_y - next_goal[1])
            reward -= 0.01 * distance  

        return self.get_state(), reward, done