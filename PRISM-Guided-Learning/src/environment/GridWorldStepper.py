from typing import Tuple
from environment.GridWorld import GridWorld


class GridWorldMover:
    """
    Handles stepping through a GridWorld environment.
    Supports variable number of goals that must be reached in sequence.
    """
    
    def __init__(self, gridWorld: GridWorld):
        self.gridWorld = gridWorld
        self.num_goals = len(gridWorld.goals)
        
        # Moving obstacle tracking (if any)
        self.current_moving_index = 0 if self.gridWorld.moving_obstacle_positions else -1
        self.current_moving_obstacle = (
            self.gridWorld.moving_obstacle_positions[0] 
            if self.gridWorld.moving_obstacle_positions else None
        )

        self.max_steps = 100
        self.steps = 0
        self.current_pos = (0, 0)
        self.visited_goals = set()  # Track which goal numbers have been visited
        self.reset()
        
    def reset(self) -> Tuple:
        """Reset environment and return initial state.
        
        Returns:
            State tuple: (x, y, g1_reached, g2_reached, ..., gN_reached)
        """
        self.current_pos = (0, 0)
        self.visited_goals = set()
        self.steps = 0
        self.current_moving_index = 0 if self.gridWorld.moving_obstacle_positions else -1
        self.current_moving_obstacle = (
            self.gridWorld.moving_obstacle_positions[0] 
            if self.gridWorld.moving_obstacle_positions else None
        )
        return self.get_state()

    def get_state(self) -> Tuple:
        """Get current state with goal flags.
        
        Returns:
            State tuple: (x, y, g1_reached, g2_reached, ..., gN_reached)
            where gK_reached is True if goal K has been visited in sequence
        """
        x, y = self.current_pos
        
        # Build goal flags tuple dynamically
        goal_flags = tuple(
            goal_num in self.visited_goals 
            for goal_num in sorted(self.gridWorld.goals.keys())
        )
        
        return (x, y) + goal_flags
    
    def _update_moving_obstacle(self):
        """Update moving obstacle position sequentially"""
        if self.gridWorld.moving_obstacle_positions:
            self.current_moving_index = (
                (self.current_moving_index + 1) % len(self.gridWorld.moving_obstacle_positions)
            )
            self.current_moving_obstacle = self.gridWorld.moving_obstacle_positions[self.current_moving_index]
    
    def step(self, action: int) -> Tuple[Tuple, float, bool]:
        """Take action and return (state, reward, done).
        
        Args:
            action: Integer action (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        self.steps += 1
        
        # Update moving obstacle position every step
        self._update_moving_obstacle()
            
        x, y = self.current_pos
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # UP, RIGHT, DOWN, LEFT
        dx, dy = moves[action]

        new_x = max(0, min(self.gridWorld.size - 1, x + dx))
        new_y = max(0, min(self.gridWorld.size - 1, y + dy))

        # Check for collision with obstacles
        hit_obstacle = False
        if (new_x, new_y) in self.gridWorld.static_obstacles:
            hit_obstacle = True
        if self.current_moving_obstacle is not None and (new_x, new_y) == self.current_moving_obstacle:
            hit_obstacle = True
            
        if hit_obstacle:
            new_x, new_y = x, y
            reward = -10.0
        else:
            self.current_pos = (new_x, new_y)
            reward = -0.1  # Small step penalty

        # Check max steps
        if self.steps >= self.max_steps:
            return self.get_state(), reward, True

        done = False
        
        # Get sorted goal numbers for sequential checking
        goal_nums = sorted(self.gridWorld.goals.keys())
        
        # Determine which goal we should be trying to reach next
        current_goal_idx = len(self.visited_goals)
        
        if current_goal_idx < self.num_goals:
            current_goal_num = goal_nums[current_goal_idx]
            current_goal_pos = self.gridWorld.goals[current_goal_num]
            
            # Check if we reached the current goal in sequence
            if self.current_pos == current_goal_pos:
                self.visited_goals.add(current_goal_num)
                
                if current_goal_idx == self.num_goals - 1:
                    # Final goal reached!
                    reward = 100.0
                    done = True
                else:
                    # Intermediate goal reached
                    reward = 10.0
            elif self.current_pos in self.gridWorld.goals.values():
                # Reached a goal out of sequence - penalty
                reward = -5.0

        # Distance-based shaping reward (only if not done and goals remain)
        if not done and len(self.visited_goals) < self.num_goals:
            next_goal_idx = len(self.visited_goals)
            next_goal_num = goal_nums[next_goal_idx]
            next_goal_pos = self.gridWorld.goals[next_goal_num]
            distance = abs(new_x - next_goal_pos[0]) + abs(new_y - next_goal_pos[1])
            reward -= 0.01 * distance

        return self.get_state(), reward, done
    
    def get_current_goal(self) -> Tuple[int, Tuple[int, int]]:
        """Get the current goal number and position that should be reached next.
        
        Returns:
            Tuple of (goal_number, goal_position) or (None, None) if all goals reached
        """
        goal_nums = sorted(self.gridWorld.goals.keys())
        current_goal_idx = len(self.visited_goals)
        
        if current_goal_idx < self.num_goals:
            goal_num = goal_nums[current_goal_idx]
            return goal_num, self.gridWorld.goals[goal_num]
        return None, None