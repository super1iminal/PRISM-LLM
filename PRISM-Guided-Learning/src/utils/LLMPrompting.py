from typing import List, Tuple
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

PROMPT_TEXT = """You are an expert path planner working on formulating paths that meet formal requirements.

The grid world is {size} x {size}. Here is the visual layout:

{grid_visual}

Legend:
- 'S' = Start position (0,0) - the initial state
- 'G' = Current goal position you must reach
- 'X' = Static obstacle (CANNOT enter - you will bounce back)
- 'F' = Future goal (treat as obstacle for now - avoid it)
- 'M' = Moving obstacle position (avoid if possible)
- '.' = Empty cell you can move through

COORDINATE SYSTEM:
- Position format: (row, col) where row is Y-axis, col is X-axis
- (0,0) is in the TOP-LEFT corner
- Row increases DOWNWARD (0 → 1 → 2...)
- Column increases RIGHTWARD (0 → 1 → 2...)

ACTIONS - pick ONE best action per state:
- 0 = UP: Move to (row-1, col) - DECREASES row
- 1 = RIGHT: Move to (row, col+1) - INCREASES column
- 2 = DOWN: Move to (row+1, col) - INCREASES row
- 3 = LEFT: Move to (row, col-1) - DECREASES column

ACTION EXAMPLES:
From (0,0): UP→(0,0), RIGHT→(0,1), DOWN→(1,0), LEFT→(0,0)
From (0,1): UP→(0,1), RIGHT→(0,2), DOWN→(1,1), LEFT→(0,0)
From (1,0): UP→(0,0), RIGHT→(1,1), DOWN→(2,0), LEFT→(1,0)

TASK DETAILS:
- Static obstacles: {s_obstacles} (marked as 'X')
- Future goals to avoid: {f_goals} (marked as 'F')
- Moving obstacles: {k_obstacles} (marked as 'M', move with 90% probability)
- Your current goal: {goal} (marked as 'G')

STOCHASTIC EXECUTION:
When you choose an action, the agent executes it with uncertainty:
- {prob_forward_pct}% probability: Moves in the intended direction
- {prob_slip_left_pct}% probability: Slips 90° LEFT of intended direction
- {prob_slip_right_pct}% probability: Slips 90° RIGHT of intended direction

Example: If you choose DOWN, there's a {prob_forward_pct}% chance of going DOWN, {prob_slip_left_pct}% chance of going LEFT, and {prob_slip_right_pct}% chance of going RIGHT.
This means your paths should be ROBUST - avoid routes that pass adjacent to obstacles since slips could cause collisions.

CRITICAL REQUIREMENTS:
1. You MUST provide the best action for ALL {total_states} states in the {size}x{size} grid
2. Never plan a path through obstacles (X) or future goals (F)
3. The best action should create a path from ANY position to the goal
4. If a cell is an obstacle, provide an escape action (how to exit if accidentally there)

Now provide the best action (0-3) for each state.
"""

PROMPT_TEMPLATE = PromptTemplate(
    template=PROMPT_TEXT,
    input_variables=["size", "grid_visual", "s_obstacles", "f_goals", "k_obstacles", "goal", "total_states",
                     "prob_forward_pct", "prob_slip_left_pct", "prob_slip_right_pct"]
)

def generate_grid_visual(size: int, goal: Tuple[int, int], s_obstacles: List[Tuple[int, int]], 
                         f_goals: List[Tuple[int, int]], k_obstacles: List[Tuple[int, int]]) -> str:
    """Generate ASCII visual representation of the grid"""
    # Initialize grid with empty cells
    grid = [['.' for _ in range(size)] for _ in range(size)]
    
    # Place elements (order matters - later placements override)
    # Start with moving obstacles
    for obs in k_obstacles:
        if 0 <= obs[0] < size and 0 <= obs[1] < size:
            grid[obs[0]][obs[1]] = 'M'
    
    # Future goals
    for fg in f_goals:
        if 0 <= fg[0] < size and 0 <= fg[1] < size:
            grid[fg[0]][fg[1]] = 'F'
    
    # Static obstacles (higher priority)
    for obs in s_obstacles:
        if 0 <= obs[0] < size and 0 <= obs[1] < size:
            grid[obs[0]][obs[1]] = 'X'
    
    # Current goal (even higher priority)
    if 0 <= goal[0] < size and 0 <= goal[1] < size:
        grid[goal[0]][goal[1]] = 'G'
    
    # Start position (highest priority - but don't override goal if they're the same)
    if grid[0][0] not in ['G']:
        grid[0][0] = 'S'
    
    # Format as string with column headers
    header = '  ' + ' '.join(str(i) for i in range(size))
    rows = [f"{i} " + ' '.join(row) for i, row in enumerate(grid)]
    visual = header + '\n' + '\n'.join(rows)
    
    return visual

def get_prompt(size: int, s_obstacles: List[Tuple[int, int]], f_goals: List[Tuple[int, int]],
               k_obstacles: List[Tuple[int, int]], goal: Tuple[int, int],
               prob_forward: float = 0.7, prob_slip_left: float = 0.15,
               prob_slip_right: float = 0.15) -> str:
    """Generate the complete prompt with visual grid"""

    # Flatten k_obstacles if it's nested
    flat_k_obstacles = []
    if k_obstacles:
        for item in k_obstacles:
            if isinstance(item, list):
                flat_k_obstacles.extend(item)
            else:
                flat_k_obstacles.append(item)

    grid_visual = generate_grid_visual(size, goal, s_obstacles, f_goals, flat_k_obstacles)
    total_states = size * size

    return PROMPT_TEMPLATE.format(
        size=size,
        grid_visual=grid_visual,
        s_obstacles=str(s_obstacles),
        f_goals=str(f_goals),
        k_obstacles=str(k_obstacles),
        goal=str(goal),
        total_states=total_states,
        prob_forward_pct=int(prob_forward * 100),
        prob_slip_left_pct=int(prob_slip_left * 100),
        prob_slip_right_pct=int(prob_slip_right * 100)
    )


class StateAction(BaseModel):
    x: int = Field(..., description="x coordinate (row, 0-indexed)")
    y: int = Field(..., description="y coordinate (column, 0-indexed)")
    best_action: int = Field(..., description="Best action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT")

class ActionPolicy(BaseModel):
    """Best action for each state in the grid"""
    states: List[StateAction]

# Keep old names as aliases for backwards compatibility during transition
StateQ = StateAction
QTables = ActionPolicy