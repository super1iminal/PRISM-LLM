from typing import List, Tuple, Dict
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from config.Settings import get_threshold_for_key

UNIFIED_PROMPT_TEXT = """You are an expert path planner working on formulating paths that meet formal requirements.

Your task is to create a best-action policy for a grid world — choose one action per cell that guides the agent from any position to the goal while avoiding static obstacles, future goals, and moving obstacles.

The grid world is {size} x {size}. Here is the visual layout:

{grid_visual}

Legend:
- 'S' = Start position (0,0) - the initial state
- 'G' = Current goal position you must reach
- 'X' = Static obstacle (CANNOT enter - you will bounce back)
- 'F' = Future goal (treat as obstacle for now - avoid it)
- 'M' = Moving obstacle patrol path (avoid if possible)
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
TASK DETAILS:
- Static obstacles: {s_obstacles} (marked as 'X')
- Future goals to avoid: {f_goals} (marked as 'F')
- Moving obstacles: {k_obstacles} (marked as 'M'{moving_note})
- Your current goal: {goal} (marked as 'G')

STOCHASTIC EXECUTION:
When you choose an action, the agent executes it with uncertainty:
- {prob_forward_pct}% probability: Moves in the intended direction
- {prob_slip_left_pct}% probability: Slips 90° LEFT of intended direction
- {prob_slip_right_pct}% probability: Slips 90° RIGHT of intended direction
{stochastic_example}
This means your paths should be ROBUST - avoid routes that pass adjacent to obstacles since slips could cause collisions.

{feedback_block}

CRITICAL REQUIREMENTS:
1. You MUST provide the best action for ALL {total_states} states in the {size}x{size} grid
2. Never plan a path through obstacles (X) or future goals (F)
2a. Avoid moving obstacles (M) along their patrol paths if possible
3. The best action should create a path from ANY position to the goal
4. If a cell is an obstacle (X, M) or future goal (F), provide an escape action (how to exit if accidentally there due to stochastic slip)

{examples_block}

INSTRUCTIONS:
Before producing your policy, reason through the following:
1. Which neighbor of the goal has the safest slip profile?
2. What corridor feeds that approach cell?
3. Which cells are squeezed between obstacles (high unavoidable risk)?
4. Are any cells trapped / unreachable?

Now provide the best action for all states.
"""

UNIFIED_TEMPLATE = PromptTemplate(
    template=UNIFIED_PROMPT_TEXT,
    input_variables=["size", "grid_visual", "examples_block",
                     "s_obstacles", "f_goals", "k_obstacles",
                     "moving_note", "goal", "total_states",
                     "prob_forward_pct", "prob_slip_left_pct", "prob_slip_right_pct",
                     "stochastic_example", "feedback_block"]
)


def identify_problems(prism_probs: Dict[str, float]) -> str:
    """Identify problems based on probabilities that are below their per-key threshold"""
    problems = []

    for key, prob in prism_probs.items():
        threshold = get_threshold_for_key(key)
        if prob < threshold:
            if key.startswith('goal'):
                problems.append(f"- Goal reachability ({key}): {prob:.4f} < {threshold} - Path to goal may be blocked or suboptimal")
            elif key.startswith('seq_'):
                problems.append(f"- Sequence ordering ({key}): {prob:.4f} < {threshold} - Goals may be visited in wrong order")
            elif key == 'complete_sequence':
                problems.append(f"- Complete sequence: {prob:.4f} < {threshold} - Not all goals reached in correct order")
            elif key.startswith('avoid_moving_seg'):
                problems.append(
                    f"- Moving obstacle collision risk: {prob:.4f} < {threshold} — "
                    f"Your path crosses the moving obstacle trajectory. "
                    f"The moving obstacle shifts position each timestep and could be anywhere along its path, "
                    f"so perfect avoidance may not be possible. Route around the trajectory where you can, "
                    f"but do NOT sacrifice goal reachability or sequence ordering to fix this."
                )
            else:
                problems.append(f"- {key}: {prob:.4f} < {threshold}")

    if not problems:
        return "No specific problems identified, but overall LTL score is below 1.0"

    return "\n".join(problems)


def extract_segment_probs(prism_probs: Dict[str, float], goal_num: int, goal_nums: List[int]) -> Dict[str, float]:
    """Extract probabilities relevant to a specific goal segment.

    Sequence ordering: goal X's policy is active when navigating to X.
    During that navigation the agent could accidentally visit any later
    goal Z, violating seq_{Z-1}_before_Z.  So for goal X we include
    every seq constraint whose later goal comes after X.
    """
    segment = {}
    # Goal reachability
    if f'goal{goal_num}' in prism_probs:
        segment[f'goal{goal_num}'] = prism_probs[f'goal{goal_num}']
    # Sequence ordering: include all future-goal ordering constraints
    # that this goal's policy could violate
    idx = goal_nums.index(goal_num)
    for later_idx in range(idx + 1, len(goal_nums)):
        prev_goal = goal_nums[later_idx - 1]
        later_goal = goal_nums[later_idx]
        key = f'seq_{prev_goal}_before_{later_goal}'
        if key in prism_probs:
            segment[key] = prism_probs[key]
    # Per-segment moving obstacle avoidance
    seg_key = f'avoid_moving_seg{goal_num}'
    if seg_key in prism_probs:
        segment[seg_key] = prism_probs[seg_key]
    return segment


def format_probability_summary(prism_probs: Dict[str, float], relevant_keys: set = None) -> str:
    """Format probability summary for feedback showing ALL probabilities.

    Each probability is shown with its threshold, a shortfall amount if below,
    and a relevance annotation if relevant_keys is provided.
    """
    lines = []
    for key, prob in prism_probs.items():
        threshold = get_threshold_for_key(key)
        passing = prob >= threshold
        parts = [f"  {key}: {prob:.4f} (threshold: {threshold})"]
        if not passing:
            parts.append(f"-- {threshold - prob:.4f} below threshold")
        else:
            parts.append("-- meets threshold")
        if relevant_keys is not None and key in relevant_keys:
            parts.append("[relevant to this goal]")
        lines.append("  ".join(parts))
    return "\n".join(lines)


def build_prompt(size: int, s_obstacles: List[Tuple[int, int]], f_goals: List[Tuple[int, int]],
                 k_obstacles: List[Tuple[int, int]], goal: Tuple[int, int],
                 prob_forward: float = 0.7, prob_slip_left: float = 0.15,
                 prob_slip_right: float = 0.15,
                 is_feedback: bool = False,
                 probability_summary: str = "",
                 policy_visual: str = "",
                 policy_raw: str = "",
                 problems: str = "") -> str:
    """Generate the complete prompt with visual grid using the unified template.

    Args:
        size: Grid size
        s_obstacles: Static obstacle positions
        f_goals: Future goal positions
        k_obstacles: Moving obstacle positions
        goal: Current goal position
        prob_forward: Forward movement probability
        prob_slip_left: Left slip probability
        prob_slip_right: Right slip probability
        is_feedback: If True, generates feedback prompt (with results, policy visual)
        probability_summary: Formatted probability summary (feedback only)
        policy_visual: ASCII policy visualization (feedback only)
        policy_raw: Raw policy listing e.g. "(0, 0)=1, ..." (feedback only)
        problems: Problem identification text (feedback only; empty = FeedbackMinus style)

    Returns:
        Formatted prompt string
    """
    # Flatten k_obstacles if nested
    flat_k_obstacles = []
    if k_obstacles:
        for item in k_obstacles:
            if isinstance(item, list):
                flat_k_obstacles.extend(item)
            else:
                flat_k_obstacles.append(item)

    grid_visual = generate_grid_visual(size, goal, s_obstacles, f_goals, flat_k_obstacles)
    total_states = size * size
    prob_forward_pct = int(prob_forward * 100)
    prob_slip_left_pct = int(prob_slip_left * 100)
    prob_slip_right_pct = int(prob_slip_right * 100)

    examples_block = ("""
=== EXAMPLE 1 ===
Grid (6x6):
  0 1 2 3 4 5
0 S . . . . F
1 . . . . . X
2 . X . F G .
3 M M . . . .
4 M M . . X .
5 . . X . . .

Goal: (2,4)
Static obstacles: [(1,5), (2,1), (4,4), (5,2)]
Future goals: [(0,5), (2,3)]
Moving obstacles: [(3,0), (3,1), (4,0), (4,1)]
Stochastic: 70% forward, 15% slip-left, 15% slip-right

Reasoning:

1. SAFE APPROACH TO GOAL
Check each neighbor of (2,4) for slip safety:
- From (1,4) via DOWN: slip-right lands on (1,5) X — 15% collision risk.
- From (2,3) via RIGHT: (2,3) is F — cannot use as approach cell.
- From (2,5) via LEFT: slip-left lands on (1,5) X — 15% collision risk.
- From (3,4) via UP: slips land on (3,3) and (3,5), both clear — SAFE.
Best approach: (3,4) -> UP.

2. MAIN CORRIDOR
Work backward from (3,4):
- Row 3 moving RIGHT from col 2-4 is clean: (3,2)->RIGHT, (3,3)->RIGHT, (3,4)->UP.
  Slip risk at (3,3): 15% up into (2,3) F — acceptable, only alternative is worse.
- Feed into row 3 via column 2 going DOWN: (0,2)->DOWN, (1,2)->DOWN, (2,2)->DOWN.
  Squeeze at (2,2)->DOWN: 15% slips hit (2,1) X and (2,3) F — unavoidable bottleneck.

3. PROBLEM ZONES
- (2,0) is squeezed: DOWN->(3,0) M, RIGHT->(2,1) X. Only safe exit is UP.
- Cells (3,0), (3,1), (4,0), (4,1) are M — assign escape actions to exit M zone.
- (5,4) cannot go UP (70% into (4,4) X) — go LEFT instead, only 15% slip-up risk.

4. OVERALL FLOW
- Top rows: move RIGHT along row 0-1 to reach column 2, then DOWN.
- Bottom rows: move UP toward row 3 highway, then RIGHT to (3,4), then UP to goal.
- Right side: (2,5)->LEFT and (3,5)->LEFT feed directly toward goal area.

Policy:
(0,0)=1  (0,1)=1  (0,2)=2  (0,3)=2  (0,4)=2  (0,5)=3
(1,0)=1  (1,1)=1  (1,2)=2  (1,3)=1  (1,4)=2  (1,5)=3
(2,0)=0  (2,1)=0  (2,2)=2  (2,3)=1  (2,4)=0  (2,5)=3
(3,0)=0  (3,1)=1  (3,2)=1  (3,3)=1  (3,4)=0  (3,5)=3
(4,0)=2  (4,1)=1  (4,2)=0  (4,3)=0  (4,4)=0  (4,5)=0
(5,0)=1  (5,1)=0  (5,2)=0  (5,3)=0  (5,4)=3  (5,5)=0

=== EXAMPLE 2 ===
Grid (6x6):
  0 1 2 3 4 5
0 S . . . G .
1 . F . . . .
2 X . . M M .
3 . X . . . .
4 . X F . . .
5 . . X . . .

Goal: (0,4)
Static obstacles: [(2,0), (3,1), (4,1), (5,2)]
Future goals: [(1,1), (4,2)]
Moving obstacles: [(2,3), (2,4)]
Stochastic: 70% forward, 15% slip-left, 15% slip-right

Reasoning:

1. SAFE APPROACH TO GOAL
Check each neighbor of (0,4) for slip safety:
- From (0,3) via RIGHT: slips hit wall-bounce and (1,3), both clear — SAFE.
- From (1,4) via UP: slips hit (1,3) and (1,5), both clear — SAFE.
- From (0,5) via LEFT: slips hit wall-bounce and (1,5), both clear — SAFE.
Multiple safe approaches — good. Policy can funnel from multiple directions.

2. MAIN CORRIDORS
- Row 0 express: (0,0)->RIGHT, (0,1)->RIGHT, (0,2)->RIGHT, (0,3)->RIGHT to goal.
  All slips are wall-bounces (up) or row-1 cells. Only risk: (0,1)->RIGHT has 15% slip into (1,1) F.
- Eastern column: cells on right side go UP through rows to reach row 0-1.
- Central: avoid M cells at (2,3) and (2,4). Route around via (3,3)->RIGHT, (3,4)->RIGHT,
  (3,5)->UP instead of going directly UP through M row.

3. PROBLEM ZONES
- TRAPPED POCKET: (3,0), (4,0), (5,0), (5,1) are walled in by X at (2,0), (3,1), (4,1), (5,2).
  No stochastically reliable exit. Assign best-effort actions but these cells effectively cannot reach goal.
- (2,1) is squeezed: UP->(1,1) F, LEFT->(2,0) X, DOWN->(3,1) X. Only option is RIGHT (70% safe).
  Slips hit F and X at 15% each — unavoidable.
- Cells near M row: (3,3) and (3,4) use RIGHT instead of UP to avoid 70% chance of entering M.

4. OVERALL FLOW
- Top half: row-0 express lane going RIGHT, row-1 feeds UP into row 0.
- Bottom-right: go UP through columns 3-5, routing around M cells.
- Bottom-left: trapped pocket, minimal reachability.

Policy:
(0,0)=1  (0,1)=1  (0,2)=1  (0,3)=1  (0,4)=0  (0,5)=3
(1,0)=0  (1,1)=0  (1,2)=1  (1,3)=0  (1,4)=0  (1,5)=0
(2,0)=0  (2,1)=1  (2,2)=0  (2,3)=0  (2,4)=0  (2,5)=0
(3,0)=2  (3,1)=0  (3,2)=0  (3,3)=1  (3,4)=1  (3,5)=0
(4,0)=2  (4,1)=1  (4,2)=0  (4,3)=0  (4,4)=0  (4,5)=0
(5,0)=0  (5,1)=3  (5,2)=1  (5,3)=0  (5,4)=0  (5,5)=0
"""
    )
    stochastic_example = f"\nExample: If you choose DOWN, there's a {prob_forward_pct}% chance of going DOWN, {prob_slip_left_pct}% chance of going LEFT, and {prob_slip_right_pct}% chance of going RIGHT.\n"
    moving_note = ", which could be anywhere along their patrol path" if k_obstacles else ""

    if is_feedback:
        parts = [
            f"A previous policy generated for this problem has the following probabilities for the requirements:\n",
            f"{probability_summary}\n",
            f"Previous policy:\n{policy_raw}\n",
            f"The following is a visualization of the previous policy:\n{policy_visual}\n",
            "Policy Legend:\n"
            "- ↑ = UP (action 0), → = RIGHT (action 1), ↓ = DOWN (action 2), ← = LEFT (action 3)\n"
            "- X↓ = Static obstacle with escape action (e.g., X↓ means obstacle, escape by going DOWN)\n"
            "- M→ = Moving obstacle with action (e.g., M→ means moving obstacle, action is RIGHT)\n"
            "- F→ = Future goal with escape action (e.g., F→ means future goal, escape by going RIGHT)\n"
            "- G = Current goal (destination)\n",
            "\nCompare this policy to the grid layout above to identify where actions lead toward obstacles or away from the goal.\n",
        ]
        if problems:
            parts.append(f"\nHints (possible issues with the previous policy):\n{problems}\n")
        feedback_block = "\n" + "\n".join(parts)
    else:
        feedback_block = ""

    return UNIFIED_TEMPLATE.format(
        size=size,
        grid_visual=grid_visual,
        examples_block=examples_block,
        s_obstacles=str(s_obstacles),
        f_goals=str(f_goals),
        k_obstacles=str(k_obstacles),
        moving_note=moving_note,
        goal=str(goal),
        total_states=total_states,
        prob_forward_pct=prob_forward_pct,
        prob_slip_left_pct=prob_slip_left_pct,
        prob_slip_right_pct=prob_slip_right_pct,
        stochastic_example=stochastic_example,
        feedback_block=feedback_block
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


def generate_policy_raw(size: int, policy: Dict[Tuple, int],
                        goal_state: Tuple[bool, ...]) -> str:
    """Generate raw policy listing: (x, y)=action for each cell.

    Args:
        size: Grid size
        policy: Dict mapping (x, y, goal_flags...) -> action
        goal_state: Current goal state tuple to filter relevant states

    Returns:
        Newline-separated string like "(0, 0)=1\\n(0, 1)=2\\n..."
    """
    parts = []
    for x in range(size):
        for y in range(size):
            state = (x, y) + goal_state
            action = policy.get(state, '?')
            parts.append(f"({x}, {y})={action}")
    return "\n".join(parts)


def generate_policy_visual(size: int, policy: Dict[Tuple, int],
                           goal_state: Tuple[bool, ...],
                           goal: Tuple[int, int],
                           s_obstacles: List[Tuple[int, int]],
                           f_goals: List[Tuple[int, int]],
                           k_obstacles: List[Tuple[int, int]] = None) -> str:
    """Generate ASCII visual of the policy actions.

    Args:
        size: Grid size
        policy: Dict mapping (x, y, goal_flags...) -> action
        goal_state: Current goal state tuple to filter relevant states
        goal: Current goal position
        s_obstacles: List of static obstacle positions
        f_goals: List of future goal positions
        k_obstacles: List of moving obstacle positions

    Returns:
        ASCII grid showing arrows for actions, with markers for special cells.
        Format uses 2-char cells: " →" for normal, "X↓" for obstacle+action,
        "F→" for future goal+action, "M→" for moving obstacle+action.
    """
    ACTION_ARROWS = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    # Create sets for quick lookup
    obstacle_set = set((obs[0], obs[1]) for obs in s_obstacles if 0 <= obs[0] < size and 0 <= obs[1] < size)
    future_goal_set = set((fg[0], fg[1]) for fg in f_goals if 0 <= fg[0] < size and 0 <= fg[1] < size)
    moving_set = set((obs[0], obs[1]) for obs in (k_obstacles or []) if 0 <= obs[0] < size and 0 <= obs[1] < size)

    # Build grid with 2-char cells
    grid = [[' ?' for _ in range(size)] for _ in range(size)]

    for x in range(size):
        for y in range(size):
            state = (x, y) + goal_state
            action_arrow = ACTION_ARROWS.get(policy.get(state), '?')

            if (x, y) == goal:
                # Current goal - just show G
                grid[x][y] = ' G'
            elif (x, y) in obstacle_set:
                # Obstacle with escape action
                grid[x][y] = 'X' + action_arrow
            elif (x, y) in moving_set:
                # Moving obstacle with action
                grid[x][y] = 'M' + action_arrow
            elif (x, y) in future_goal_set:
                # Future goal with escape action
                grid[x][y] = 'F' + action_arrow
            else:
                # Normal cell with action
                grid[x][y] = ' ' + action_arrow

    # Format with headers (2-char width per column)
    header = '  ' + ' '.join(f'{i:>2}' for i in range(size))
    rows = [f"{i} " + ' '.join(row) for i, row in enumerate(grid)]
    return header + '\n' + '\n'.join(rows)


def get_prompt(size: int, s_obstacles: List[Tuple[int, int]], f_goals: List[Tuple[int, int]],
               k_obstacles: List[Tuple[int, int]], goal: Tuple[int, int],
               prob_forward: float = 0.7, prob_slip_left: float = 0.15,
               prob_slip_right: float = 0.15) -> str:
    """Generate the initial prompt (backward-compatible wrapper around build_prompt)"""
    return build_prompt(
        size=size,
        s_obstacles=s_obstacles,
        f_goals=f_goals,
        k_obstacles=k_obstacles,
        goal=goal,
        prob_forward=prob_forward,
        prob_slip_left=prob_slip_left,
        prob_slip_right=prob_slip_right,
        is_feedback=False
    )


class StateAction(BaseModel):
    x: int = Field(..., description="x coordinate (row, 0-indexed)")
    y: int = Field(..., description="y coordinate (column, 0-indexed)")
    best_action: int = Field(..., ge=0, le=3, description="Best action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT")

class ActionPolicy(BaseModel):
    """Best action for each state in the grid"""
    states: List[StateAction]

# Keep old names as aliases for backwards compatibility during transition
StateQ = StateAction
QTables = ActionPolicy
