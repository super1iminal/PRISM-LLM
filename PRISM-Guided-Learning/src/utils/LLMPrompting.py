from typing import List, Tuple, Dict
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from config.Settings import get_threshold_for_key

UNIFIED_PROMPT_TEXT = """You are an expert path planner{role_block}

Your task is to create a best-action policy for a grid world — choose one action per cell that guides the agent from any position to the goal while avoiding static obstacles, future goals, and moving obstacles.
{results_block}
The grid world is {size} x {size}. Here is the visual layout:

{grid_visual}

{legend_label}Legend:
- 'S' = Start position (0,0) - the initial state
- 'G' = Current goal position you must reach
- 'X' = Static obstacle (CANNOT enter - you will bounce back)
- 'F' = Future goal (treat as obstacle for now - avoid it)
- 'M' = Moving obstacle patrol path (avoid if possible)
- '.' = Empty cell you can move through
{policy_block}
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
{examples_block}{problems_block}
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
This means {paths_word}paths should be ROBUST - avoid routes that pass adjacent to obstacles since slips could cause collisions.

{closing_block}
"""

UNIFIED_TEMPLATE = PromptTemplate(
    template=UNIFIED_PROMPT_TEXT,
    input_variables=["role_block", "results_block", "size", "grid_visual",
                     "legend_label", "policy_block", "examples_block",
                     "problems_block", "s_obstacles", "f_goals", "k_obstacles",
                     "moving_note", "goal", "total_states",
                     "prob_forward_pct", "prob_slip_left_pct", "prob_slip_right_pct",
                     "stochastic_example", "paths_word", "closing_block"]
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


def format_probability_summary(prism_probs: Dict[str, float]) -> str:
    """Format probability summary for feedback.

    Hides moving-obstacle avoidance metrics that already meet their
    threshold so the LLM doesn't try to optimise a non-problem.
    """
    lines = ["Current verification probabilities:"]
    for key, prob in prism_probs.items():
        threshold = get_threshold_for_key(key)
        # Don't show obstacle avoidance metrics that meet threshold —
        # they distract the LLM from the real issues.
        if key.startswith('avoid_moving') and prob >= threshold:
            continue
        status = "✓" if prob >= threshold else "✗"
        lines.append(f"  {status} {key}: {prob:.4f}")
    return "\n".join(lines)


def build_prompt(size: int, s_obstacles: List[Tuple[int, int]], f_goals: List[Tuple[int, int]],
                 k_obstacles: List[Tuple[int, int]], goal: Tuple[int, int],
                 prob_forward: float = 0.7, prob_slip_left: float = 0.15,
                 prob_slip_right: float = 0.15,
                 is_feedback: bool = False,
                 probability_summary: str = "",
                 policy_visual: str = "",
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

    if is_feedback:
        role_block = ". Your previous plan did NOT achieve perfect probabilities."
        results_block = f"\nPREVIOUS ATTEMPT RESULTS:\n{probability_summary}\n"
        legend_label = "Grid "
        policy_block = f"""
YOUR PREVIOUS POLICY (visualized as arrows):
{policy_visual}

Policy Legend:
- ↑ = UP (action 0), → = RIGHT (action 1), ↓ = DOWN (action 2), ← = LEFT (action 3)
- X↓ = Static obstacle with escape action (e.g., X↓ means obstacle, escape by going DOWN)
- F→ = Future goal with escape action (e.g., F→ means future goal, escape by going RIGHT)
- G = Current goal (destination)

Compare this policy to the grid layout above to identify where actions lead toward obstacles or away from the goal.

"""
        examples_block = f"""
ACTION EXAMPLES:
From (0,0): UP→(0,0), RIGHT→(0,1), DOWN→(1,0), LEFT→(0,0)
From (0,1): UP→(0,1), RIGHT→(0,2), DOWN→(1,1), LEFT→(0,0)
From (1,0): UP→(0,0), RIGHT→(1,1), DOWN→(2,0), LEFT→(1,0)

"""
        stochastic_example = f"\nExample: If you choose DOWN, there's a {prob_forward_pct}% chance of going DOWN, {prob_slip_left_pct}% chance of going LEFT, and {prob_slip_right_pct}% chance of going RIGHT.\n"
        moving_note = ", which could be anywhere along the patrol path" if k_obstacles else ""
        paths_word = ""

        if problems:
            # Feedback (full) style
            problems_block = f"""
HINTS (possible issues):
{problems}

"""
            closing_block = (
                f"CRITICAL REQUIREMENTS:\n"
                f"1. You MUST provide the best action for ALL {total_states} states in the {size}x{size} grid\n"
                "2. Never plan a path through obstacles (X) or future goals (F)\n"
                "2a. Avoid moving obstacles (M) along their patrol paths if possible\n"
                "3. The best action should create a path from ANY position to the goal\n"
                "4. If a cell is an obstacle (X, M) or future goal (F), provide an escape action (how to exit if accidentally there due to stochastic slip)\n\n"
                "Now provide the best action (0-3) for each state."
            )
        else:
            # FeedbackMinus style - no problems block
            problems_block = ""
            closing_block = (
                f"CRITICAL REQUIREMENTS:\n"
                f"1. You MUST provide the best action for ALL {total_states} states in the {size}x{size} grid\n"
                "2. Never plan a path through obstacles (X) or future goals (F)\n"
                "2a. Avoid moving obstacles (M) along their patrol paths if possible\n"
                "3. The best action should create a path from ANY position to the goal\n"
                "4. If a cell is an obstacle (X, M) or future goal (F), provide an escape action (how to exit if accidentally there due to stochastic slip)\n\n"
                "Now provide the best action (0-3) for each state."
            )
    else:
        # Initial (Vanilla) style
        role_block = " working on formulating paths that meet formal requirements."
        results_block = ""
        legend_label = ""
        policy_block = "\n"
        examples_block = f"""
ACTION EXAMPLES:
From (0,0): UP→(0,0), RIGHT→(0,1), DOWN→(1,0), LEFT→(0,0)
From (0,1): UP→(0,1), RIGHT→(0,2), DOWN→(1,1), LEFT→(0,0)
From (1,0): UP→(0,0), RIGHT→(1,1), DOWN→(2,0), LEFT→(1,0)

"""
        problems_block = ""
        moving_note = ", which could be anywhere along the trajectory each timestep" if k_obstacles else ""
        stochastic_example = f"\nExample: If you choose DOWN, there's a {prob_forward_pct}% chance of going DOWN, {prob_slip_left_pct}% chance of going LEFT, and {prob_slip_right_pct}% chance of going RIGHT.\n"
        paths_word = "your "
        closing_block = (
            f"CRITICAL REQUIREMENTS:\n"
            f"1. You MUST provide the best action for ALL {total_states} states in the {size}x{size} grid\n"
            "2. Never plan a path through obstacles (X) or future goals (F)\n"
            "2a. Avoid moving obstacles (M) along their patrol paths if possible\n"
            "3. The best action should create a path from ANY position to the goal\n"
            "4. If a cell is an obstacle (X, M) or future goal (F), provide an escape action (how to exit if accidentally there due to stochastic slip)\n\n"
            "Now provide the best action (0-3) for each state."
        )

    return UNIFIED_TEMPLATE.format(
        role_block=role_block,
        results_block=results_block,
        size=size,
        grid_visual=grid_visual,
        legend_label=legend_label,
        policy_block=policy_block,
        examples_block=examples_block,
        problems_block=problems_block,
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
        paths_word=paths_word,
        closing_block=closing_block
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


def generate_policy_visual(size: int, policy: Dict[Tuple, int],
                           goal_state: Tuple[bool, ...],
                           goal: Tuple[int, int],
                           s_obstacles: List[Tuple[int, int]],
                           f_goals: List[Tuple[int, int]]) -> str:
    """Generate ASCII visual of the policy actions.

    Args:
        size: Grid size
        policy: Dict mapping (x, y, goal_flags...) -> action
        goal_state: Current goal state tuple to filter relevant states
        goal: Current goal position
        s_obstacles: List of static obstacle positions
        f_goals: List of future goal positions

    Returns:
        ASCII grid showing arrows for actions, with markers for special cells.
        Format uses 2-char cells: " →" for normal, "X↓" for obstacle+action, "F→" for future goal+action.
    """
    ACTION_ARROWS = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    # Create sets for quick lookup
    obstacle_set = set((obs[0], obs[1]) for obs in s_obstacles if 0 <= obs[0] < size and 0 <= obs[1] < size)
    future_goal_set = set((fg[0], fg[1]) for fg in f_goals if 0 <= fg[0] < size and 0 <= fg[1] < size)

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
