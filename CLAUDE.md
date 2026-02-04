# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PRISM-Guided-Learning combines Large Language Models with Q-Learning and Probabilistic Model Checking (PRISM) to solve grid-world navigation problems with formal probabilistic guarantees. The agent navigates from (0,0) to multiple goals in sequence while avoiding static and moving obstacles.

## Setup Commands

```bash
# Create conda environment
conda env create -f prism_portable.yml -n prism3
conda activate prism3

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Update PRISM path in src/config/Settings.py
# Windows: "C:\\Program Files\\prism-4.10\\bin\\prism.bat"
# Linux: "/path/to/prism/bin/prism"
```

## Running the Project

```bash
cd PRISM-Guided-Learning

# Run main evaluation (configure models and dataset in Evaluator.py)
python src/Evaluator.py

# Generate result plots
python src/PlotResults.py
```

To change the dataset, modify line 30 in `src/Evaluator.py`. To change which models run, modify the `models` list on line 33.

## Architecture

```
Evaluator.py (entry point)
    │
    ├── DataLoader → loads grid configs from CSV (n, goals, static obstacles, BFS_steps)
    │
    ├── Planners (choose one):
    │   ├── VanillaLLMPlanner    → single LLM call per goal, no feedback
    │   ├── FeedbackLLMPlanner   → iterative LLM with PRISM feedback loop
    │   ├── RLCounterfactual     → Q-learning with PMC verification
    │   └── UniformPlanner       → random baseline
    │
    └── Verification Pipeline:
        ├── PrismModelGenerator → converts Q-table to PRISM DTMC model
        ├── PrismVerifier       → executes PRISM binary
        └── SimplifiedVerifier  → extracts LTL score from output
```

**Q-values to transition probabilities**: Uses stochastic slip model with deterministic policy (argmax of Q-values) and stochastic execution: 0.7 forward, 0.15 left, 0.15 right

**LTL Score**: Weighted combination of individual goal probabilities, sequential goal achievement, and obstacle avoidance probabilities.

## Key Conventions

- **Grid coordinates**: (0,0) is top-left; row increases downward, column rightward
- **Actions**: [0=UP, 1=RIGHT, 2=DOWN, 3=LEFT]
- **State representation**: `(x, y, goal1_reached, goal2_reached, ...)`
- **Goals must be reached in sequence**

## Visualization Utilities

Located in `src/utils/LLMPrompting.py`:

### `generate_grid_visual()`
Creates ASCII grid showing the environment layout:
```
  0 1 2 3 4
0 S . . . .
1 . X . . .
2 . . G . .
3 . . . F .
4 . . . . .
```
- `S` = Start (0,0), `G` = Current goal, `X` = Static obstacle, `F` = Future goal, `M` = Moving obstacle, `.` = Empty

### `generate_policy_visual()`
Creates ASCII grid showing policy actions as arrows, with escape actions visible at obstacles/future goals:
```
    0  1  2  3  4
0   →  → X↓  ↓  ↓
1   →  →  ↓  ←  ↓
2   ↑  →  G  ←  ↓
3   ↑  →  → F↓  ↓
4   ↑  ↑  ↑  ↑  ←
```
- ` →` = Normal cell with action (↑=UP, →=RIGHT, ↓=DOWN, ←=LEFT)
- `X↓` = Static obstacle with escape action (here: escape DOWN)
- `F↓` = Future goal with escape action
- ` G` = Current goal (destination)

Used by `FeedbackLLMPlanner` to show the LLM its previous policy visually, enabling spatial reasoning about where actions lead toward obstacles or away from goals.

## Output Locations

- `out/logs/` - execution logs per worker
- `out/eval_results/` - evaluation metrics
- `out/results/{timestamp}/` - CSV results and PNG plots

## Data Format

CSV files in `data/` with columns: `n`, `goals`, `static`, `BFS_steps`
- `n`: grid size
- `goals`: `"[{1: (2, 2)}, {2: (3, 3)}]"` (ordered dict of goal positions)
- `static`: `"[(1,1), (2,2)]"` (static obstacle positions)
- `BFS_steps`: optimal path length

## Data Collection by Planner

### VanillaLLMPlanner
| Field | Type | Description |
|-------|------|-------------|
| `LTL_Score` | float | Final LTL verification score |
| `Prism_Probabilities` | dict | Individual probabilities (goal reachability, sequences, obstacle avoidance) |
| `Evaluation_Time` | float | Total wall-clock time in seconds |
| `Total_PRISM_Time` | float | Time spent on PRISM verification (seconds) |
| `Total_LLM_Time` | float | Time spent on LLM inference (seconds) |
| `Total_Mistakes` | int | Count of probabilities below threshold (default 0.9) |
| `Total_Cost` | float | Sum of (1.0 - prob) for all probs < 1.0 |
| `Success` | bool | True if all probabilities meet threshold |

### FeedbackLLMPlanner
| Field | Type | Description |
|-------|------|-------------|
| `LTL_Score` | float | Final LTL verification score |
| `Prism_Probabilities` | dict | Individual probabilities |
| `Evaluation_Time` | float | Total wall-clock time in seconds |
| `Iterations_Used` | int | Number of LLM feedback iterations (1 = initial only, 2+ = with feedback) |
| `Total_PRISM_Time` | float | Total time spent on PRISM verification (seconds) |
| `Total_LLM_Time` | float | Total time spent on LLM inference (seconds) |
| `Iteration_Times` | list[float] | Time taken per iteration (seconds) |
| `Avg_Iteration_Time` | float | Average time per iteration (seconds) |
| `Total_Mistakes` | int | Cumulative count of probabilities below threshold across all iterations |
| `Total_Cost` | float | Cumulative sum of (1.0 - prob) for all probs < 1.0 across all iterations |
| `Iteration_Mistakes` | list[int] | Number of probabilities below threshold per iteration |
| `Iteration_Costs` | list[float] | Cost value (sum of 1.0 - prob) per iteration |
| `Success` | bool | True if all probabilities meet threshold at end |

### RLCounterfactual
| Field | Type | Description |
|-------|------|-------------|
| `LTL_Score` | float | Best LTL verification score achieved |
| `Prism_Probabilities` | dict | Individual probabilities |
| `Evaluation_Time` | float | Time taken in seconds |
| `Episode_Rewards` | list | Rewards per episode during training |
| `Training_Stats` | dict | Additional training statistics |

### Prism_Probabilities Dict Structure
```python
{
    'goal1': float,              # P(reach goal 1)
    'goal2': float,              # P(reach goal 2)
    'seq_1_before_2': float,     # P(goal 1 before goal 2)
    'complete_sequence': float,  # P(all goals in order)
    'avoid_obstacle': float      # P(never hit obstacle)
}
```

### Data Saved to CSV (Evaluator.py)
| Column | Source |
|--------|--------|
| `ltl_score` | `result['LTL_Score']` |
| `prism_probabilities` | `result['Prism_Probabilities']` |
| `evaluation_time` | `result['Evaluation_Time']` |
| `size` | `gridworld.size` |
| `goals` | `len(gridworld.goals)` |
| `obstacles` | `len(gridworld.static_obstacles)` |
| `complexity` | `expected_steps` (BFS optimal path length) |

**Note:** `Iterations_Used` (Feedback) and `Episode_Rewards`/`Training_Stats` (RL) are returned but not currently saved to CSV.
