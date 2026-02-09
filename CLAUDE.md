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
- `out/results/{timestamp}/` - Parquet results and PNG plots

## Data Format

CSV files in `data/` with columns: `n`, `goals`, `static`, `BFS_steps`
- `n`: grid size
- `goals`: `"[{1: (2, 2)}, {2: (3, 3)}]"` (ordered dict of goal positions)
- `static`: `"[(1,1), (2,2)]"` (static obstacle positions)
- `BFS_steps`: optimal path length

## Data Collection by Planner

Both VanillaLLMPlanner and FeedbackLLMPlanner return per-iteration data in a unified format for consistency.

### VanillaLLMPlanner & FeedbackLLMPlanner (Unified Format)
| Field | Type | Description |
|-------|------|-------------|
| `LTL_Score` | float | Final LTL verification score |
| `Prism_Probabilities` | dict | Final probabilities (goal reachability, sequences, obstacle avoidance) |
| `Evaluation_Time` | float | Total wall-clock time in seconds |
| `Iterations_Used` | int | Number of iterations (always 1 for Vanilla, 1+ for Feedback) |
| `Iteration_Times` | list[float] | Total time per iteration (seconds) |
| `Iteration_PRISM_Times` | list[float] | PRISM verification time per iteration (seconds) |
| `Iteration_LLM_Times` | list[float] | LLM inference time per iteration (seconds) |
| `Iteration_Prism_Probs` | list[dict] | Full probability dict at each iteration |
| `Iteration_Mistakes` | list[int] | Count of probabilities below threshold per iteration |
| `Iteration_Costs` | list[float] | Sum of (1.0 - prob) for probs < 1.0 per iteration |
| `Success` | bool | True if all probabilities meet threshold |

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

### Data Saved to Parquet (Evaluator.py)

Results are saved as Parquet files with a MultiIndex DataFrame structure `(sample_id, iteration)`:

| Column | Description |
|--------|-------------|
| `sample_id` | Index of the gridworld sample (0-indexed) |
| `iteration` | Iteration number (1-indexed, 1 = initial) |
| `size` | Grid size |
| `goals` | Number of goals |
| `obstacles` | Number of static obstacles |
| `complexity` | BFS optimal path length |
| `iteration_time` | Total time for this iteration (seconds) |
| `prism_time` | PRISM verification time for this iteration (seconds) |
| `llm_time` | LLM inference time for this iteration (seconds) |
| `mistakes` | Count of probabilities below threshold at this iteration |
| `cost` | Sum of (1.0 - prob) for probs < 1.0 at this iteration |
| `prob_goal1`, `prob_goal2`, ... | Individual goal reachability probabilities |
| `prob_seq_1_before_2`, ... | Sequence ordering probabilities |
| `prob_complete_sequence` | Complete sequence probability |
| `prob_avoid_obstacle` | Obstacle avoidance probability |
| `is_final` | True if this is the final iteration |
| `final_ltl_score` | Final LTL score (same for all iterations of a sample) |
| `success` | True if all probabilities meet threshold (same for all iterations) |

**Example Analysis:**
```python
import pandas as pd
df = pd.read_parquet("results.parquet")

# Get final iterations only
finals = df[df["is_final"]]

# Average metrics by iteration number
df.groupby("iteration").mean()

# Compare first vs final iteration probabilities
first_iters = df.xs(1, level="iteration")
```
