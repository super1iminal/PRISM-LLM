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
# Windows: "C:\\Program Files\\prism-4.9\\bin\\prism.bat"
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

**Q-values to transition probabilities**: Uses softmax with temperature τ=0.1

**LTL Score**: Weighted combination of individual goal probabilities, sequential goal achievement, and obstacle avoidance probabilities.

## Key Conventions

- **Grid coordinates**: (0,0) is top-left; row increases downward, column rightward
- **Actions**: [0=UP, 1=RIGHT, 2=DOWN, 3=LEFT]
- **State representation**: `(x, y, goal1_reached, goal2_reached, ...)`
- **Goals must be reached in sequence**

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
