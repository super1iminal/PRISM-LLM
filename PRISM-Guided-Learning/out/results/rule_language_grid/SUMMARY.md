# Rule language on grid_20_large (Haiku 5.5) (auto-generated 2026-10-08 09:21)

Gridworld, 20 grids of sizes 9 to 13 (20 solvable), anthropic/claude-haiku-5.5, thinking on, restart on slow progress, obstacle phase visible. The base vocabulary has 2 runs; each general-vocabulary condition is one unseeded run, so the paired tests are exploratory. General vocabulary: instance constants (grid size, goal coordinates), per-action features (v1: wall, hazard, toward; v2 adds obstacle and risky; closer where named), rules with action `any`, and no numbers but 0 and 1. Five general-vocabulary grids (v1: 14, 18; v2: 9, 19; v2 + closer: 19) ended with an error because a PRISM timeout did not take effect (fixed since; DECISIONS.md): they count as unsolved, and are left out of the means and the paired tests.

Regenerate: `python viz/ablation_summary.py configs/plot/rule_language_grid.yaml`.

## Overview

![conditions](conditions.png)

## Distributions and costs (medians, interquartile range)

![costs](costs.png)

## Budget curves

![budget](budget.png)

## Loop mechanics (symbolic)

![mechanics](mechanics.png)

## Outcomes (per grid, seeds pooled)

| cond | change | seeds | solved (of solvable) | req. met: mean (seed range) | median [IQR] | best case | shortfall: median [IQR] | uncovered % | vs | Δ met [95% CI] | p | feedback rounds improved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| haiku_large_R4 | Base vocabulary (coordinates) | 2 | 12/20 | 8.18 (8.15 to 8.20) | 9 [7 to 9] | 8.18 | 0.00 [0.00 to 0.07] | 0 |  |  |  | 78% |
| haiku_rl_grid_large | General v1: constants, wall/hazard/toward, any | 1 | 6/20 | 6.56 (6.56 to 6.56) | 6 [5 to 9] | 8.00 | 0.73 [0.00 to 1.60] | 0 | haiku_large_R4 | -1.53 [-2.61, -0.53] | 0.013 | 52% |
| haiku_rl_grid_large_closer | General v1 + shortest-path feature closer | 1 | 9/20 | 7.90 (7.90 to 7.90) | 8 [7 to 9] | 8.55 | 0.02 [0.00 to 0.10] | 0 | haiku_large_R4 | -0.28 [-0.65, +0.15] | 0.255 | 48% |
| haiku_rl2_grid_large | General v2: v1 + obstacle, risky | 1 | 3/20 | 6.39 (6.39 to 6.39) | 7 [5 to 8] | 8.11 | 0.82 [0.45 to 2.18] | 0 | haiku_large_R4 | -1.83 [-2.97, -0.78] | 0.005 | 60% |
| haiku_rl2_grid_large_closer | General v2 + closer | 1 | 9/20 | 7.95 (7.95 to 7.95) | 8 [7 to 9] | 8.42 | 0.02 [0.00 to 0.14] | 0 | haiku_large_R4 | -0.26 [-0.68, +0.21] | 0.331 | 29% |

## Costs (mean per grid)

| cond | input tokens | output tokens | PRISM time (s) | wall time (min) | rules (final policy) | spend (USD at Haiku 5.5 prices) |
|---|---|---|---|---|---|---|
| haiku_large_R4 | 16.3k | 26.0k | 8 | 2.8 | 32.0 | 0.0146 |
| haiku_rl_grid_large | 19.2k | 13.4k | 95 | 3.7 | 10.6 | 0.00863 |
| haiku_rl_grid_large_closer | 17.0k | 7.7k | 79 | 2.3 | 5.5 | 0.00557 |
| haiku_rl2_grid_large | 21.7k | 13.9k | 114 | 4.1 | 10.8 | 0.00912 |
| haiku_rl2_grid_large_closer | 15.1k | 6.2k | 32 | 1.3 | 6.5 | 0.00462 |

## Requirements met after k rounds

| cond | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| haiku_large_R4 | 4.78 | 7.30 | 7.65 | 7.97 | 8.18 |
| haiku_rl_grid_large | 5.39 | 6.39 | 6.50 | 6.56 | 6.56 |
| haiku_rl_grid_large_closer | 6.80 | 7.40 | 7.85 | 7.90 | 7.90 |
| haiku_rl2_grid_large | 5.06 | 5.72 | 6.39 | 6.39 | 6.39 |
| haiku_rl2_grid_large_closer | 7.37 | 7.47 | 7.53 | 7.63 | 7.95 |

