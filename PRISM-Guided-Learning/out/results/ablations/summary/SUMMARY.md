# Ablation results (auto-generated 2026-10-03 08:21)

Gridworld, 20 grids (19 solvable), qwen3 14B, 5 rounds, obstacle phase visible to rules and legacy. Symbolic numbers are worst case. Paired tests compare per-grid means (seeds averaged) against the reference with a sign-flip permutation test; with 20 grids and 2 seeds, treat p > 0.05 as noise.

Regenerate: `python viz/ablation_summary.py`.

**Pending runs:** S1/seed_1, B1/seed_1

## Overview

![conditions](conditions.png)

## Budget curves

![budget](budget.png)

## Loop mechanics (symbolic)

![mechanics](mechanics.png)

## Table

| cond | change | seeds | solved | req. met (seed range) | best case | shortfall | uncovered % | vs | Δ met [95% CI] | p | refine improved | out tokens | min/grid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B2 | Symbolic defaults | 2 | 0.0 | 4.45 (4.20 to 4.70) | 4.75 | 2.43 | 3 |  |  |  | 33% | 7.0k | 3.7 |
| R1 | Never restart | 2 | 0.0 | 4.12 (3.95 to 4.30) | 4.62 | 2.93 | 6 | B2 | -0.33 [-0.65, +0.03] | 0.102 | 25% | 6.8k | 3.6 |
| R2 | Restart after 1 stall | 2 | 0.0 | 4.72 (4.40 to 5.05) | 4.80 | 2.44 | 1 | B2 | +0.28 [-0.20, +0.75] | 0.319 | 32% | 6.4k | 3.3 |
| R3 | Restart every 3rd round | 2 | 0.0 | 4.65 (4.60 to 4.70) | 4.67 | 2.42 | 2 | B2 | +0.20 [-0.03, +0.45] | 0.172 | 33% | 6.3k | 3.4 |
| R4 | Restart when gain < 0.05 | 2 | 0.0 | 4.95 (4.80 to 5.10) | 5.03 | 2.34 | 1 | B2 | +0.50 [+0.05, +0.93] | 0.055 | 32% | 6.2k | 3.2 |
| R5 | Always restart (no feedback) | 2 | 0.0 | 4.72 (4.65 to 4.80) | 4.78 | 2.44 | 2 | B2 | +0.28 [-0.20, +0.78] | 0.342 |  | 6.2k | 3.1 |

## Requirements met after k rounds

| cond | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| B2 | 2.98 | 3.67 | 3.85 | 4.25 | 4.45 |
| R1 | 2.75 | 3.75 | 3.85 | 4.03 | 4.12 |
| R2 | 2.85 | 3.52 | 4.17 | 4.53 | 4.72 |
| R3 | 2.95 | 3.75 | 4.00 | 4.47 | 4.65 |
| R4 | 3.00 | 3.77 | 4.42 | 4.78 | 4.95 |
| R5 | 3.27 | 3.75 | 4.20 | 4.55 | 4.72 |

