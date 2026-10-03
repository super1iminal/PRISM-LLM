# Ablation results (auto-generated 2026-10-03 14:46)

Gridworld, 20 grids (19 solvable), qwen3 14B, 5 rounds, obstacle phase visible to rules and legacy. Symbolic numbers are worst case. Paired tests compare per-grid means (seeds averaged) against the reference with a sign-flip permutation test; with 20 grids and 2 seeds, treat p > 0.05 as noise.

Regenerate: `python viz/ablation_summary.py`.

## Overview

![conditions](conditions.png)

## Budget curves

![budget](budget.png)

## Loop mechanics (symbolic)

![mechanics](mechanics.png)

## Table

| cond | change | seeds | solved | req. met (seed range) | best case | shortfall | uncovered % | vs | Δ met [95% CI] | p | feedback rounds improved | out tokens | min/grid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B2 | Symbolic defaults | 2 | 0.0 | 4.45 (4.20 to 4.70) | 4.75 | 2.43 | 3 |  |  |  | 33% | 7.0k | 3.7 |
| R1 | Never restart | 2 | 0.0 | 4.12 (3.95 to 4.30) | 4.62 | 2.93 | 6 | B2 | -0.33 [-0.65, +0.03] | 0.102 | 26% | 6.8k | 3.6 |
| R2 | Restart after 1 stall | 2 | 0.0 | 4.72 (4.40 to 5.05) | 4.80 | 2.44 | 1 | B2 | +0.28 [-0.20, +0.75] | 0.319 | 33% | 6.4k | 3.3 |
| R3 | Restart every 3rd round | 2 | 0.0 | 4.65 (4.60 to 4.70) | 4.67 | 2.42 | 2 | B2 | +0.20 [-0.03, +0.45] | 0.172 | 33% | 6.3k | 3.4 |
| R4 | Restart when gain < 0.05 | 2 | 0.0 | 4.95 (4.80 to 5.10) | 5.03 | 2.34 | 1 | B2 | +0.50 [+0.05, +0.93] | 0.055 | 35% | 6.2k | 3.2 |
| R5 | Always restart (no feedback) | 2 | 0.0 | 4.72 (4.65 to 4.80) | 4.78 | 2.44 | 2 | B2 | +0.28 [-0.20, +0.78] | 0.342 |  | 6.2k | 3.1 |
| S1 | Results table only | 2 | 0.0 | 4.08 (3.75 to 4.40) | 4.53 | 2.74 | 4 | B2 | -0.38 [-0.82, +0.07] | 0.171 | 20% | 7.2k | 3.6 |
| S4 | No examples in the prompt | 2 | 0.0 | 4.67 (4.65 to 4.70) | 4.80 | 2.66 | 5 | B2 | +0.23 [-0.28, +0.78] | 0.479 | 39% | 6.0k | 3.1 |
| S5 | Blame by one-step regret | 2 | 0.0 | 4.58 (4.40 to 4.75) | 4.58 | 2.69 | 1 | B2 | +0.12 [-0.33, +0.57] | 0.683 | 23% | 6.7k | 3.4 |
| V1 | Blame on random rules/states | 2 | 0.0 | 4.17 (4.15 to 4.20) | 4.50 | 2.77 | 4 | B2 | -0.28 [-0.75, +0.17] | 0.319 | 32% | 6.1k | 3.1 |
| V2 | No blame section | 2 | 0.0 | 4.40 (4.15 to 4.65) | 4.88 | 2.53 | 8 | B2 | -0.05 [-0.65, +0.50] | 0.931 | 26% | 7.2k | 3.7 |

## Requirements met after k rounds

| cond | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| B2 | 2.98 | 3.67 | 3.85 | 4.25 | 4.45 |
| R1 | 2.75 | 3.75 | 3.85 | 4.03 | 4.12 |
| R2 | 2.85 | 3.52 | 4.17 | 4.53 | 4.72 |
| R3 | 2.95 | 3.75 | 4.00 | 4.47 | 4.65 |
| R4 | 3.00 | 3.77 | 4.42 | 4.78 | 4.95 |
| R5 | 3.27 | 3.75 | 4.20 | 4.55 | 4.72 |
| S1 | 2.98 | 3.58 | 3.73 | 4.00 | 4.08 |
| S4 | 3.35 | 3.92 | 4.35 | 4.55 | 4.67 |
| S5 | 3.17 | 3.85 | 4.08 | 4.40 | 4.58 |
| V1 | 2.83 | 3.27 | 3.50 | 3.90 | 4.17 |
| V2 | 3.00 | 3.88 | 3.98 | 4.22 | 4.40 |

## UUV (U1: symbolic defaults on the paper's two scenarios, with the energy budget)

![uuv](uuv.png)

| seed | scenario | solved (worst case) | rounds | rules | final worst-case values |
|---|---|---|---|---|---|
| seed_1 | North Sea | True | 1 | 25 | no_thruster_failure 0.673, done_in_time 0.807, energy_budget 26.182 |
| seed_1 | Caribbean | False | 5 | 25 | no_thruster_failure 0.348, done_in_time 0.860, energy_budget 62.992 |
| seed_2 | North Sea | True | 1 | 25 | no_thruster_failure 0.673, done_in_time 0.807, energy_budget 26.182 |
| seed_2 | Caribbean | False | 5 | 25 | no_thruster_failure 0.348, done_in_time 0.860, energy_budget 62.992 |

