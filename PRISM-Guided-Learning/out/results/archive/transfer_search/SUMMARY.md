# Transfer search: held-out grids 12-19 of grid_20_large

Held-out targets: 12-19 of `grid_20_large.csv` (8). Mean over runs, range in parentheses. Certified: held-out targets certified per rule set. Met: share of requirements met in the worst case on them.

| condition | runs | sets solved | certified (of 8) | best rule set | met | prompt tokens / set | output tokens / set | $ / set |
|---|---|---|---|---|---|---|---|---|
| haiku_mi_grid_large | 3 | 0% | 0.00 | 0 | 61% (61%–61%) | 31,440 (30,860–31,792) | 6,787 (6,262–7,785) | 0.007 (0.006–0.007) |
| haiku_mi_grid_large_closer | 3 | 0% | 5.33 (5.25–5.50) | 6 (6–7) | 89% (88%–90%) | 33,158 (31,788–33,978) | 6,488 (5,772–7,816) | 0.007 (0.006–0.007) |
| sonnet5_5_mi_grid_large | 1 | 25% | 1.75 | 2 | 72% | 50,883 | 66,941 | 1.157 |

Share of rule sets certifying each held-out target:

| condition | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|
| haiku_mi_grid_large | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| haiku_mi_grid_large_closer | 0% | 100% | 75% | 75% | 75% | 100% | 100% | 8% |
| sonnet5_5_mi_grid_large | 0% | 0% | 0% | 50% | 0% | 25% | 100% | 0% |
