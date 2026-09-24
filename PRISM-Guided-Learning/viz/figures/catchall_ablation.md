# Catch-all instruction ablation: qwen3:14b, 20 gridworlds, grouped by grid size

| metric | Legacy | Symbolic: catch-all asked | Symbolic: no catch-all |
|---|---|---|---|
| solved (all requirements; symbolic worst case) | 0/20 | 1/20 | 0/20 |
| requirements met, mean | 4.30 | 5.25 | 5.05 |
| shortfall, mean | 3.162 | 2.110 | 1.949 |
| output tokens, mean | 12150 | 5029 | 5038 |
| wall time (s), mean | 415 | 170 | 179 |
| best-case requirements met, mean | n/a | 5.40 | 5.55 |
| rounds in first (improved) | n/a | 20 (20) | 20 (20) |
| rounds in retry (improved) | n/a | 10 (6) | 15 (7) |
| rounds in refine (improved) | n/a | 63 (22) | 59 (22) |
| rounds in extend (improved) | n/a | 4 (4) | 6 (3) |
| uncovered situations, mean % over rounds | n/a | 9.3% | 12.8% |
| rounds with any uncovered situation | n/a | 22/97 | 28/100 |
| rules per round, mean | n/a | 36.7 | 37.2 |
| invalid answers, total | n/a | 3 | 0 |
