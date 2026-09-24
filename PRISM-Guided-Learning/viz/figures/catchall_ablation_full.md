# Catch-all ablation (instruction and example removed): qwen3:14b, 20 gridworlds

| metric | Legacy | Symbolic: catch-all asked | Symbolic: no catch-all at all |
|---|---|---|---|
| solved (all requirements; symbolic worst case) | 0/20 | 1/20 | 0/20 |
| requirements met, mean | 4.30 | 5.25 | 4.50 |
| shortfall, mean | 3.162 | 2.110 | 2.623 |
| output tokens, mean | 12150 | 5029 | 5182 |
| wall time (s), mean | 415 | 170 | 176 |
| best-case requirements met, mean | n/a | 5.40 | 5.25 |
| rounds in first (improved) | n/a | 20 (20) | 20 (20) |
| rounds in retry (improved) | n/a | 10 (6) | 17 (7) |
| rounds in refine (improved) | n/a | 63 (22) | 52 (12) |
| rounds in extend (improved) | n/a | 4 (4) | 11 (2) |
| uncovered situations, mean % over rounds | n/a | 9.3% | 25.9% |
| rounds with any uncovered situation | n/a | 22/97 | 46/100 |
| rules per round, mean | n/a | 36.7 | 41.2 |
| invalid answers, total | n/a | 3 | 0 |
