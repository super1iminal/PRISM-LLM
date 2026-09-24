# Legacy vs symbolic: end-to-end comparison

- Legacy run: `out\results\legacy_grid20`
- Symbolic run: `out\results\symbolic_grid20_capped`
- Samples: 20

Success = all requirements meet their thresholds (symbolic: in the **worst case** over all completions).

| metric | legacy (per-state) | symbolic (rules) |
|---|---|---|
| success | 0/20 | 1/20 |
| success in best case only | n/a | 1/20 |
| mean requirements met (of 9) | 4.30 | 5.25 (best case 5.40) |
| mean total shortfall below thresholds | 3.162 | 2.110 (best case 2.025) |
| mean iterations | 5.00 | 4.85 |
| mean LLM calls | 15.0 | 5.0 |
| mean output tokens | 12150 | 5029 |
| mean prompt tokens | 24313 | 14806 |
| mean LLM time (s) | 408.1 | 160.9 |
| mean PRISM time (s) | 6.8 | 7.4 |
| mean wall time per sample (s) | 414.9 | 169.7 |
| mean final rules | n/a | 34.6 |
| invalid LLM answers (total) | n/a | 3 |

## Mean final probability per requirement

| requirement | legacy | symbolic worst | symbolic best | optimum (bare MDP) |
|---|---|---|---|---|
| `avoid_moving_seg1` | 0.657 | 0.790 | 0.811 | 1.000 |
| `avoid_moving_seg2` | 0.822 | 0.844 | 0.844 | 1.000 |
| `avoid_moving_seg3` | 0.884 | 0.874 | 0.901 | 1.000 |
| `complete_sequence` | 0.101 | 0.163 | 0.165 | 0.985 |
| `goal1` | 0.545 | 0.806 | 0.841 | 1.000 |
| `goal2` | 0.627 | 0.753 | 0.757 | 1.000 |
| `goal3` | 0.562 | 0.861 | 0.861 | 1.000 |
| `seq_1_before_2` | 0.336 | 0.505 | 0.508 | 1.000 |
| `seq_2_before_3` | 0.193 | 0.259 | 0.262 | 1.000 |

## By grid size

| size | legacy success | symbolic success | legacy req. met | symbolic req. met (worst) | legacy output tokens | symbolic output tokens |
|---|---|---|---|---|---|---|
| 4 | 0/4 | 0/4 | 5.25 | 5.50 | 5368 | 5976 |
| 5 | 0/4 | 0/4 | 4.75 | 5.25 | 8034 | 4085 |
| 6 | 0/4 | 1/4 | 4.50 | 7.00 | 11155 | 3822 |
| 7 | 0/4 | 0/4 | 3.25 | 4.00 | 15852 | 5786 |
| 8 | 0/4 | 0/4 | 3.75 | 4.50 | 20339 | 5474 |
