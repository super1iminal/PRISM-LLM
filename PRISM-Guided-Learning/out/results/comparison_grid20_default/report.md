# Legacy vs symbolic: end-to-end comparison

- Legacy run: `out\results\legacy_grid20`
- Symbolic run: `out\results\symbolic_grid20_default`
- Samples: 20

Success = all requirements meet their thresholds (symbolic: in the **worst case** over all completions).

| metric | legacy (per-state) | symbolic (rules) |
|---|---|---|
| success | 0/20 | 0/20 |
| success, of jointly solvable instances | 0/19 | 0/19 |
| success in best case only | n/a | 0/20 |
| shortfall below the achievable (mean) | 3.157 | 1.852 |
| mean requirements met (of 9) | 4.30 | 5.10 (best case 5.10) |
| mean total shortfall below thresholds | 3.162 | 1.857 (best case 1.857) |
| mean iterations | 5.00 | 5.00 |
| mean LLM calls | 15.0 | 5.0 |
| mean output tokens | 12150 | 5783 |
| mean prompt tokens | 24313 | 13034 |
| mean LLM time (s) | 408.1 | 177.1 |
| mean PRISM time (s) | 6.8 | 6.9 |
| mean wall time per sample (s) | 414.9 | 185.5 |
| mean final rules | n/a | 30.4 |
| uncovered situations, last round's policy (mean %) | n/a | 7.0 |
| invalid LLM answers (total) | n/a | 0 |

## Medians [interquartile range] per sample

| metric | legacy | symbolic |
|---|---|---|
| requirements met | 4 [3–5] | 5 [5–6] |
| shortfall | 3.27 [1.95–4.62] | 1.96 [0.39–2.83] |
| output tokens | 11102 [7904–16063] | 5512 [4206–6887] |
| prompt (input) tokens | 23800 [21866–26524] | 12820 [10719–14248] |
| PRISM time (s) | 6.7 [6.7–6.8] | 6.9 [6.9–7.0] |
| wall time (s) | 414 [269–588] | 188 [142–222] |

## Mean final probability per requirement

| requirement | legacy | symbolic worst | symbolic best | optimum (bare MDP) |
|---|---|---|---|---|
| `avoid_moving_seg1` | 0.657 | 0.740 | 0.740 | 1.000 |
| `avoid_moving_seg2` | 0.822 | 0.835 | 0.835 | 1.000 |
| `avoid_moving_seg3` | 0.884 | 0.921 | 0.921 | 1.000 |
| `complete_sequence` | 0.101 | 0.281 | 0.281 | 0.985 |
| `goal1` | 0.545 | 0.788 | 0.788 | 1.000 |
| `goal2` | 0.627 | 0.755 | 0.755 | 1.000 |
| `goal3` | 0.562 | 0.828 | 0.828 | 1.000 |
| `seq_1_before_2` | 0.336 | 0.573 | 0.573 | 1.000 |
| `seq_2_before_3` | 0.193 | 0.373 | 0.373 | 1.000 |

## By grid size

| size | legacy success | symbolic success | legacy req. met | symbolic req. met (worst) | legacy output tokens | symbolic output tokens |
|---|---|---|---|---|---|---|
| 4 | 0/4 | 0/4 | 5.25 | 5.75 | 5368 | 5369 |
| 5 | 0/4 | 0/4 | 4.75 | 5.75 | 8034 | 6397 |
| 6 | 0/4 | 0/4 | 4.50 | 5.25 | 11155 | 6338 |
| 7 | 0/4 | 0/4 | 3.25 | 4.25 | 15852 | 6589 |
| 8 | 0/4 | 0/4 | 3.75 | 4.50 | 20339 | 4221 |
