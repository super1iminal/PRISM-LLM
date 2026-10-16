# Fresh transfer (branch claude/fresh-transfer-5c4595): results

Certified validation members per list (shared `transfer.py`, ids 16-35 / 76-95 / 136-155, 20 per family),
3 independent runs (`final/run1..3`). The ablations reuse the same LLM answers and change only the offline steps;
lists that the ablation leaves unchanged keep their certification (`ablation/` holds the rest, each certified on
its own family's 20 members).

| variant | rooms | crossing | adjacent | mean |
|---|---|---|---|---|
| frozen method: repair + `any` resolved | 19, 19, 19 (19.00) | 20, 20, 20 (20.00) | 19, 20, 19 (19.33) | 19.44 (19.33-19.67) |
| repair, `any` left open | 16, 17, 16 (16.33) | 20, 20, 20 (20.00) | 19, 17, 19 (18.33) | 18.22 (18.00-18.33) |
| no repair, `any` resolved | 19, 9, 19 (15.67) | 9, 20, 20 (16.33) | 19, 20, 19 (19.33) | 17.11 (15.67-19.33) |
| no repair, `any` left open | 16, 8, 16 (13.33) | 2, 16, 16 (11.33) | 19, 17, 19 (18.33) | 14.33 (12.33-17.00) |
| PPO / decision trees / current LLM | 19 / 2 / 8.4 | 0 / 19 / 4.6 | 19 / 16 / 16.0 | 12.67 / 12.33 / 9.67 |

Repair changed the kept list in 4 of 9 cases (run 1 crossing, run 2 rooms and crossing, run 3 crossing).
Cost per repeat: rooms $0.45-0.51, crossing $0.49-0.59, adjacent $0.09-0.18 (Haiku via OpenRouter; 1.0-1.5M tokens
per family, mostly reasoning); the ablations cost nothing extra.

Pac-Man (`pm/trial2_strict`, `pm/run2`, `pm/run3`): 3, 4, 3 of 6 validation horizons (to beat: 4); every kept list
reaches the training optimum 0.5511 on horizons 14, 16, 18.

## Verifier picks the action order each round (`attr/inloop`, one repeat)
Each candidate's `any` resolved by the best of 24 orders on the training members before it is scored and fed back:
rooms 9, crossing 20, adjacent 19 (mean 16.00); frozen method 19.00 / 20.00 / 19.33 (19.44).

## Random grids (`grids/`, grid_20_large.csv, my vocabulary with `hazard` hidden)
One list per training set [0,4,8], [1,5,9], [2,6,10], [3,7,11]; certified of 8 validation grids (ids 12-19):

| repeat | set 0 | set 1 | set 2 | set 3 | mean | cost |
|---|---|---|---|---|---|---|
| 1 | 8 | 8 | 8 | 8 | 8.00 | $0.91 |
| 2 | 8 | 7 | 8 | 8 | 7.75 | $0.87 |
| 3 | 8 | 8 | 8 | 8 | 8.00 | $0.83 |
| mean | | | | | **7.92** (7.75-8.00) | |

To beat: 6.25 (current method), 6.00 (template), 5.50 (PPO). Set 0's kept list never passes all three training
grids (2 of 3, margins -0.034/-0.006/-0.034) yet certifies all 8 validation grids each time.

## MER (`mer/x`, `mer/n`; train ids 0-2, certified of validation ids 3-8)
Explicit-MDP checker (PRISM exports, matches PRISM), same pipeline as the families (8 chains x 6 rounds, repair,
largest margin, `any` resolved by the best of 24 sampled action orders), prompt with the model's named codes.

| benchmark | repeat 1 | repeat 2 | repeat 3 | mean | cost per repeat |
|---|---|---|---|---|---|
| across x (mer_transfer.csv) | 3 | 3 | 3 | 3.00 | $0.44, $0.49, $0.38 |
| across n (mer_n_transfer.csv) | 6 | 0 | 0 | 2.00 | $0.54, $0.51, $0.53 |

Across x every kept list holds the error probability at 0.2, enough only where the threshold is below 0.2 (x <= 0.03:
ids 3, 7, 8); none exploits the x-dependent part (training: x = 0.13 fails in all three repeats). Across n, repeat
1's list is optimal on every training instance and stays optimal on n = 4..15 (6/6); repeats 2 and 3 miss n = 3 in
training and certify nothing.

## Pac-Man v2 (`pm_v2/`): uniform-regret feedback and selection, 3 repeats
After a list reaches the training optimum, the loop keeps revising it with the states (of every training model)
where it crashes most often compared with the best controller; finals selected by lowest mean regret.
Certified of 6 validation horizons: 3, 3, 3 (mean 3.00; horizons 8, 10, 12; 20 fails at 0.5768 each time);
v1 (`pm/`) 3, 4, 3 (3.33); to beat: 4. Cost per repeat $0.22-0.25.

## Random grids without `closer` (`grids_noclose/`, hidden: hazard, closer)
| repeat | set 0 | set 1 | set 2 | set 3 | mean | cost |
|---|---|---|---|---|---|---|
| 1 | 2 | 4 | 5 | 3 | 3.50 | $0.98 |
| 2 | 4 | 4 | 4 | 4 | 4.00 | $1.13 |
| 3 | 3 | 4 | 4 | 2 | 3.25 | $1.08 |
| mean | | | | | **3.58** (3.25-4.00) | |

To beat: 6.00 (template), 5.50 (PPO with `closer`). Not beaten. Without `closer` the only progress sensor is
`toward`, which stalls behind obstacles: many LLM lists pass no training grid, set 0's kept list passes 1 of 3 in
every repeat, and even lists that pass all three training grids (sets 1 and 2) certify only 4-5 of 8. The certification
was also slow (PRISM's Gauss-Seidel fallback on looping policies: up to 33 min per repeat).

## Attribution (families, one repeat each unless noted; certified per list of 20)
| variant | rooms | crossing | adjacent | mean | LLM cost (3 families) |
|---|---|---|---|---|---|
| frozen method (3 repeats) | 19.00 | 20.00 | 19.33 | 19.44 | $1.03-1.25 |
| no repair (3 repeats, same answers) | 15.67 | 16.33 | 19.33 | 17.11 | (same) |
| verifier picks the action order each round | 9 | 20 | 19 | 16.00 | $0.82 |
| feedback without the per-action sensor and risk lines | 19 | 9 | 19 | 15.67 | $1.16 |
| one chain instead of 8 (same stop rule: ended after 16, 6, 2 answers, not 56) | 19 | 18 | 18 | 18.33 | $0.21 |

## Pac-Man v3 (`pm_v3/`): v1 plus numbers on positions and headings (not on steps/moves_left/MAXSTEPS)
Certified with `rules.general = false` (numbers allowed by the parser); my validator rejects any number above 1 in a
comparison that involves anything but xP, yP, dP, xG0, yG0, dG0, xG1, yG1, dG1. Certified of 6 validation horizons:
4, 3, 4 (mean 3.67; horizons 8, 10, 12 always, 20 in repeats 1 and 3 (0.5585, 0.5511); 25 and 30 never).
To beat: 4. Cost per repeat $0.21-0.27.

## Pac-Man v4 (`pm_v4/`): v3 plus early-game regret
After a list reaches the training optimum, the loop keeps revising it with the situations of the first 7 moves of
the 18-move training game (12+ moves left, no ghost removed: the regime of most of a long game) where it crashes most
often compared with the best controller; finals and action orders selected by the lowest such regret.
Certified of 6 validation horizons: 4, 4, 4 (mean 4.00; horizons 8, 10, 12, 20 every time; 25 at 0.64 / 0.70 / 0.77
and 30 at 0.68 / 0.79 / 0.82 fail). To beat: 4 (tied in every repeat, not beaten). Cost per repeat $0.23-0.25.

| Pac-Man variant | repeats | mean |
|---|---|---|
| v1: 0/1 literals only | 3, 4, 3 | 3.33 |
| v2: v1 + uniform regret over all states | 3, 3, 3 | 3.00 |
| v3: numbers on positions and headings | 4, 3, 4 | 3.67 |
| v4: v3 + early-game regret | 4, 4, 4 | 4.00 |

## Random grids without `closer`, coordinates and goal constants allowed (`grids_noclose_pos/`)
Same frozen method with `--position` (x, y, N, G*R, G*C; numbers 0/1). Certified of 8:
3, 4, 4, 4 / 2, 4, 4, 2 / 4, 3, 3, 5 (means 3.75, 3.00, 3.75; overall **3.50**), costs $1.05 / $1.07 / $1.10.
No better than without coordinates (3.58): Haiku hardly used them (14 of 625 valid answers; 1 of 12 kept lists), so
the template's offset logic was never tried. Certification took 21-86 min per repeat (PRISM's fallback solver).

## MER across x with optimal-scheduler feedback (`mer_v2/x`)
Feedback (training-only) shows, on the states the best scheduler or the rules' worst case reaches, where the action
the rules take loses against the optimal one (with optimal continuation) and the optimal actions. Certified of 6:
3, 3, 3 (mean 3.00, ids 3, 7, 8 as before); cost per repeat $0.35-0.38. Training is unchanged: over 316 answers of
both MER-x variants the best value at x = 0.13 is 0.2027 (needs 0.212, optimum 0.2232). Not tried on MER across n.

## Random grids without `closer`, coordinates plus a one-sentence hint (`grids_noclose_hint/`)
The vocabulary text adds: "With these you can compare where the agent is with where the next goal is, e.g. `x < G1R`,
`G1R - x > G1C - y`; the flags tell which goal is next." Certified of 8: 4, 4, 3, 4 / 1, 5, 5, 2 / 4, 4, 4, 4
(means 3.75, 3.25, 4.00; overall **3.67**), costs $0.94 / $0.94 / $0.96. Haiku still rarely used coordinates
(22 of 632 answers; 3 of 12 kept lists).

| grids without `closer` | mean of 8 |
|---|---|
| sensors only (strict) | 3.58 |
| + coordinates and goal constants | 3.50 |
| + coordinates and the hint | 3.67 |
| template / PPO | 6.00 / 5.50 |
